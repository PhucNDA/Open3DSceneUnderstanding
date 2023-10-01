import argparse
import os
import copy
import cv2
import matplotlib.pyplot as plt
import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageFont
import argparse
import multiprocessing as mp
import time
import tqdm
import matplotlib.pyplot as plt
from torch.nn import functional as F
from tqdm import tqdm, trange

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    build_sam,
    build_sam_hq,
    SamPredictor
)

# Point mapper
import pyviz3d.visualizer as viz
import open3d as o3d
from benchmark_scripts.opensun3d_challenge_utils import FrameReaderDataLoaderLowRes
from fusion_util import PointCloudToImageMapper, save_fused_feature, NMS, mask_nms
from PIL import Image, ImageDraw

# OV-seg
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from ovseg.open_vocab_seg import add_ovseg_config
from ovseg.open_vocab_seg.utils import VisualizationDemo

# SegmentAnything3D
import pointops
import random
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from os.path import join
from SegmentAnything3D.util import *

##############################################

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_ovseg_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def load_image(image_pil):

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            # T.RandomResize([400], max_size=400),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cuda")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()
    return model.cuda()


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    
    ### Get encoder-decoder multi-head attention weights -- pending
    # use lists to store the outputs via up-values
    # conv_features, enc_attn_weights, dec_attn_weights = [], [], []
    
    # hooks = [
    #     model.backbone[-2].register_forward_hook(
    #         lambda self, input, output: conv_features.append(output)
    #     ),
    #     model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
    #         lambda self, input, output: enc_attn_weights.append(output[0])
    #     ),
    #     model.transformer.decoder.layers[-1].cross_attn.register_forward_hook(
    #         lambda self, input, output: dec_attn_weights.append(output[0])
    #     ),
    # ]
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    # attn_weight =  outputs['weights'].permute(1,0,2,3,4)
    # attn_mem =  outputs['memory']
    # breakpoint()
    # for hook in hooks:
    #     hook.remove()
    # conv_features = conv_features[0]
    # enc_attn_weights = enc_attn_weights[0]
    # dec_attn_weights = dec_attn_weights[0] #(nqueries, 256)
    # ### Hard code here
    # interval = [0, 13400, 16750, 17600] # [i, i+1)
    # for q in range(dec_attn_weights.shape[1]): 
    #     map_value, map_indices = map_to_original_values(dec_attn_weights[0][q].unsqueeze(0).unsqueeze(0), attn_mem, attn_weight)
    #     # Save fig here
    #     numpy_array = (image.cpu() * 255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    #     numpy_array = Image.fromarray(numpy_array)
    #     for value in map_indices:
    #         for i in range(len(interval)):
    #             if value < interval[i]:
    #                 value -= interval[i - 1]
    #                 conv = conv_features[i - 1].tensors
    #                 row = value // conv.shape[2]
    #                 col = value % conv.shape[2]
    #                 larger_row = row * (image.shape[1] - 1) / (conv.shape[2] - 1)
    #                 larger_column = col * (image.shape[2] - 1) / (conv.shape[3] - 1)
    #                 # interpolated_value = cv2.remap(numpy_array, np.array([[larger_column]], dtype=np.float32),
    #                 #                np.array([[larger_row]], dtype=np.float32), interpolation=cv2.INTER_LINEAR)
    #                 draw = ImageDraw.Draw(numpy_array)
    #                 draw.rectangle([(larger_row-10, larger_column-10), (larger_row+10, larger_column+10)], outline=(255,0,0), width=2)
    #     numpy_array.save("output_image.png")
    #     breakpoint()
    #     break
        
        
    
    # probas = outputs["pred_logits"].softmax(-1)[0, :, :-1]
    logits = outputs["pred_logits"].sigmoid()[0]  # (nqueries, 256)
    boxes = outputs["pred_boxes"][0]  # (nqueries, 4)
    

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def map_to_original_values(attended_output, value_tensor, softmax_weights):
    # Calculate attention scores
    attention_scores = torch.matmul(attended_output, value_tensor.permute(1, 2, 0))
    # Apply softmax function
    breakpoint()
    softmax_weights = torch.softmax(attention_scores, dim=-1)
    # Flatten the softmax weights and the value tensor to make indexing easier
    flattened_weights = softmax_weights.view(-1, softmax_weights.size(-1))
    flattened_values = value_tensor.squeeze().expand(flattened_weights.size(0), -1, -1)
    # Multiply weights with flattened values and sum along the flattened dimension
    mapped_output = torch.sum(flattened_weights.unsqueeze(-1) * flattened_values, dim=0)
    # Find indices with the highest attention weights
    topk_score = torch.argsort(flattened_weights[0])[-10:]
    max_attention_indices = torch.argmax(flattened_weights, dim=-1)
    return mapped_output, topk_score

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)


def get_parser():
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    # Grounding DINO + SAM
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument("--grounded_checkpoint", type=str, required=True, help="path to checkpoint file")
    parser.add_argument("--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file")
    parser.add_argument("--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file")
    parser.add_argument("--use_sam_hq", action="store_true", help="using sam-hq for prediction")
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")

    # SegmentAnything3D
    parser.add_argument("--voxel_size", type=float, default=0.05, help="voxel size for SAM3D")

    # OV-Seg
    parser.add_argument("--config-file", default="configs/ovseg_swinB_vitL_demo.yaml", metavar="FILE", help="path to config file ovseg")
    parser.add_argument("--opts",help="Modify config options using the command-line 'KEY VALUE' pairs",default=[],nargs=argparse.REMAINDER)
    return parser

def rotate_3d_feature_vector_anticlockwise_90(feature_vector):
    feature_vector = feature_vector.permute(0, 2, 3, 1)
    rotated_vector = feature_vector.permute(0, 2, 1, 3)
    rotated_vector = torch.flip(rotated_vector, dims=(1,))
    
    return rotated_vector.permute(0, 3, 1, 2)

def make_open3d_point_cloud(input_dict, voxelize, th):
    input_dict["group"] = remove_small_group(input_dict["group"], th)
    # input_dict = voxelize(input_dict)

    xyz = input_dict["coord"]
    if np.isnan(xyz).any():
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


def cal_group(input_dict, new_input_dict, match_inds, ratio=0.5):
    group_0 = input_dict["group"]
    group_1 = new_input_dict["group"]
    group_1[group_1 != -1] += group_0.max() + 1
    
    unique_groups, group_0_counts = np.unique(group_0, return_counts=True)
    group_0_counts = dict(zip(unique_groups, group_0_counts))
    unique_groups, group_1_counts = np.unique(group_1, return_counts=True)
    group_1_counts = dict(zip(unique_groups, group_1_counts))

    # Calculate the group number correspondence of overlapping points
    group_overlap = {}
    for i, j in match_inds:
        group_i = group_1[i]
        group_j = group_0[j]
        if group_i == -1:
            group_1[i] = group_0[j]
            continue
        if group_j == -1:
            continue
        if group_i not in group_overlap:
            group_overlap[group_i] = {}
        if group_j not in group_overlap[group_i]:
            group_overlap[group_i][group_j] = 0
        group_overlap[group_i][group_j] += 1

    # Update group information for point cloud 1
    for group_i, overlap_count in group_overlap.items():
        # for group_j, count in overlap_count.items():
        max_index = np.argmax(np.array(list(overlap_count.values())))
        group_j = list(overlap_count.keys())[max_index]
        count = list(overlap_count.values())[max_index]
        total_count = min(group_0_counts[group_j], group_1_counts[group_i]).astype(np.float32)
        # print(count / total_count)
        if count / total_count >= ratio:
            group_1[group_1 == group_i] = group_j
    return group_1

def cal_2_scenes(pcd_list, index, voxel_size, voxelize, th=5):
    if len(index) == 1:
        return(pcd_list[index[0]])
    # print(index, flush=True)
    input_dict_0 = pcd_list[index[0]]
    input_dict_1 = pcd_list[index[1]]
    pcd0 = make_open3d_point_cloud(input_dict_0, voxelize, th)
    pcd1 = make_open3d_point_cloud(input_dict_1, voxelize, th)
    if pcd0 == None:
        if pcd1 == None:
            return None
        else:
            return input_dict_1
    elif pcd1 == None:
        return input_dict_0

    # Cal Dul-overlap
    pcd0_tree = o3d.geometry.KDTreeFlann(copy.deepcopy(pcd0))
    match_inds = get_matching_indices(pcd1, pcd0_tree, 1.5 * voxel_size, 1)
    pcd1_new_group = cal_group(input_dict_0, input_dict_1, match_inds)
    # print(pcd1_new_group)

    pcd1_tree = o3d.geometry.KDTreeFlann(copy.deepcopy(pcd1))
    match_inds = get_matching_indices(pcd0, pcd1_tree, 1.5 * voxel_size, 1)
    input_dict_1["group"] = pcd1_new_group
    pcd0_new_group = cal_group(input_dict_1, input_dict_0, match_inds)
    # print(pcd0_new_group)

    pcd_new_group = np.concatenate((pcd0_new_group, pcd1_new_group), axis=0)
    pcd_new_group = num_to_natural(pcd_new_group)
    pcd_new_coord = np.concatenate((input_dict_0["coord"], input_dict_1["coord"]), axis=0)
    pcd_new_color = np.concatenate((input_dict_0["color"], input_dict_1["color"]), axis=0)
    pcd_dict = dict(coord=pcd_new_coord, color=pcd_new_color, group=pcd_new_group)

    pcd_dict = voxelize(pcd_dict)
    return pcd_dict

def get_sam(image, masks):
    group_ids = np.full((image.shape[0], image.shape[1]), -1, dtype=int)
    if masks == None: # If frame has no corresponding mask
        return group_ids
    num_masks = len(masks)
    group_counter = 0
    for i in reversed(range(num_masks)):
        # print(masks[i]["predicted_iou"])
        group_ids[masks[i][0].cpu()] = group_counter
        group_counter += 1
    return group_ids

def get_pcd(point, frame, masks):
    depth_intrinsic = frame['intrinsics']
    pose = frame['pose']
    depth_image = frame['depth']
    color_image = frame['image']
    mask = (depth_image != 0)

    group_ids = get_sam(color_image, masks)
    color_image = np.reshape(color_image[mask], [-1,3])
    group_ids = group_ids[mask]
    group_ids = num_to_natural(group_ids)
    
    subsample = 1

    intrinsic_4x4 = np.identity(4)
    intrinsic_4x4[:3, :3] = depth_intrinsic

    u, v = np.meshgrid(
        range(0, depth_image.shape[1], subsample),
        range(0, depth_image.shape[0], subsample),
    )
    d = depth_image[v, u]
    d_filter = d != 0
    mat = np.vstack(
        (
            u[d_filter] * d[d_filter],
            v[d_filter] * d[d_filter],
            d[d_filter],
            np.ones_like(u[d_filter]),
        )
    )
    new_points_3d = np.dot(np.linalg.inv(intrinsic_4x4), mat)[:3]
    new_points_3d_padding = np.vstack((new_points_3d, np.ones((1, new_points_3d.shape[1]))))
    world_coord_padding = np.dot(pose, new_points_3d_padding)
    new_points_3d = world_coord_padding[:3]


    group_ids = num_to_natural(group_ids)
    save_dict = dict(coord=new_points_3d.T, color=color_image, group=group_ids)

    return save_dict

def seg_pcd(point, frames, voxel_size, voxelizer,scene_id, th=5):
    pcd_list = []
    # cnt = 0 
    # for component in frames:
    #     frame = component['info']
    #     masks = component['masks']
    #     if masks ==None:
    #         cnt += 1
    for component in frames:
        frame = component['info']
        masks = component['masks']
        pcd_dict = get_pcd(point, frame, masks)
        if len(pcd_dict["coord"]) == 0:
            continue
        pcd_dict = voxelizer(pcd_dict)
        pcd_list.append(pcd_dict)
    while len(pcd_list) > 1:
        print(len(pcd_list), flush=True)
        new_pcd_list = []
        for indice in pairwise_indices(len(pcd_list)):
            # print(indice)
            pcd_frame = cal_2_scenes(pcd_list, indice, voxel_size=voxel_size, voxelize=voxelizer, th=th)
            if pcd_frame is not None:
                new_pcd_list.append(pcd_frame)
        pcd_list = new_pcd_list
    seg_dict = pcd_list[0]
    seg_dict["group"] = num_to_natural(remove_small_group(seg_dict["group"], th))

    scene_coord = torch.tensor(point).cuda().contiguous()
    new_offset = torch.tensor(scene_coord.shape[0]).cuda()
    gen_coord = torch.tensor(seg_dict["coord"]).cuda().contiguous().float()
    offset = torch.tensor(gen_coord.shape[0]).cuda()
    gen_group = seg_dict["group"]
    indices, dis = pointops.knn_query(1, gen_coord, offset, scene_coord.to(torch.float32), new_offset)
    indices = indices.cpu().numpy()
    group = gen_group[indices.reshape(-1)].astype(np.int16)
    mask_dis = dis.reshape(-1).cpu().numpy() > 0.6
    group[mask_dis] = -1
    group = group.astype(np.int16)
    torch.save(num_to_natural(group), scene_id)

if __name__ == "__main__":
    
    args = get_parser().parse_args()
    
    # Scene
    img_folder = 'lowres_wide'
    inputdir = '../../Dataset/iccvw/ChallengeDevelopmentSet/computed_feature'
    arkitscenes_root_dir = "../../Dataset/iccvw" #"PATH/TO/ARKITSCENES/DOWNLOAD/DIR"
    data_type = "ChallengeDevelopmentSet" # or "ChallengeTestSet"
    scene_ids = ['42445173', '42445677', '42445935', '42446478', '42446588'] # Dev Scenes
    scene_id = scene_ids[3]
    gt_data = '../../Dataset/iccvw/ChallengeDevelopmentSet/pcl/'+scene_id+'.pth'
    point_feature_path = inputdir+'/'+scene_id+'_grounded_ov.pt'

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    text_prompt = args.text_prompt # This should be Nouns
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device

    class_names = text_prompt.split('.')
    class_names.remove('')
    class_names = [name.strip() for name in class_names]

    ### Set up ovseg
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)
    ## CLIP adapter
    adapter = None

    # Grouding DINO and SAM
    model = load_model(config_file, grounded_checkpoint, device=device)
    if use_sam_hq:
        predictor = SamPredictor(build_sam_hq(checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
    
    # Segment Anything 3D
    voxelizer = Voxelize(voxel_size=args.voxel_size, mode="train", keys=("coord", "color", "group"))

    ### Set up dataloader
    data_root_dir = os.path.join(arkitscenes_root_dir, data_type)
    scene_dir = os.path.join(data_root_dir, scene_id)
    print(os.path.abspath(scene_dir))
    try:
        loader = FrameReaderDataLoaderLowRes(root_path=scene_dir)
    except FileNotFoundError:
        print('>>> Error: Data not found. Did you download it?')
        print('>>> Try: python download_data_opensun3d.py --data_type=challenge_development_set --download_dir=data')
        exit()

    print("Number of frames to load: ", len(loader))
    point, color = torch.load(gt_data)
    point_features = torch.load(point_feature_path)['feat']

    ### Point2Image mapper
    visibility_threshold = 0.25 # threshold for the visibility check
    img_dim=(256, 192)
    cut_num_pixel_boundary = 10 # do not use the features on the image boundary
    point2img_mapper = PointCloudToImageMapper(
        image_dim=img_dim, intrinsics=None,
        visibility_threshold=visibility_threshold,
        cut_bound=cut_num_pixel_boundary)
    
    ### Features similarity
    _, adapter, _ = demo.run_on_image(np.zeros((1, img_dim[1], img_dim[0], 3)), ['None'], adapter_return=True)
    text_features = adapter.get_text_features(class_names)
    
    ### dependent softmax others include
    predicted_class = F.softmax(adapter.get_sim_logits(text_features, point_features.cuda().to(torch.float32))[...,:-1],dim=-1)
    
    ## independent softmax
    for i in range(len(class_names)): # Don't get others
        tmp=[class_names[i], 'others']
        predicted_class[:,i] = F.softmax(adapter.get_sim_logits(adapter.get_text_features(tmp),point_features.cuda().to(torch.float32))[...,:-1], dim=-1)[...,0]
    if False: # Visualization
        v = viz.Visualizer()
        color = (color + 1) * 127.5
        # There are 2 query class
        v.add_points(f'pcl color', point, color, point_size=20, visible=True)
        color[(predicted_class[:,0]>0.6).cpu()]=np.array((255,0,0))
        v.add_points(f'pcl gt_label', point, color, point_size=20, visible=True)
        color[(predicted_class[:,1]>0.6).cpu()]=np.array((0,255,0))
        v.add_points(f'pcl gt_label1', point, color, point_size=20, visible=True)
        # color[(predicted_class[:,2]>0.7).cpu()]=np.array((0,0,255))
        # v.add_points(f'pcl gt_label2', point, color, point_size=20, visible=True)
        # color[(predicted_class[:,3]>0.7).cpu()]=np.array((65,150,127))
        # v.add_points(f'pcl gt_label3', point, color, point_size=20, visible=True)
        breakpoint()
        v.save('viz')
    
    anchor = torch.where(predicted_class>0.6)[0]
    # Free Mem
    adapter, demo, predicted_class = None, None, None
    torch.cuda.empty_cache()
    ### Currently, we are dealing with class-agnostic mask
    # Point where there might have objects
    

    thresholding_point = 0.001 # 0.0 means getting all frames

    batch_processing = True
    def single_process():
        ### Feature bank
        feat_dim = 768
        interval = 1 # interval consecutive frames default 1: correct lifting
        frames=[] # Get all frame
        for i in trange(0, len(loader),interval):
            frame = loader[i]
            frame_id = frame['frame_id']  # str
            depth = frame['depth']  # (h, w)
            image = frame['image']  # (h, w, 3), [0-255], uint8
            image_path = frame['image_path']  # str
            intrinsics = frame['intrinsics']  # (3,3) - camera intrinsics
            pose = frame['pose']  # (4,4) - camera pose
            pcd = frame['pcd']  # (n, 3) - backprojected point coordinates in world coordinate system! not camera coordinate system! backprojected and transformed to world coordinate system
            pcd_color = frame['color']  # (n,3) - color values for each point
            
            ### Point mapping
            n_points = point.shape[0]
            mapping = np.ones([n_points, 4], dtype=int)
            mapping[:, 1:4] = point2img_mapper.compute_mapping(pose, point, depth, intrinsics)
            if mapping[:, 3].sum() == 0: # no points corresponds to this image, skip sure
                continue
            # indices of projectable pixels
            idx = np.where(mapping[:,3] == 1)[0]
            
            # visible points on 2D image
            valid_point = np.where(mapping[anchor.cpu()][:,3]==1)[0]
            # -> mapping[anchor.cpu()][valid_point] are projected semantic heated pixels from heated points
            if valid_point.shape[0] < thresholding_point*anchor.shape[0]:
                frames.append({'info': frame, 'masks': None})
                continue

            #### Processing DINO ####
            image_pil = Image.open(image_path).convert("RGB")
            rotated_pil = Image.fromarray(np.rot90(np.array(image_pil),k=-1))
            image_pil, image_infer = load_image(rotated_pil)
            boxes_filt, pred_phrases = get_grounding_output(model, image_infer, text_prompt, box_threshold, text_threshold, device=device)
            
            #### Segment Anything ####
            image_sam = cv2.imread(image_path)
            image_sam = cv2.cvtColor(image_sam, cv2.COLOR_BGR2RGB)
            image_sam = cv2.rotate(image_sam, cv2.ROTATE_90_CLOCKWISE)
            predictor.set_image(image_sam)
            size = image_pil.size
            H, W = size[1], size[0]
            for j in range(boxes_filt.size(0)):
                boxes_filt[j] = boxes_filt[j] * torch.Tensor([W, H, W, H])
                boxes_filt[j][:2] -= boxes_filt[j][2:] / 2
                boxes_filt[j][2:] += boxes_filt[j][:2]
            
            result = None
            conf = [float(s.split('(')[1][:-1]) for s in pred_phrases]
            boxes_filt = boxes_filt.cpu()
            # Empty proposals
            if len(conf)==0:
                masks = None
                frames.append({'info': frame, 'masks': result})
                continue
            # BOX NMS
            boxes_filt, confidence = NMS(boxes_filt, conf, 0.5)
            transformed_boxes = predictor.transform.apply_boxes_torch(torch.stack(boxes_filt), image_sam.shape[:2]).to(device)
            masks, _, _ = predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes.to(device),
                multimask_output = False)
            
            if masks == None:
                pass
            # Rotate back the mask
            else:
                masks = rotate_3d_feature_vector_anticlockwise_90(masks)
            
            # Mask NMS
            mk = []
            for mask in masks:
                mk.append(mask[0].cpu())
            mk = np.stack(mk)
            confidence = np.array(confidence)
            selected_indices = mask_nms(mk, confidence, iou_threshold=0.7)
            masks = masks[selected_indices]
            ### Visualization of predicted mask --Test: passed
            if False:
                # draw output image
                plt.figure(figsize=(10, 10))
                plt.imshow(image)
                for mask in masks:
                    show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
                for box, label in zip(boxes_filt, pred_phrases):
                    show_box(box.numpy(), plt.gca(), label)
                plt.axis('off')
                plt.savefig(os.path.join("testfig/grounded_sam_output"+str(i)+'.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)
                continue            
            if False:
                # original image
                img = Image.fromarray(image)
                for mask in masks:
                    draw = ImageDraw.Draw(img)
                    pos = torch.where(mask[0] == True)
                    for i in range(pos[0].shape[0]):
                        iter2 = pos[1][i]
                        iter1 = pos[0][i]
                        draw.rectangle([iter2-2,iter1-2,iter2+2,iter1+2], outline='red', width=1)
                img.save('grounded_sam_output.jpg')
            ##### Mask filtering -> filter out unrelated masks - mask lies outside of the region - class agnostic
            # mapping[anchor.cpu()][valid_point][:,[1,2]] ->[npixels, 2] = (img_dim[1], img_dim[0])
            # masks -> [n,1, img_dim[1], img_dim[0]]

            if masks != None:
                result = []
                maskfilter_thresh = 0.3
                maskvalid_thresh = 100
                for mask_ in masks:
                    mask=mask_[0]
                    mask_pix = torch.where(mask==True) # max (192,256)
                    XY = torch.stack(mask_pix).permute(1,0).cpu()
                    total_mask = mask_pix[0].shape[0]
                    tracer = mapping[anchor.cpu()][valid_point][:,[1,2]] # max: (192,256)
                    num_valid = np.sum(np.all(np.isin(XY.numpy(), tracer), axis=1))
                    num_invalid = XY.shape[0] - num_valid
                    # if num_valid/total_mask < maskfilter_thresh and num_invalid > maskvalid_thresh:
                        # continue
                    if num_invalid > maskvalid_thresh: 
                        continue
                    result.append(mask_)
                
                if len(result) == 0: # empty frame
                    result = None
                else:
                    result = torch.stack(result)
                #######
                # Visualization of a single image
                if False:
                    # draw output image
                    plt.figure(figsize=(10, 10))
                    plt.imshow(image)
                    for mask in masks:
                        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
                    for box, label in zip(boxes_filt, pred_phrases):
                        show_box(box.numpy(), plt.gca(), label)
                    plt.axis('off')
                    plt.savefig(os.path.join("testfig/grounded_sam_output"+str(i)+'.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)
                    # breakpoint()
            
            frames.append({'info': frame, 'masks': result})
            torch.cuda.empty_cache()

        save_path = '../../Dataset/iccvw/ChallengeDevelopmentSet/computed_mask/'+scene_id+'.pth'
        torch.save(frames, save_path)
        return frames
        
    frames = single_process() # Grounding DINO + SAM
    frames = torch.load('../../Dataset/iccvw/ChallengeDevelopmentSet/computed_mask/'+scene_id+'.pth')
    seg_pcd(point, frames, args.voxel_size, voxelizer)


    

