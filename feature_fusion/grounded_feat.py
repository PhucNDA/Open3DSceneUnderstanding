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
import open_clip
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
from ovseg.open_vocab_seg.modeling.clip_adapter.adapter import PIXEL_MEAN, PIXEL_STD
from ovseg.open_vocab_seg.modeling.clip_adapter.utils import crop_with_mask_sam
from detectron2.structures import BitMasks
from ovseg.open_vocab_seg.utils.predictor import OVSegVisualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from ovseg.open_vocab_seg import add_ovseg_config
from ovseg.open_vocab_seg.utils import VisualizationDemo

##############################################

def load_image(image_pil):

    transform = T.Compose(
        [
            # T.RandomResize([800], max_size=1333),
            T.RandomResize([400], max_size=400),
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
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    
    ### Get encoder-decoder multi-head attention weights -- pending
    # use lists to store the outputs via up-values
    conv_features, enc_attn_weights, dec_attn_weights = [], [], []
    
    hooks = [
        model.backbone[-2].register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),
        model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
            lambda self, input, output: enc_attn_weights.append(output[0])
        ),
        model.transformer.decoder.layers[-1].cross_attn.register_forward_hook(
            lambda self, input, output: dec_attn_weights.append(output[0])
        ),
    ]
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
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nqueries, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nqueries, 4)
    
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

    # OV-Seg
    parser.add_argument("--clip_checkpoint", type=str, required=False, help="path to CLIP checkpoint file")
    return parser

def rotate_3d_feature_vector_anticlockwise_90(feature_vector):
    feature_vector = feature_vector.permute(0, 2, 3, 1)
    rotated_vector = feature_vector.permute(0, 2, 1, 3)
    rotated_vector = torch.flip(rotated_vector, dims=(1,))
    
    return rotated_vector.permute(0, 3, 1, 2)

def interpolate_coordinates(t, l, b, r, H, W, H_target, W_target):
    t_new = t * (H_target / H)
    l_new = l * (W_target / W)
    b_new = b * (H_target / H)
    r_new = r * (W_target / W)
    return t_new, l_new, b_new, r_new

def rotate_coordinates_90_ccw(t, l, b, r, H):
    new_t = l
    new_l = H - b
    new_b = r
    new_r = H - t
    return new_t, new_l, new_b, new_r

if __name__ == "__main__":
    
    args = get_parser().parse_args()
    
    outputdir = '../../Dataset/iccvw/ChallengeDevelopmentSet/computed_feature'
    # Scene
    img_folder = 'lowres_wide'
    inputdir = '../../Dataset/iccvw/ChallengeDevelopmentSet/computed_feature'
    arkitscenes_root_dir = "../../Dataset/iccvw" #"PATH/TO/ARKITSCENES/DOWNLOAD/DIR"
    data_type = "ChallengeDevelopmentSet" # or "ChallengeTestSet"
    scene_ids = ['42445173', '42445677', '42445935', '42446478', '42446588'] # Dev Scenes
    scene_id = scene_ids[3]
    gt_data = '../../Dataset/iccvw/ChallengeDevelopmentSet/pcl/'+scene_id+'.pth'
    point_feature_path = inputdir+'/'+scene_id+'_ovseg.pt'

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
    adapter, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained=args.clip_checkpoint)

    # Grouding DINO and SAM
    model = load_model(config_file, grounded_checkpoint, device=device)
    if use_sam_hq:
        predictor = SamPredictor(build_sam_hq(checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
    
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
    
    thresholding_point = 100 # 0 means getting all frames
    granularity = 0.8
    batch_processing = True
    def single_process():
        ### Feature bank
        feat_dim = 768
        interval = 10 # interval consecutive frames default 1: correct lifting
        feat_dim = 768
        sum_features = torch.zeros((point.shape[0], feat_dim))
        counter = torch.zeros((point.shape[0], ))
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
            if idx.shape[0]<100:
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
            
            conf = [float(s.split('(')[1][:-1]) for s in pred_phrases]
            boxes_filt = boxes_filt.cpu()
            # Empty proposals
            if len(conf)==0:
                continue
            # BOX NMS
            boxes_filt, confidence = NMS(boxes_filt, conf, 0.5)
            transformed_boxes = predictor.transform.apply_boxes_torch(torch.stack(boxes_filt), image_sam.shape[:2]).to(device)
            masks, _, _ = predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes.to(device),
                multimask_output = False)
            boxes_filt = torch.stack(boxes_filt)
            for ind in range(boxes_filt.shape[0]):
                n_t,n_l,n_b,n_r = rotate_coordinates_90_ccw(boxes_filt[ind][0], boxes_filt[ind][1], boxes_filt[ind][2], boxes_filt[ind][3], masks.shape[3])
                boxes_filt[ind][0] = n_t
                boxes_filt[ind][1] = n_l
                boxes_filt[ind][2] = n_b
                boxes_filt[ind][3] = n_r
            if masks == None:
                continue
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
            boxes_filt = boxes_filt[selected_indices]

            # masks = [batch, 1, H, W]
            # boxes_filt = [bach, 4]
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
                breakpoint()
            
            #result
            pred_masks = BitMasks(masks.squeeze(1))
            bboxes = pred_masks.get_bounding_boxes()
            mask_fill = [255.0 * c for c in PIXEL_MEAN]

            img = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            
            regions = []
            for bbox, mask in zip(bboxes, pred_masks):
                region, _ = crop_with_mask_sam(img.cuda(),mask,bbox,fill=mask_fill)
                regions.append(region.unsqueeze(0))
            regions = [F.interpolate(r.to(torch.float), size=(224, 224), mode="bicubic") for r in regions]
            pixel_mean = torch.tensor(PIXEL_MEAN).reshape(1, -1, 1, 1)
            pixel_std = torch.tensor(PIXEL_STD).reshape(1, -1, 1, 1)
            imgs = [(r/255.0 - pixel_mean.cuda()) / pixel_std.cuda() for r in regions]
            imgs = torch.cat(imgs)

            img_batches = torch.split(imgs, 32, dim=0)
            class_preds = torch.zeros((imgs.shape[0], len(class_names)))
            image_features = []
            for img_batch in img_batches:
                with torch.no_grad(), torch.cuda.amp.autocast():
                    adapter.cuda()
                    image_feat = adapter.encode_image(img_batch.cuda())
                    image_feat /= image_feat.norm(dim=-1, keepdim=True)
                    image_features.append(image_feat.detach())
            image_features = torch.cat(image_features, dim=0)
            with torch.no_grad(), torch.cuda.amp.autocast():
                adapter.cuda()
                for ind in range(len(class_names)):
                    txts = [class_names[ind], 'others']
                    text = open_clip.tokenize(txts)
                    text_features = adapter.encode_text(text.cuda())
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    class_preds[:, ind]=(100.0 * image_features.half() @ text_features.T).softmax(dim=-1)[:,0]
            
            select_cls = torch.zeros((imgs.shape[0], len(class_names))).to('cuda')
            select_features = torch.zeros(imgs.shape[0], image_features.shape[1]).to('cuda')
            max_scores, select_mask = torch.max(class_preds, dim=0)

            if granularity < 1:
                thresh = 0.7
                select_mask = []
                for ind in range(len(class_names)):
                    cls_pred = class_preds[:,ind]
                    locs = torch.where(cls_pred > thresh)
                    select_mask.extend(locs[0].tolist())
            for ind in select_mask:
                select_cls[ind] = class_preds[ind]
                select_features[ind] = image_features[ind]
            semseg = torch.einsum("qc,qhw->chw", select_cls.float().cuda(), pred_masks.tensor.float().cuda())
            final_feat = torch.einsum("qc,qhw->chw", select_features, pred_masks.tensor.float().cuda())
            r = semseg
            blank_area = (r[0] == 0)
            pred_mask = r.argmax(dim=0).to('cpu')
            pred_mask[blank_area] = 255
            pred_mask = np.array(pred_mask, dtype=np.int)
            visualizer = OVSegVisualizer(image, None, class_names=class_names)
            vis_output = visualizer.draw_sem_seg(pred_mask)
            vis_output.save('testfig/'+str(i)+'.png')
            ### Summing features
            features = final_feat.cpu()
            sum_features[idx] += features[:,mapping[idx][:,[1,2]][:,0], mapping[idx][:,[1,2]][:,1]].permute(1,0)
            positive_points = torch.where(features[:,mapping[idx][:,[1,2]][:,0], mapping[idx][:,[1,2]][:,1]][0,:] != 0)[0] # check for active points
            counter[idx[positive_points]] += 1

            torch.cuda.empty_cache()

        print('Averaging features')
        counter[counter==0] = 1e-5 
        counter_reshaped = counter.unsqueeze(1).expand(-1, sum_features.size(1))
        feat_bank = sum_features/counter_reshaped
        torch.save({"feat": feat_bank.cpu()}, os.path.join(outputdir, scene_id +'_grounded_ov.pt'))
        return None
        
    frames = single_process() # Grounding DINO + SAM

    point_features = torch.load(os.path.join(outputdir, scene_id +'_grounded_ov.pt'))['feat'].cuda()
    predicted_class = torch.zeros((point.shape[0], len(class_names)))
    with torch.no_grad(), torch.cuda.amp.autocast():
        adapter.cuda()
        for ind in range(len(class_names)): 
            txts = [class_names[ind], 'others']
            text = open_clip.tokenize(txts)
            text_features = adapter.encode_text(text.cuda()).cuda()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            predicted_class[:, ind] = (100.0 * point_features.half() @ text_features.T).softmax(dim=-1)[:,0]
        v = viz.Visualizer()
        # There are 2 query class
        color = (color + 1)*127.5
        v.add_points(f'pcl color', point, color, point_size=20, visible=True)
        color[(predicted_class[:,0]>0.7).cpu()]=np.array((255,0,0))
        v.add_points(f'pcl gt_label', point, color, point_size=20, visible=True)
        color[(predicted_class[:,1]>0.7).cpu()]=np.array((0,255,0))
        v.add_points(f'pcl gt_label1', point, color, point_size=20, visible=True)
        v.save('viz')
    

