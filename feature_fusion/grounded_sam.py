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
from fusion_util import PointCloudToImageMapper, save_fused_feature
from PIL import Image, ImageDraw

# OV-seg
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from ovseg.open_vocab_seg import add_ovseg_config
from ovseg.open_vocab_seg.utils import VisualizationDemo

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
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

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
    parser.add_argument("--config-file", default="configs/ovseg_swinB_vitL_demo.yaml", metavar="FILE", help="path to config file ovseg")
    parser.add_argument("--opts",help="Modify config options using the command-line 'KEY VALUE' pairs",default=[],nargs=argparse.REMAINDER)
    return parser

def rotate_3d_feature_vector_anticlockwise_90(feature_vector):
    feature_vector = feature_vector.permute(0, 2, 3, 1)
    rotated_vector = feature_vector.permute(0, 2, 1, 3)
    rotated_vector = torch.flip(rotated_vector, dims=(1,))
    
    return rotated_vector.permute(0, 3, 1, 2)

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
    predicted_class = F.softmax(adapter.get_sim_logits(text_features, point_features.cuda().to(torch.float32))[...,:-1],dim=-1)
    
    # Visualization PCL: passed
    if False:
        v = viz.Visualizer()
        color = (color + 1) * 127.5 
        v.add_points(f'pcl color', point, color, point_size=20, visible=True)
        color[(predicted_class[:,0]>0.8).cpu()]=np.array((255,0,0))
        v.add_points(f'pcl gt_label', point, color, point_size=20, visible=True)
        color[(predicted_class[:,1]>0.8).cpu()]=np.array((0,255,0))
        v.add_points(f'pcl gt_label1', point, color, point_size=20, visible=True)
        v.save('viz')
        breakpoint()

    ### Currently, we are dealing with class-agnostic mask
    # Point where there might have objects
    anchor = torch.where(predicted_class>0.8)[0]
    
    thresholding_point = 0.25

    batch_processing = True
    def single_process():
        ### Feature bank
        feat_dim = 768
        sum_features = torch.zeros((point.shape[0], feat_dim))
        counter = torch.zeros((point.shape[0], ))
        interval = 100 # interval consecutive frames default 10
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
            if mapping[:, 3].sum() == 0: # no points corresponds to this image, skip
                continue
            # indices of projectable pixels
            idx = np.where(mapping[:,3] == 1)[0]
            
            valid_point = np.where(mapping[anchor.cpu()][:,3]==1)[0]
            if valid_point.shape[0] < thresholding_point*anchor.shape[0]:
                continue

            #### Processing DINO ####
            image_pil = Image.open(image_path).convert("RGB")
            rotated_pil = Image.fromarray(np.rot90(np.array(Image.open(image_path).convert("RGB")),k=-1))
            image_pil, image_infer = load_image(rotated_pil)
            boxes_filt, pred_phrases = get_grounding_output(model, image_infer, text_prompt, box_threshold, text_threshold, device=device)
            
            #### Segment Anything ####
            image_sam = cv2.imread(image_path)
            image_sam = cv2.cvtColor(image_sam, cv2.COLOR_BGR2RGB)
            image_sam = cv2.rotate(image_sam, cv2.ROTATE_90_CLOCKWISE)
            predictor.set_image(image_sam)
            size = image_pil.size
            H, W = size[1], size[0]
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]

            boxes_filt = boxes_filt.cpu()
            transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_sam.shape[:2]).to(device)
            masks, _, _ = predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes.to(device),
                multimask_output = False)
            
            # Rotate back the mask
            masks = rotate_3d_feature_vector_anticlockwise_90(masks)
            
            ### Visualization of predicted mask --Test: passed
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
            #####



    single_process()



    

