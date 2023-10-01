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
import nltk
import pyviz3d.visualizer as viz

# Loader Point Cloud mapper, Dataloader, Nouns processing
from benchmark_scripts.opensun3d_challenge_utils import FrameReaderDataLoaderLowRes
from fusion_util import PointCloudToImageMapper, save_fused_feature, get_nouns, rotate_3db_feature_vector_anticlockwise_90, rotate_3d_feature_vector_anticlockwise_90
from fusion_util import PointCloudToImageMapper, save_fused_feature, NMS, mask_nms

# OV Seg
from detectron2.utils.logger import setup_logger
from ovseg_feat import setup_cfg
from ovseg.open_vocab_seg import add_ovseg_config
from ovseg.open_vocab_seg.utils import VisualizationDemo


# Grounding SAM
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from segment_anything import (build_sam,build_sam_hq,SamPredictor)
from grounded_sam3dv2 import load_image, get_grounding_output, load_model

# SegmentAnything3D
import pointops
import random
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from os.path import join
from SegmentAnything3D.util import *
from grounded_sam3dv2 import seg_pcd
############################################

def ovseg(args):
    '''
        OV Seg pipeline extractor
    '''
    outputdir = '../../Dataset/iccvw/ChallengeDevelopmentSet/computed_feature'
    arkitscenes_root_dir = "../../Dataset/iccvw" #"PATH/TO/ARKITSCENES/DOWNLOAD/DIR"
    data_type = "ChallengeDevelopmentSet" # or "ChallengeTestSet"
    # scene_ids = ['42445173', '42445677', '42445935', '42446478', '42446588'] # Dev Scenes
    scene_id = args.scene
    gt_data = '../../Dataset/iccvw/ChallengeDevelopmentSet/pcl/'+scene_id+'.pth'
    
    ### Set up OV-Seg model
    mp.set_start_method("spawn", force=True)
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)
    class_names = get_nouns(args.captions)

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

    ### Point2Image mapper
    visibility_threshold = 0.25 # threshold for the visibility check
    img_dim=(256, 192)
    cut_num_pixel_boundary = 10 # do not use the features on the image boundary
    point2img_mapper = PointCloudToImageMapper(
        image_dim=img_dim, intrinsics=None,
        visibility_threshold=visibility_threshold,
        cut_bound=cut_num_pixel_boundary)
    
    batch_processing = True
    def batch_process():
        global color
        ### Feature bank
        feat_dim = 768
        sum_features = torch.zeros((point.shape[0], feat_dim))
        counter = torch.zeros((point.shape[0], ))
        
        ### Batch process
        mappings = []
        images = []

        interval = 1 # interval consecutive frames
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
            
            n_points = point.shape[0]
            mapping = np.ones([n_points, 4], dtype=int)
            mapping[:, 1:4] = point2img_mapper.compute_mapping(pose, point, depth, intrinsics)
            if mapping[:, 3].sum() == 0: # no points corresponds to this image, skip
                continue

            img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            mappings.append(mapping)
            images.append(img)

        ### Batch forwarding
        images = np.array(images)
        mappings = np.array(mappings)
        batch_size = 16
        num_batches = (images.shape[0] - 1) // batch_size + 1
        
        # get adapter
        _, adapter, _ = demo.run_on_image(images[0:1], class_names, adapter_return=True)

        iter = 0
        for i in trange(num_batches):
            # batch indices
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, images.shape[0])
            predictions=[]
            for class_id in range(len(class_names)): # Independent class names -> not to miss objects if class_name increase exponentially
                prediction, visualized_output = demo.run_on_image(images[start_idx:end_idx], [class_names[class_id], 'others'], adapter_return=False)
                predictions.append(prediction)
            for b in range (end_idx-start_idx):
                for class_id in range(len(class_names)): 
                    prediction = predictions[class_id]
                    coords = rotate_3d_feature_vector_anticlockwise_90(prediction[b]['sem_seg'].permute(1, 2, 0)).permute(2, 0, 1)
                    features = rotate_3d_feature_vector_anticlockwise_90(prediction[b]['feat_seg'].permute(1, 2, 0)).permute(2, 0, 1)
                    features = features.cpu()
                    # 768, 192, 256
                    idx = np.where(mappings[iter][:,3] == 1)[0]
                    sum_features[idx] += features[:,mappings[iter][idx][:,[1,2]][:,0], mappings[iter][idx][:,[1,2]][:,1]].permute(1,0)
                    counter[idx] += 1
                iter += 1
                    
        print('Averaging features')
        counter[counter==0] = 1e-5 
        counter_reshaped = counter.unsqueeze(1).expand(-1, sum_features.size(1))
        feat_bank = sum_features/counter_reshaped
        ### Save Fused Feature
        torch.save({"feat": feat_bank.half().cpu()}, os.path.join(outputdir, scene_id +'_ovseg.pt'))


    if batch_processing == False: 
        pass
    else:
        batch_process()

    return os.path.join(outputdir, scene_id +'_ovseg.pt')        

def groundedsam(args):
    # Scene
    inputdir = '../../Dataset/iccvw/ChallengeDevelopmentSet/computed_feature'
    arkitscenes_root_dir = "../../Dataset/iccvw" #"PATH/TO/ARKITSCENES/DOWNLOAD/DIR"
    data_type = "ChallengeDevelopmentSet" # or "ChallengeTestSet"
    # scene_ids = ['42445173', '42445677', '42445935', '42446478', '42446588'] # Dev Scenes
    scene_id = args.scene
    gt_data = '../../Dataset/iccvw/ChallengeDevelopmentSet/pcl/'+scene_id+'.pth'
    point_feature_path = inputdir+'/'+scene_id+'_ovseg.pt'

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device

    class_names = get_nouns(args.captions)

    text_prompt = ''
    for name in class_names: 
        text_prompt += name + '. ' # Format of Grounding DINO
    text_prompt = text_prompt[:-1]

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
    
    #### Choosing anchors (points) with score > 0.6
    anchor = torch.where(predicted_class>0.6)[0]
    # Free Mem
    adapter, demo, predicted_class = None, None, None
    torch.cuda.empty_cache()
    ### Currently, we are dealing with class-agnostic mask
    # Point where there might have objects
    
    thresholding_point = 0.001 # 0.0 means getting all frames

    batch_processing = True
    def single_process():
        # breakpoint()
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
                masks = rotate_3db_feature_vector_anticlockwise_90(masks)
            
            # Mask NMS
            mk = []
            for mask in masks:
                mk.append(mask[0].cpu())
            mk = np.stack(mk)
            confidence = np.array(confidence)
            selected_indices = mask_nms(mk, confidence, iou_threshold=0.7)
            masks = masks[selected_indices]
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
            frames.append({'info': frame, 'masks': result})
            torch.cuda.empty_cache()

        save_path = '../../Dataset/iccvw/ChallengeDevelopmentSet/computed_mask/'+scene_id+'.pth'
        torch.save(frames, save_path)
        return frames
    
    frames = single_process() # Grounding DINO + SAM
    # frames = torch.load('../../Dataset/iccvw/ChallengeDevelopmentSet/computed_mask/'+scene_id+'.pth')
    seg_pcd(point, frames, args.voxel_size, voxelizer)

def instance_merging(args):
    '''
       Merge Instance
    '''
    # Scene
    inputdir = '../../Dataset/iccvw/ChallengeDevelopmentSet/computed_feature'
    arkitscenes_root_dir = "../../Dataset/iccvw" #"PATH/TO/ARKITSCENES/DOWNLOAD/DIR"
    data_type = "ChallengeDevelopmentSet" # or "ChallengeTestSet"
    # scene_ids = ['42445173', '42445677', '42445935', '42446478', '42446588'] # Dev Scenes
    scene_id = args.scene
    gt_data = '../../Dataset/iccvw/ChallengeDevelopmentSet/pcl/'+scene_id+'.pth'
    point_feature_path = inputdir+'/'+scene_id+'_ovseg.pt'
    instance_path = 'test_scene.pth'

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device

    class_names = get_nouns(args.captions)

    text_prompt = ''
    for name in class_names: 
        text_prompt += name + '. ' # Format of Grounding DINO
    text_prompt = text_prompt[:-1]

    ### Set up ovseg
    mp.set_start_method("spawn", force=True)
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)
    ## CLIP adapter
    adapter = None

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
        breakpoint()
        v.save('viz')

    ### Instance merging processing
    instance = torch.load(instance_path) # [-1, n] 
    n_instance = np.unique(instance).shape[0] # -> instance = [0, n_instance - 2]
    color = (color + 1) * 127.5

    semantic = torch.zeros_like(torch.Tensor(instance))
    semantic -= 1
    instances_result = semantic.clone()
    for ins in range(0, n_instance - 1):
        indices = np.where(instance == ins)[0] # subset of points in the mask
        keep = -1 # -1: No keep, 0, 1, 2...: class_idx
        cmp = torch.zeros(len(class_names))
        for label in range(len(class_names)):
            anchor = torch.where(predicted_class[:, label]>0.6)[0]
            num_valid = np.isin(anchor.cpu().numpy(), indices).sum() # number positive point in the mask
            cmp[label] = num_valid
        
        target = torch.argmax(cmp).item()
        keep = target
        if cmp[target] / indices.shape[0] < 0.1:
            keep = -1
        else:
            semantic[indices] = keep
            instances_result[indices] = ins
    # torch.save(semantic, "test_scene_final.pth")
    torch.save(instances_result, "test_scene_final.pth")
    return None

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for open vocabulary segmentation")
    
    # OV-Seg features
    parser.add_argument("--config-file",default="configs/ovseg_swinB_vitL_demo.yaml",metavar="FILE",help="config file of ov-seg",)
    parser.add_argument("--scene",type=str,required = True,help="scene id in challenge")
    parser.add_argument("--captions",type=str,required = True,help="text caption")
    parser.add_argument("--opts",help="OV-Seg weights",default=[],nargs=argparse.REMAINDER,)

    # Grounding DINO + SAM
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument("--grounded_checkpoint", type=str, required=True, help="path to checkpoint file")
    parser.add_argument("--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file")
    parser.add_argument("--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file")
    parser.add_argument("--use_sam_hq", action="store_true", help="using sam-hq for prediction")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")

    # SegmentAnything3D
    parser.add_argument("--voxel_size", type=float, default=0.05, help="voxel size for SAM3D")
    return parser

if __name__=='__main__':
    
    ### OV-Seg
    args = get_parser().parse_args()
    
    print('----------OVSeg_Features----------')
    # ovseg(args)

    ### Grounding DINO + SAM
    print('----------Grounding DINO and SAM----------')
    groundedsam(args)

    ### Merging instance mask 
    print('----------Instance mask merge----------')
    instance_merging(args)
