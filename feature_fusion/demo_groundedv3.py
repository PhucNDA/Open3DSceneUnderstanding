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
import matplotlib.pyplot as plt
from torch.nn import functional as F
from tqdm import tqdm, trange
import nltk
import pyviz3d.visualizer as viz
from scipy.spatial import ConvexHull

# Loader Point Cloud mapper, Dataloader, Nouns processing
from benchmark_scripts.opensun3d_challenge_utils import FrameReaderDataLoaderLowRes
from fusion_util import PointCloudToImageMapper, save_fused_feature, get_nouns, rotate_3db_feature_vector_anticlockwise_90, rotate_3d_feature_vector_anticlockwise_90
from fusion_util import PointCloudToImageMapper, save_fused_feature, NMS, mask_nms
from dataset.voxelizer import Voxelizer

# OV Seg
from detectron2.utils.logger import setup_logger
from ovseg_feat import setup_cfg
from ovseg.open_vocab_seg import add_ovseg_config
from ovseg.open_vocab_seg.utils import VisualizationDemo
from feature_fusion.grounded_feat import rotate_coordinates_90_ccw
import open_clip
from ovseg.open_vocab_seg.modeling.clip_adapter.adapter import PIXEL_MEAN, PIXEL_STD
from ovseg.open_vocab_seg.modeling.clip_adapter.utils import crop_with_mask_sam
from detectron2.structures import BitMasks
from ovseg.open_vocab_seg.utils.predictor import OVSegVisualizer
from detectron2.utils.visualizer import ColorMode

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

def minimum_area_bounding_rectangle(points):
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    hull_edges = hull_points[1:] - hull_points[:-1]
    
    angles = np.arctan2(hull_edges[:, 1], hull_edges[:, 0])
    min_angle = np.min(angles)
    max_angle = np.max(angles)
    
    cosines = np.cos(angles)
    sines = np.sin(angles)
    
    rotated_points = np.dot(hull_points - np.mean(hull_points, axis=0), np.array([[cosines, -sines], [sines, cosines]]))
    
    min_proj = np.min(rotated_points, axis=0)
    max_proj = np.max(rotated_points, axis=0)
    
    width = max_proj[1] - min_proj[1]
    length = np.linalg.norm(hull_edges[np.argmax(angles) - np.argmin(angles)])
    
    center = np.mean(hull_points, axis=0)
    angle = (min_angle + max_angle) / 2
    
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    corners = [
        center + np.dot(rotation_matrix, np.array([-length / 2, -width / 2])),
        center + np.dot(rotation_matrix, np.array([-length / 2, width / 2])),
        center + np.dot(rotation_matrix, np.array([length / 2, width / 2])),
        center + np.dot(rotation_matrix, np.array([length / 2, -width / 2]))
    ]
    
    return corners
###############################

"""
This version, we use pure CLIP encoder to extract box, not mask as v2
"""
CLIP_THRESH_1 = 0.7
CLIP_THRESH_2 = 0.8
CLIP_THRESH_3 = 0.9
def grounded_features(args):
    ''' 
        Feature aggregation
    '''
    experiment = args.exp
    datapath = args.datapath
    exp_path = os.path.join(datapath, experiment)
    computed_feature = os.path.join(exp_path, 'computed_feature')
    computed_feature1 = os.path.join(exp_path, 'computed_feature1')
    computed_mask = os.path.join(exp_path, 'computed_mask')
    sam3d = os.path.join(exp_path, 'sam3d')
    mergedsam3d = os.path.join(exp_path, 'mergedsam3d')
    pclpath = os.path.join(datapath, 'pcl')
    
    # Scene
    data_type = "ChallengeTestSet" # or "ChallengeTestSet"
    scene_id = args.scene
    gt_data = os.path.join(pclpath, scene_id+'.pth')
    point_feature_path = os.path.join(computed_feature,scene_id+'_grounded_ov.pt')
    scene_dir = os.path.join(datapath, scene_id)

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device

    # prompt processing
    args.captions = args.captions.replace('_',' ')
    args.classname = args.classname.replace('_',' ')
    text_prompt = args.captions
    class_names = args.classname.split('.') # split fullstop

    ### Set up ovseg
    adapter, _, preprocess = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.clip_checkpoint)
    
    ### Set up dataloader
    print(os.path.abspath(scene_dir))
    try:
        loader = FrameReaderDataLoaderLowRes(root_path=scene_dir)
    except FileNotFoundError:
        print('>>> Error: Data not found. Did you download it?')
        print('>>> Try: python download_data_opensun3d.py --data_type=challenge_development_set --download_dir=data')
        exit()

    print("Number of frames to load: ", len(loader))
    # Point Voxelization
    point, color = torch.load(gt_data)
    # voxelizer = Voxelizer(voxel_size=args.voxel_size,clip_bound=None)
    # point, color, _,  v2p, p2v = voxelizer.voxelize(point, color, None, center = None, link = None, return_ind=True)

    ### Point2Image mapper
    img_dim=(256, 192)
    cut_num_pixel_boundary = 10 # do not use the features on the image boundary
    point2img_mapper = PointCloudToImageMapper(
        image_dim=img_dim, intrinsics=None,
        cut_bound=cut_num_pixel_boundary)
    
    # Segment Anything 3D
    voxelizer = Voxelize(voxel_size=args.voxel_size, mode="train", keys=("coord", "color", "group"))
    
    granularity = 0.8
    batch_processing = True
    def single_process(point, color):
        # Grouding DINO and SAM
        model = load_model(config_file, grounded_checkpoint, device=device)
        predictor = SamPredictor(build_sam_hq(checkpoint=sam_checkpoint).to(device))
        
        ### Feature bank
        interval = args.interval # interval consecutive frames default 1: correct lifting
        feat_dim = args.feat_dim #bigG = 1280
        sum_features = torch.zeros((point.shape[0], feat_dim)).cuda()
        counter = torch.zeros((point.shape[0], )).cpu()
        total_mask=[]
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
            mapping = torch.ones([n_points, 4], dtype=int).cuda()
            mapping[:, 1:4] = torch.tensor(point2img_mapper.compute_mapping(pose, point, depth, intrinsics)).cuda()
            # indices of projectable pixels
            idx = torch.where(mapping[:,3] == 1)[0]

            # no points corresponds to this image, visible points on 2D image
            if mapping[:, 3].sum() == 0 or idx.shape[0]<100: 
                total_mask.append(None)
                continue

            #### Processing DINO ####
            image_pil = Image.open(image_path).convert("RGB")
            rotated_pil = image_pil.copy()
            for ntime in range(int(args.rotate)):
                rotated_pil = Image.fromarray(np.rot90(np.array(rotated_pil),k=-1))
            image_pil, image_infer = load_image(rotated_pil)
            boxes_filt, pred_phrases = get_grounding_output(model, image_infer, text_prompt, box_threshold, text_threshold, device=device)
            
            #### Segment Anything ####
            image_sam = cv2.imread(image_path)
            image_sam = cv2.cvtColor(image_sam, cv2.COLOR_BGR2RGB)
            for ntime in range(int(args.rotate)):
                image_sam = cv2.rotate(image_sam, cv2.ROTATE_90_CLOCKWISE)

            predictor.set_image(image_sam)
            size = image_pil.size
            H, W = size[1], size[0]
            target_id_valid = []
            for j in range(boxes_filt.size(0)):
                boxes_filt[j] = boxes_filt[j] * torch.Tensor([W, H, W, H]).cuda()
                boxes_filt[j][:2] -= boxes_filt[j][2:] / 2
                boxes_filt[j][2:] += boxes_filt[j][:2]
                l, t, r, b = int(boxes_filt[j][0].item()), int(boxes_filt[j][1].item()), int(boxes_filt[j][2].item()), int(boxes_filt[j][3].item())
                l = max(l, 0)
                t = max(t, 0)
                r = min(r, W)
                b = min(b, H)
                # meaningful box
                if b - t > 1 and r - l > 1:
                    target_id_valid.append(j)
            boxes_filt = boxes_filt[target_id_valid]
            pred_phrases = [pred_phrases[dex] for dex in target_id_valid]

            conf = torch.Tensor([float(s.split('(')[1][:-1]) for s in pred_phrases])
            # Extract name and process distractor
            names = [s.split('(')[0] for s in pred_phrases]
            target_id = [i for i, item in enumerate(names) if args.distractor != item.lower()] # previously was 'not in'
            pred_phrases = [pred_phrases[_] for _ in target_id]
            boxes_filt = boxes_filt[target_id].cpu()
            conf = conf[target_id]
            conf = conf.tolist()
            # Empty proposals
            if len(conf)==0:
                total_mask.append(None)
                continue
            # BOX NMS
            # boxes_filt, confidence = NMS(boxes_filt, conf, 0.5)
            boxes_filt, confidence, pred_phrases = NMS(boxes_filt, conf, pred_phrases,  0.1)
            boxes_filt = torch.stack(boxes_filt)
            transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_sam.shape[:2]).to(device)
            masks, _, _ = predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes.to(device),
                multimask_output = False)
            
            # Empty masks
            if masks == None:
                total_mask.append(None)
                continue
            # masks = [batch, 1, H, W]
            # boxes_filt = [bach, 4]
            pred_masks = BitMasks(masks.squeeze(1))
            bboxes = pred_masks.get_bounding_boxes() # ->mask box ### careful

            # Cropped regions and mask fitting
            masks_fitted = torch.zeros_like(masks, dtype=bool)
            regions = []
            iter = 0          
            for box in boxes_filt:
                l, t, r, b = int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item())
                l = max(l, 0)
                t = max(t, 0)
                r = min(r, W)
                b = min(b, H)
                masks_fitted[iter, 0, t:b, l:r] = True
                iter += 1
                regions.append(preprocess(Image.fromarray(np.array(image_sam)[t:b, l:r,:])))
            masks= torch.logical_and(masks, masks_fitted) # fitting
            pred_masks = BitMasks(masks.squeeze(1))

            imgs = torch.stack(regions)
            img_batches = torch.split(imgs, 32, dim=0)
            class_preds = torch.zeros((imgs.shape[0], len(class_names)))
            image_features = []
            with torch.no_grad(), torch.cuda.amp.autocast():
                adapter.cuda()
                for img_batch in img_batches:
                    image_feat = adapter.encode_image(img_batch.cuda())
                    image_feat /= image_feat.norm(dim=-1, keepdim=True)
                    image_features.append(image_feat.detach())
            image_features = torch.cat(image_features, dim=0)
            with torch.no_grad(), torch.cuda.amp.autocast():
                for ind in range(len(class_names)):
                    txts = [class_names[ind], 'other']
                    text = open_clip.tokenize(txts)
                    text_features = adapter.encode_text(text.cuda())
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    class_preds[:, ind]=(100.0 * image_features.half() @ text_features.T).softmax(dim=-1)[:,0]

            select_cls = torch.zeros((imgs.shape[0], len(class_names))).to('cuda')
            select_features = torch.zeros(imgs.shape[0], image_features.shape[1]).to('cuda')

            if granularity < 1:
                select_mask = []
                for ind in range(len(class_names)):
                    cls_pred = class_preds[:,ind]
                    locs = torch.where(cls_pred > CLIP_THRESH_1)
                    select_mask.extend(locs[0].tolist())
                for ind in select_mask:
                    select_cls[ind] = class_preds[ind]
                    select_features[ind] = image_features[ind]
            final_feat = torch.einsum("qc,qhw->chw", select_features, pred_masks.tensor.float().cuda())
            # semseg = torch.einsum("qc,qhw->chw", select_cls.float().cuda(), pred_masks.tensor.float().cuda())
            # r = semseg
            # blank_area = (r[0] == 0)
            # pred_mask = r.argmax(dim=0).to('cpu')
            # pred_mask[blank_area] = 255
            # pred_mask = np.array(pred_mask, dtype=np.int)
            # visualizer = OVSegVisualizer(image, None, class_names=class_names)
            # vis_output = visualizer.draw_sem_seg(pred_mask)
            # vis_output.save('testfig/'+str(i)+'.png')
            ### Summing features
            features = final_feat.unsqueeze(0)
            for ntime in range(int(args.rotate)):
                features = rotate_3db_feature_vector_anticlockwise_90(features)
            features = features.squeeze(0)
            sum_features[idx] += features[:,mapping[idx][:,[1,2]][:,0], mapping[idx][:,[1,2]][:,1]].permute(1,0)
            positive_points = torch.where(features[:,mapping[idx][:,[1,2]][:,0], mapping[idx][:,[1,2]][:,1]][0,:] != 0)[0] # check for active points
            counter[idx[positive_points]] += 1
            
            # Image rotation Rotate back the mask
            for ntime in range(int(args.rotate)):
                for ind in range(boxes_filt.shape[0]):
                    n_t,n_l,n_b,n_r = rotate_coordinates_90_ccw(boxes_filt[ind][0], boxes_filt[ind][1], boxes_filt[ind][2], boxes_filt[ind][3], masks.shape[3])
                    boxes_filt[ind][0] = n_t
                    boxes_filt[ind][1] = n_l
                    boxes_filt[ind][2] = n_b
                    boxes_filt[ind][3] = n_r
                masks = rotate_3db_feature_vector_anticlockwise_90(masks)
            total_mask.append(masks.cpu())
            torch.cuda.empty_cache()
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
        print('Averaging features -- aggregations')
        tmp_counter = counter.clone().cpu()
        tmp_sum_features = sum_features.clone().cpu()
        counter[counter==0] = 1e-5 
        counter_reshaped = counter.unsqueeze(1).expand(-1, sum_features.size(1)).cpu()
        sum_features = sum_features.cpu()
        feat_bank = sum_features/counter_reshaped
        # Devoxelization -- no
        torch.save({"feat": feat_bank, "sum_feat": tmp_sum_features , "count_feat": tmp_counter}, os.path.join(computed_feature, scene_id +'_grounded_ov.pt'))

        #################### 2D mask filtering based on semantic features ####################
        point_features = torch.load(point_feature_path)['feat']
        ### dependent softmax others include
        predicted_class = torch.zeros((point.shape[0], len(class_names)))
        with torch.no_grad(), torch.cuda.amp.autocast():
            adapter.cuda()
            for ind in range(len(class_names)):
                txts = [class_names[ind], 'other']
                text = open_clip.tokenize(txts)
                text_features = adapter.encode_text(text.cuda())
                text_features /= text_features.norm(dim=-1, keepdim=True)
                predicted_class[:, ind]=(100.0 * point_features.half().cuda() @ text_features.T).softmax(dim=-1)[:,0]
        #### Choosing anchors (points) with score > CLIP_THRESH_2
        anchor = torch.where(predicted_class > CLIP_THRESH_2)[0]
        frames = []
        idd = 0
        thresholding_point = 0.001 # 0 means getting all frames
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
            
            masks = total_mask[idd]
            idd += 1
            
            if mapping[:, 3].sum() == 0: # no points corresponds to this image, skip sure
                frames.append({'info': frame, 'masks': None})
                continue
            
            idx = np.where(mapping[:,3] == 1)[0] # indices of projectable pixels
            valid_point = np.where(mapping[anchor.cpu()][:,3]==1)[0] # visible points on 2D image
            # mapping[anchor.cpu()][valid_point] are projected semantic heated pixels from heated points
            
            if valid_point.shape[0] < thresholding_point*anchor.shape[0]:
                frames.append({'info': frame, 'masks': None})
                continue
            
            if masks != None:
                masks = masks.cuda()
            
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
                    sum_mask = mask_pix[0].shape[0]
                    tracer = mapping[anchor.cpu()][valid_point][:,[1,2]] # max: (192,256)
                    num_valid = np.sum(np.all(np.isin(XY.numpy(), tracer), axis=1))
                    num_invalid = XY.shape[0] - num_valid
                    # if num_valid/sum_mask < maskfilter_thresh and num_invalid > maskvalid_thresh:
                        # continue
                    # if num_invalid > maskvalid_thresh: 
                    #     continue
                    if num_valid == 0: 
                        continue
                    result.append(mask_)
                if len(result) == 0: # empty frame
                    result = None
                else:
                    result = torch.stack(result)          
                frames.append({'info': frame, 'masks': result})

        save_path = os.path.join(computed_mask,scene_id+'.pth')
        torch.save(frames, save_path)
        return frames
    
    frames = single_process(point, color) # Grounding DINO + SAM
    frames = torch.load(os.path.join(computed_mask, scene_id+'.pth'))
    save_path = os.path.join(sam3d, str(scene_id)+".pth")
    print('Sam3D')
    seg_pcd(point, frames, args.voxel_size, voxelizer, save_path)
    print('Done')

def instance_merging(args):
    '''
       Merge Instance
    '''
    experiment = args.exp
    datapath = args.datapath
    exp_path = os.path.join(datapath, experiment)
    computed_feature = os.path.join(exp_path, 'computed_feature')
    computed_feature1 = os.path.join(exp_path, 'computed_feature1')
    computed_mask = os.path.join(exp_path, 'computed_mask')
    sam3d = os.path.join(exp_path, 'sam3d')
    mergedsam3d = os.path.join(exp_path, 'mergedsam3d')
    pclpath = os.path.join(datapath, 'pcl')
    
    # Scene
    data_type = "ChallengeTestSet" # or "ChallengeTestSet"
    scene_id = args.scene
    gt_data = os.path.join(pclpath, scene_id+'.pth')
    point_feature_path = os.path.join(computed_feature,scene_id+'_grounded_ov.pt')
    scene_dir = os.path.join(datapath, scene_id)
    instance_path = os.path.join(sam3d,scene_id+'.pth')

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device

    args.captions = args.captions.replace('_',' ')
    args.classname = args.classname.replace('_',' ')
    text_prompt = args.captions
    class_names = args.classname.split('.') # split fullstop    

    ### Set up dataloader
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
    img_dim=(256, 192)
    cut_num_pixel_boundary = 10 # do not use the features on the image boundary
    point2img_mapper = PointCloudToImageMapper(
        image_dim=img_dim, intrinsics=None,
        cut_bound=cut_num_pixel_boundary)
    
    ### Features similarity
    adapter, _, _ = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.clip_checkpoint)
    predicted_class = torch.zeros((point.shape[0], len(class_names)))
    with torch.no_grad(), torch.cuda.amp.autocast():
        adapter.cuda()
        for ind in range(len(class_names)): 
            txts = [class_names[ind], 'other']
            text = open_clip.tokenize(txts)
            text_features = adapter.encode_text(text.cuda()).cuda()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            predicted_class[:, ind] = (100.0 * point_features.half().cuda() @ text_features.T).softmax(dim=-1)[:,0]

    ### Instance merging processing
    instance = torch.load(instance_path) # [-1, n] 
    n_instance = np.unique(instance).shape[0] # -> instance = [0, n_instance - 2]
    color = (color + 1) * 127.5

    semantic = torch.zeros_like(torch.Tensor(instance))
    semantic -= 1
    instances_result = semantic.clone()
    compress = 0 # compress indices
    for ins in range(0, n_instance - 1):
        indices = np.where(instance == ins)[0] # subset of points in the mask
        keep = -1 # -1: No keep, 0, 1, 2...: class_idx
        cmp = torch.zeros(len(class_names))
        for label in range(len(class_names)):
            anchor = torch.where(predicted_class[:, label] > CLIP_THRESH_1)[0]
            num_valid = np.isin(anchor.cpu().numpy(), indices).sum() # number positive point in the mask
            cmp[label] = num_valid
        
        target = torch.argmax(cmp).item()
        keep = target
        if cmp[target] / indices.shape[0] < 0.1:
            keep = -1
            compress += 1
        else:
            # IMPORTANT step to make it clean: we wrap it by ssemantic
            anchor = torch.where(predicted_class[:, keep]>CLIP_THRESH_1)[0]
            filtered = np.where(np.isin(anchor.cpu().numpy(), indices)==True)[0]
            semantic[anchor[filtered]] = keep
            instances_result[anchor[filtered]] = ins - compress
            #semantic[indices] = keep
            #instances_result[indices] = ins
    dic = {'sem': semantic, 'ins': instances_result}
    save_path = os.path.join(mergedsam3d, str(scene_id)+'.pth')
    torch.save(dic, save_path)
    print('Done')



def CascadeAggregator(args):
    '''
       Cascade Aggregator
       SAM mask from cascade features
    '''
    experiment = args.exp
    datapath = args.datapath
    exp_path = os.path.join(datapath, experiment)
    computed_feature = os.path.join(exp_path, 'computed_feature')
    computed_feature1 = os.path.join(exp_path, 'computed_feature1')
    computed_mask = os.path.join(exp_path, 'computed_mask')
    sam3d = os.path.join(exp_path, 'sam3d')
    mergedsam3d = os.path.join(exp_path, 'mergedsam3d')
    pclpath = os.path.join(datapath, 'pcl')
    
    # Scene
    data_type = "ChallengeTestSet" # or "ChallengeTestSet"
    scene_id = args.scene
    gt_data = os.path.join(pclpath, scene_id+'.pth')
    point_feature_path = os.path.join(computed_feature,scene_id+'_grounded_ov.pt')
    scene_dir = os.path.join(datapath, scene_id)
    instance_path = os.path.join(mergedsam3d,scene_id+'.pth')

    args.captions = args.captions.replace('_',' ')
    args.classname = args.classname.replace('_',' ')
    text_prompt = args.captions
    class_names = args.classname.split('.') # split fullstop    

    ### Set up dataloader
    print(os.path.abspath(scene_dir))
    try:
        loader = FrameReaderDataLoaderLowRes(root_path=scene_dir)
    except FileNotFoundError:
        print('>>> Error: Data not found. Did you download it?')
        print('>>> Try: python download_data_opensun3d.py --data_type=challenge_development_set --download_dir=data')
        exit()

    print("Number of frames to load: ", len(loader))
    
    ### Point2Image mapper
    img_dim=(256, 192)
    cut_num_pixel_boundary = 10 # do not use the features on the image boundary
    point2img_mapper = PointCloudToImageMapper(
        image_dim=img_dim, intrinsics=None,
        cut_bound=cut_num_pixel_boundary)
    
    instance_load = torch.load(instance_path)
    instance = instance_load['ins'].clone().detach()
    sem = instance_load['sem'].clone().detach()
    n_instance = torch.unique(instance).shape[0]

    point, color = torch.load(gt_data)
    features_load = torch.load(point_feature_path)
    sum_features = features_load['sum_feat'].cuda()
    counter = features_load['count_feat'].cuda()

    ### Features similarity
    adapter, _, preprocess = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.clip_checkpoint)
    
    cropped_regions = []
    batch_index = []

    interval = args.interval
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
        
        # Get image shape
        image_pil = Image.open(image_path).convert("RGB")
        size = image_pil.size
        H, W = size[1], size[0]
        
        ### Point mapping
        n_points = point.shape[0]
        mapping = torch.ones([n_points, 4], dtype=int)
        mapping[:, 1:4] = torch.tensor(point2img_mapper.compute_mapping(pose, point, depth, intrinsics)).cuda()
        if mapping[:, 3].sum() == 0: # no points corresponds to this image, skip sure
            continue
        # indices of projectable pixels 
        for inst in range(n_instance - 1):
            related_indices = torch.tensor(torch.where(instance==inst)[0])
            idx = torch.where(mapping[:,3] == 1)[0]
            # query_points = mapping[related_indices]
            visible_querypoints = torch.where(mapping[related_indices][:,3]==1)[0]
            # visible points on 2D image
            if idx.shape[0]<100 or visible_querypoints.shape[0]<5:
                continue
            projected_points = torch.tensor(mapping[related_indices][visible_querypoints][:,[1,2]]).cuda()
            # Calculate the bounding rectangle
            mi = torch.min(projected_points, axis=0)
            ma = torch.max(projected_points, axis=0)
            x1, y1 = mi[0][0].item(), mi[0][1].item()
            x2, y2 = ma[0][0].item(), ma[0][1].item()
            cropped_image = image[x1:x2, y1:y2, :]
            # rotate cropped image
            if cropped_image.shape[0] == 0 or cropped_image.shape[1]==0:
                continue
            # Multiscale clip crop
            kexp=0.2
            batch_point = related_indices[visible_querypoints]
            for round in range (5):
                cropped_image = image[x1:x2, y1:y2, :]
                if cropped_image.shape[0] == 0 or cropped_image.shape[1]==0:
                    continue
                for ntime in range(int(args.rotate)):
                    cropped_image = cv2.rotate(cropped_image, cv2.ROTATE_90_CLOCKWISE)
                cropped_regions.append(preprocess(Image.fromarray(cropped_image)))
                batch_index.append(batch_point)
                tmpx1 = int(max(0, x1-(x2-x1)*kexp*round))
                tmpy1 = int(max(0, y1-(y2-y1)*kexp*round))
                tmpx2 = int(min(H-1, x2+(x2-x1)*kexp*round))
                tmpy2 = int(min(W-1, y2+(y2-y1)*kexp*round))
                x1, y1, x2, y2 = tmpx1, tmpy1, tmpx2, tmpy2
    
    if len(cropped_regions)!=0:
        crops = torch.stack(cropped_regions)
        img_batches = torch.split(crops, 32, dim=0)
        image_features = []
        with torch.no_grad(), torch.cuda.amp.autocast():
            adapter.cuda()
            for img_batch in img_batches:
                image_feat = adapter.encode_image(img_batch.cuda())
                image_feat /= image_feat.norm(dim=-1, keepdim=True)
                image_features.append(image_feat.detach())
        image_features = torch.cat(image_features, dim=0)

    print('Cascade-Averaging features')
    for count in trange(len(cropped_regions)):
        sum_features[batch_index[count]]+=image_features[count]
        counter[batch_index[count]] += 1     

    tmp_counter = counter.clone().cpu()
    tmp_sum_features = sum_features.clone().cpu()
    counter[counter==0] = 1e-5
    counter_reshaped = counter.unsqueeze(1).expand(-1, sum_features.size(1)).cuda()
    feat_bank = sum_features/counter_reshaped
    torch.save({"feat": feat_bank.cpu(), "sum_feat": tmp_sum_features , "count_feat": tmp_counter}, os.path.join(computed_feature1, scene_id +'_grounded_ov.pt'))
    print('Done')

def CascadeInstance(args):
    '''
    Generate instance mask using cascaded features
    '''
    experiment = args.exp
    datapath = args.datapath
    exp_path = os.path.join(datapath, experiment)
    computed_feature = os.path.join(exp_path, 'computed_feature')
    computed_feature1 = os.path.join(exp_path, 'computed_feature1')
    computed_mask = os.path.join(exp_path, 'computed_mask')
    sam3d = os.path.join(exp_path, 'sam3d')
    mergedsam3d = os.path.join(exp_path, 'mergedsam3d')
    pclpath = os.path.join(datapath, 'pcl')
    
    # Scene
    data_type = "ChallengeTestSet" # or "ChallengeTestSet"
    scene_id = args.scene
    gt_data = os.path.join(pclpath, scene_id+'.pth')
    point_feature_path = os.path.join(computed_feature1,scene_id+'_grounded_ov.pt')
    scene_dir = os.path.join(datapath, scene_id)
    instance_path = os.path.join(mergedsam3d,scene_id+'.pth')

    args.captions = args.captions.replace('_',' ')
    args.classname = args.classname.replace('_',' ')
    text_prompt = args.captions
    class_names = args.classname.split('.') # split fullstop
    
    ### Set up dataloader
    print(os.path.abspath(scene_dir))
    try:
        loader = FrameReaderDataLoaderLowRes(root_path=scene_dir)
    except FileNotFoundError:
        print('>>> Error: Data not found. Did you download it?')
        print('>>> Try: python download_data_opensun3d.py --data_type=challenge_development_set --download_dir=data')
        exit()

    print("Number of frames to load: ", len(loader))
    
    ### Point2Image mapper
    img_dim=(256, 192)
    cut_num_pixel_boundary = 10 # do not use the features on the image boundary
    point2img_mapper = PointCloudToImageMapper(
        image_dim=img_dim, intrinsics=None,
        cut_bound=cut_num_pixel_boundary)

    point_features = torch.load(point_feature_path)['feat']
    point, color = torch.load(gt_data)
    adapter, _, preprocess = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.clip_checkpoint)
    predicted_class = torch.zeros((point.shape[0], len(class_names)))
    with torch.no_grad(), torch.cuda.amp.autocast():
        adapter.cuda()
        for ind in range(len(class_names)): 
            txts = [class_names[ind], 'other']
            text = open_clip.tokenize(txts)
            text_features = adapter.encode_text(text.cuda()).cuda()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            predicted_class[:, ind] = (100.0 * point_features.half().cuda() @ text_features.T).softmax(dim=-1)[:,0]
    ### Focus Mask Generator
    related_indices = np.array(torch.where(predicted_class[:,0] > CLIP_THRESH_3)[0])
    interval = args.interval
    predictor = SamPredictor(build_sam_hq(checkpoint=args.sam_checkpoint).to('cuda'))
    frames = []
    for i in trange(0, len(loader),interval):
        # adapter.cpu()
        frame = loader[i]
        frame_id = frame['frame_id']  # str
        depth = frame['depth']  # (h, w)
        image = frame['image']  # (h, w, 3), [0-255], uint8
        image_path = frame['image_path']  # str
        intrinsics = frame['intrinsics']  # (3,3) - camera intrinsics
        pose = frame['pose']  # (4,4) - camera pose
        pcd = frame['pcd']  # (n, 3) - backprojected point coordinates in world coordinate system! not camera coordinate system! backprojected and transformed to world coordinate system
        pcd_color = frame['color']  # (n,3) - color values for each point
        
        # Get image shape
        image_pil = Image.open(image_path).convert("RGB")
        size = image_pil.size
        H, W = size[1], size[0]
        
        ### Point mapping
        n_points = point.shape[0]
        mapping = torch.ones([n_points, 4], dtype=int)
        mapping[:, 1:4] = torch.tensor(point2img_mapper.compute_mapping(pose, point, depth, intrinsics)).cuda()
        if mapping[:, 3].sum() == 0: # no points corresponds to this image, skip sure
            frames.append({'info': frame, 'masks': None})
            continue
        # indices of projectable pixels 
        
        idx = torch.where(mapping[:,3] == 1)[0]
        query_points = mapping[related_indices]
        visible_querypoints = torch.where(mapping[related_indices][:,3]==1)[0]
        # visible points on 2D image
        if idx.shape[0]<100 or visible_querypoints.shape[0]<20:
            frames.append({'info': frame, 'masks': None})
            continue
        projected_points = np.array(mapping[related_indices][visible_querypoints][:,[1,2]])

        # Calculate the bounding rectangle
        left = min(point[0] for point in projected_points)
        right = max(point[0] for point in projected_points)
        top = min(point[1] for point in projected_points)
        bottom = max(point[1] for point in projected_points)
        boxes_filt = torch.tensor([top, left, bottom, right]).cuda().unsqueeze(0)
        #### Segment Anything ####
        image_sam = cv2.imread(image_path)
        image_sam = cv2.cvtColor(image_sam, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_sam)
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_sam.shape[:2]).to('cuda')

        masks, _, _ = predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to('cuda'),
            multimask_output = False)
        if masks == None:
            frames.append({'info': frame, 'masks': None})
            continue
        else:
            frames.append({'info': frame, 'masks': masks})
            if False:
                # draw output image
                plt.figure(figsize=(10, 10))
                plt.imshow(image)
                for mask in masks:
                    show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
                for box in boxes_filt:
                    show_box(box.cpu().numpy(), plt.gca(), 'ins')
                plt.axis('off')
                plt.savefig(os.path.join("testfig/grounded_sam_output"+str(i)+'.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)
     # Segment Anything 3D
    voxelizer = Voxelize(voxel_size=args.voxel_size, mode="train", keys=("coord", "color", "group"))
    save_path = 'tmp.pth'
    seg_pcd(point, frames, args.voxel_size, voxelizer, save_path)
            

def Final_Result(args):
    '''
       Final Result
    '''
    experiment = args.exp
    datapath = args.datapath
    exp_path = os.path.join(datapath, experiment)
    computed_feature = os.path.join(exp_path, 'computed_feature')
    computed_feature1 = os.path.join(exp_path, 'computed_feature1')
    computed_mask = os.path.join(exp_path, 'computed_mask')
    sam3d = os.path.join(exp_path, 'sam3d')
    mergedsam3d = os.path.join(exp_path, 'mergedsam3d')
    pclpath = os.path.join(datapath, 'pcl')
    final_path = os.path.join(exp_path, 'final_result')
    
    # Scene
    data_type = "ChallengeTestSet" # or "ChallengeTestSet"
    scene_id = args.scene
    gt_data = os.path.join(pclpath, scene_id+'.pth')
        # Cascaded features stage 1
    point_feature_path = os.path.join(computed_feature1,scene_id+'_grounded_ov.pt')
    scene_dir = os.path.join(datapath, scene_id)
    instance_path = os.path.join(mergedsam3d,scene_id+'.pth')

    args.captions = args.captions.replace('_',' ')
    args.classname = args.classname.replace('_',' ')
    text_prompt = args.captions
    class_names = args.classname.split('.') # split fullstop    

    ## CLIP adapter
    adapter, _, preprocess = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.clip_checkpoint)

    data = torch.load(instance_path)
    instance = torch.tensor(data['ins'].clone().detach())
    sem = torch.tensor(data['sem'].clone().detach())
    n_instance = torch.unique(instance).shape[0]

    point, color = torch.load(gt_data)
    features_load = torch.load(point_feature_path)
    point_features = features_load['feat'].cuda()

    predicted_class = torch.zeros((point.shape[0], len(class_names)))
    with torch.no_grad(), torch.cuda.amp.autocast():
        adapter.cuda()
        for ind in range(len(class_names)): 
            txts = [class_names[ind], 'other']
            text = open_clip.tokenize(txts)
            text_features = adapter.encode_text(text.cuda()).cuda()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            predicted_class[:, ind] = (100.0 * point_features.half().cuda() @ text_features.T).softmax(dim=-1)[:,0]
    
    # Averaging scores for each instance
    scores = []
    for i in range(n_instance - 1):
        instance_ind = torch.where(instance==i)[0]
        score = torch.mean(predicted_class[instance_ind, 0]).item()
        scores.append(score)
    scores = torch.tensor(scores)
    dic = {'ins': instance, 'sem': sem, 'conf': scores}
    torch.save(dic, os.path.join(final_path, scene_id + '.pth'))
    print('Done')


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for open vocabulary segmentation")
    # Scene preprocessing
    # CLIP features
    parser.add_argument("--scene",type=str,required = True,help="scene id in challenge")
    parser.add_argument("--classname",type=str,required = True,help="classname")
    parser.add_argument("--captions",type=str,required = True,help="text caption")
    parser.add_argument("--rotate",type=str,required = True,help="scene rotation")
    parser.add_argument("--distractor",type=str,required = True,help="distractor")
    parser.add_argument("--clip_model", type=str, required=True, help="Open CLIP model")
    parser.add_argument("--clip_checkpoint", type=str, required=True, help="path to CLIP checkpoint file")
    parser.add_argument("--feat_dim", type=int, required=True, help="CLIP feature dimension")
    
    # Grounding DINO + SAM
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument("--grounded_checkpoint", type=str, required=True, help="path to checkpoint file")
    parser.add_argument("--sam_checkpoint", type=str, required=True, help="path to sam checkpoint file")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")

    # SegmentAnything3D
    parser.add_argument("--voxel_size", type=float, default=0.05, help="voxel size for SAM3D")
    parser.add_argument("--interval", type=int, required=True, help="interval of 2D views")
    
    # Workspace
    parser.add_argument("--datapath", type=str, required=True, help="dataset path")
    parser.add_argument("--exp", type=str, required=True, help="experiments name")
    return parser

def create_workspace(args):
    '''
    Create workspace
    '''
    experiment = args.exp
    datapath = args.datapath
    exp_path = os.path.join(datapath, experiment)
    computed_feature = os.path.join(exp_path, 'computed_feature')
    computed_feature1 = os.path.join(exp_path, 'computed_feature1')
    computed_mask = os.path.join(exp_path, 'computed_mask')
    # all_mask = os.path.join(exp_path, 'all_mask')
    sam3d = os.path.join(exp_path, 'sam3d')
    mergedsam3d = os.path.join(exp_path, 'mergedsam3d')
    final_result = os.path.join(exp_path, 'final_result')
    try:
        os.makedirs(exp_path)
        os.makedirs(computed_feature)
        os.makedirs(computed_feature1)
        os.makedirs(computed_mask)
        os.makedirs(mergedsam3d)
        # os.makedirs(all_mask)
        os.makedirs(sam3d)
        os.makedirs(final_result)
        print(f"Workspace '{experiment}' created successfully.")
    except FileExistsError:
        print(f"Workspace '{experiment}' already exists.")
    except OSError as e:
        print(f"Error creating {experiment}: {e}")

if __name__=='__main__':
    
    args = get_parser().parse_args()
    
    # ### Create workspace
    print('---------- Create workspace ----------')
    create_workspace(args)

    # ### CLIP features
    print('---------- Grouned OV Features ----------')
    grounded_features(args)

    # ### Merging instance mask 
    print('----------Instance mask merge----------')
    instance_merging(args)
    
    # ### Cascade
    print('----------Cascade features----------')
    CascadeAggregator(args)

    # Instance Gen
    # print('----------Instance Mask----------')
    # CascadeInstance(args)

    # Final Result
    print('---------Final Result----------')
    Final_Result(args)