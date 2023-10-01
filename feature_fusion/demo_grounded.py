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


############################################

def grounded_features(args):
    outputdir = '../../Dataset/iccvw/ChallengeTestSet/computed_feature'
    # Scene
    img_folder = 'lowres_wide'
    inputdir = '../../Dataset/iccvw/ChallengeTestSet/computed_feature'
    arkitscenes_root_dir = "../../Dataset/iccvw" #"PATH/TO/ARKITSCENES/DOWNLOAD/DIR"
    data_type = "ChallengeTestSet" # or "ChallengeTestSet"
    # scene_ids = ['42445173', '42445677', '42445935', '42446478', '42446588'] # Dev Scenes
    scene_id = args.scene
    gt_data = '../../Dataset/iccvw/ChallengeTestSet/pcl/'+scene_id+'.pth'

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    captions = args.captions
    device = args.device
    rotate = args.rotate

    args.captions = args.captions.replace('_',' ')
    text_prompt = args.captions
    class_names = args.captions.split('.')
    # class_names = get_nouns(args.captions)
    # text_prompt = ''
    # for name in class_names: 
    #     text_prompt += name + '. ' # Format of Grounding DINO
    # text_prompt = text_prompt[:-1]

    ### Set up ovseg
    adapter, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained=args.clip_checkpoint)

    # Grouding DINO and SAM
    model = load_model(config_file, grounded_checkpoint, device=device)
    predictor = SamPredictor(build_sam_hq(checkpoint=sam_checkpoint).to(device))
    
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
    
    thresholding_point = 100 # 0 means getting all frames
    granularity = 0.8
    batch_processing = True
    def single_process():
        ### Feature bank
        feat_dim = 768
        interval = 1 # interval consecutive frames default 1: correct lifting
        feat_dim = 768
        sum_features = torch.zeros((point.shape[0], feat_dim)).cuda()
        counter = torch.zeros((point.shape[0], )).cuda()
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
            
            ### Point mapping
            n_points = point.shape[0]
            mapping = torch.ones([n_points, 4], dtype=int).cuda()
            mapping[:, 1:4] = torch.tensor(point2img_mapper.compute_mapping(pose, point, depth, intrinsics)).cuda()
            if mapping[:, 3].sum() == 0: # no points corresponds to this image, skip sure
                continue
            # indices of projectable pixels
            idx = torch.where(mapping[:,3] == 1)[0]
            
            # visible points on 2D image
            if idx.shape[0]<100:
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
            for j in range(boxes_filt.size(0)):
                boxes_filt[j] = boxes_filt[j] * torch.Tensor([W, H, W, H]).cuda()
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

            for ntime in range(int(args.rotate)):
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
                for ntime in range(int(args.rotate)):
                    masks = rotate_3db_feature_vector_anticlockwise_90(masks)
            
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
            if True:
                # draw output image
                plt.figure(figsize=(10, 10))
                plt.imshow(image)
                for mask in masks:
                    show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
                for box, label in zip(boxes_filt, pred_phrases):
                    show_box(box.numpy(), plt.gca(), label)
                plt.axis('off')
                plt.savefig(os.path.join("testfig/grounded_sam_output"+str(i)+'.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)
            
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
            # max_scores, select_mask = torch.max(class_preds, dim=0)

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
            final_feat = torch.einsum("qc,qhw->chw", select_features, pred_masks.tensor.float().cuda())
            if False:
                semseg = torch.einsum("qc,qhw->chw", select_cls.float().cuda(), pred_masks.tensor.float().cuda())
                r = semseg
                blank_area = (r[0] == 0)
                pred_mask = r.argmax(dim=0).to('cpu')
                pred_mask[blank_area] = 255
                pred_mask = np.array(pred_mask, dtype=np.int)
                visualizer = OVSegVisualizer(image, None, class_names=class_names)
                vis_output = visualizer.draw_sem_seg(pred_mask)
                vis_output.save('testfig/'+str(i)+'.png')
            ### Summing features
            features = final_feat
            sum_features[idx] += features[:,mapping[idx][:,[1,2]][:,0], mapping[idx][:,[1,2]][:,1]].permute(1,0)
            positive_points = torch.where(features[:,mapping[idx][:,[1,2]][:,0], mapping[idx][:,[1,2]][:,1]][0,:] != 0)[0] # check for active points
            counter[idx[positive_points]] += 1

            torch.cuda.empty_cache()

        print('Averaging features')
        counter[counter==0] = 1e-5 
        counter_reshaped = counter.unsqueeze(1).expand(-1, sum_features.size(1))
        feat_bank = sum_features/counter_reshaped
        torch.save({"feat": feat_bank.cpu()}, os.path.join(outputdir, scene_id +'_grounded_ov.pt'))
        return os.path.join(outputdir, scene_id +'_grounded_ov.pt')
    
    return single_process()

def groundedsam(args):
    # Scene
    inputdir = '../../Dataset/iccvw/ChallengeTestSet/computed_feature'
    arkitscenes_root_dir = "../../Dataset/iccvw" #"PATH/TO/ARKITSCENES/DOWNLOAD/DIR"
    data_type = "ChallengeTestSet" # or "ChallengeTestSet"
    # scene_ids = ['42445173', '42445677', '42445935', '42446478', '42446588'] # Dev Scenes
    scene_id = args.scene
    gt_data = '../../Dataset/iccvw/ChallengeTestSet/pcl/'+scene_id+'.pth'
    point_feature_path = inputdir+'/'+scene_id+'_grounded_ov.pt'

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device
    rotate = args.rotate

    args.captions = args.captions.replace('_',' ')
    text_prompt = args.captions
    class_names = args.captions.split('.')
    # class_names = get_nouns(args.captions)
    # text_prompt = ''
    # for name in class_names: 
    #     text_prompt += name + '. ' # Format of Grounding DINO
    # text_prompt = text_prompt[:-1]

    ## CLIP adapter
    adapter, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained=args.clip_checkpoint)

    # Grouding DINO and SAM
    model = load_model(config_file, grounded_checkpoint, device=device)
    predictor = SamPredictor(build_sam_hq(checkpoint=sam_checkpoint).to(device))
    
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
    
    ### dependent softmax others include
    predicted_class = torch.zeros((point.shape[0], len(class_names)))
    with torch.no_grad(), torch.cuda.amp.autocast():
        adapter.cuda()
        for ind in range(len(class_names)):
            txts = [class_names[ind], 'others']
            text = open_clip.tokenize(txts)
            text_features = adapter.encode_text(text.cuda())
            text_features /= text_features.norm(dim=-1, keepdim=True)
            predicted_class[:, ind]=(100.0 * point_features.half().cuda() @ text_features.T).softmax(dim=-1)[:,0]

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
    
    #### Choosing anchors (points) with score > 0.7
    anchor = torch.where(predicted_class>0.7)[0]
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
            
            rotated_pil = image_pil.copy()
            for n_times in range(int(args.rotate)):
                rotated_pil = Image.fromarray(np.rot90(np.array(rotated_pil),k=-1))

            image_pil, image_infer = load_image(rotated_pil)
            boxes_filt, pred_phrases = get_grounding_output(model, image_infer, text_prompt, box_threshold, text_threshold, device=device)
            
            #### Segment Anything ####
            image_sam = cv2.imread(image_path)
            image_sam = cv2.cvtColor(image_sam, cv2.COLOR_BGR2RGB)
            
            for n_times in range(int(args.rotate)):
                image_sam = cv2.rotate(image_sam, cv2.ROTATE_90_CLOCKWISE)

            predictor.set_image(image_sam)
            size = image_pil.size
            H, W = size[1], size[0]
            for j in range(boxes_filt.size(0)):
                boxes_filt[j] = boxes_filt[j] * torch.Tensor([W, H, W, H]).cuda()
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
                for n_times in range(int(args.rotate)):
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

        save_path = '../../Dataset/iccvw/ChallengeTestSet/computed_mask/'+scene_id+'.pth'
        torch.save(frames, save_path)
        return frames
    
    frames = single_process() # Grounding DINO + SAM
    # frames = torch.load('../../Dataset/iccvw/ChallengeTestSet/computed_mask/'+scene_id+'.pth')
    save_path = "../../Dataset/iccvw/ChallengeDevelopmentSet/sam3d/"+str(scene_id)+".pth"
    seg_pcd(point, frames, args.voxel_size, voxelizer, save_path)

def instance_merging(args):
    '''
       Merge Instance
    '''
    # Scene
    inputdir = '../../Dataset/iccvw/ChallengeTestSet/computed_feature'
    arkitscenes_root_dir = "../../Dataset/iccvw" #"PATH/TO/ARKITSCENES/DOWNLOAD/DIR"
    data_type = "ChallengeTestSet" # or "ChallengeTestSet"
    # scene_ids = ['42445173', '42445677', '42445935', '42446478', '42446588'] # Dev Scenes
    scene_id = args.scene
    gt_data = '../../Dataset/iccvw/ChallengeTestSet/pcl/'+scene_id+'.pth'
    point_feature_path = inputdir+'/'+scene_id+'_grounded_ov.pt'
    instance_path = '../../Dataset/iccvw/ChallengeTestSet/sam3d/'+scene_id+'.pth'

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device

    args.captions = args.captions.replace('_',' ')
    text_prompt = args.captions
    class_names = args.captions.split('.')
    # class_names = get_nouns(args.captions)
    # text_prompt = ''
    # for name in class_names: 
    #     text_prompt += name + '. ' # Format of Grounding DINO
    # text_prompt = text_prompt[:-1]

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
    adapter, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained=args.clip_checkpoint)
    predicted_class = torch.zeros((point.shape[0], len(class_names)))
    with torch.no_grad(), torch.cuda.amp.autocast():
        adapter.cuda()
        for ind in range(len(class_names)): 
            txts = [class_names[ind], 'others']
            text = open_clip.tokenize(txts)
            text_features = adapter.encode_text(text.cuda()).cuda()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            predicted_class[:, ind] = (100.0 * point_features.half().cuda() @ text_features.T).softmax(dim=-1)[:,0]
    
    if False: # Visualization
        v = viz.Visualizer()
        color = (color + 1) * 127.5
        # There are 2 query class
        v.add_points(f'pcl color', point, color, point_size=20, visible=True)
        color[(predicted_class[:,0]>0.8).cpu()]=np.array((255,0,0))
        v.add_points(f'pcl gt_label', point, color, point_size=20, visible=True)
        color[(predicted_class[:,1]>0.8).cpu()]=np.array((0,255,0))
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
    compress = 0 # compress indices
    for ins in range(0, n_instance - 1):
        indices = np.where(instance == ins)[0] # subset of points in the mask
        keep = -1 # -1: No keep, 0, 1, 2...: class_idx
        cmp = torch.zeros(len(class_names))
        for label in range(len(class_names)):
            anchor = torch.where(predicted_class[:, label]>0.7)[0]
            num_valid = np.isin(anchor.cpu().numpy(), indices).sum() # number positive point in the mask
            cmp[label] = num_valid
        
        target = torch.argmax(cmp).item()
        keep = target
        if cmp[target] / indices.shape[0] < 0.1:
            keep = -1
            compress += 1
        else:
            # IMPORTANT step to make it clean: we wrap it by semantic
            anchor = torch.where(predicted_class[:, keep]>0.7)[0]
            filtered = np.where(np.isin(anchor.cpu().numpy(), indices)==True)[0]
            semantic[anchor[filtered]] = keep
            instances_result[anchor[filtered]] = ins - compress
            #semantic[indices] = keep
            #instances_result[indices] = ins
    dic = {'sem': semantic, 'ins': instances_result}
    torch.save(dic, '../../Dataset/iccvw/ChallengeTestSet/mergedsam3d/'+str(scene_id)+'.pth')
    return None

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for open vocabulary segmentation")
    # Scene preprocessing
    # OV-Seg features
    parser.add_argument("--scene",type=str,required = True,help="scene id in challenge")
    parser.add_argument("--captions",type=str,required = True,help="text caption")
    parser.add_argument("--rotate",type=str,required = True,help="scene rotation")
    parser.add_argument("--clip_checkpoint", type=str, required=False, help="path to CLIP checkpoint file")
    # Grounding DINO + SAM
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument("--grounded_checkpoint", type=str, required=True, help="path to checkpoint file")
    parser.add_argument("--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")

    # SegmentAnything3D
    parser.add_argument("--voxel_size", type=float, default=0.05, help="voxel size for SAM3D")
    return parser

if __name__=='__main__':
    
    ### OV-Seg
    args = get_parser().parse_args()
    
    print('----------Grouned OV Features----------')
    grounded_features(args)

    ### Grounding DINO + SAM
    print('----------Grounding DINO and SAM----------')
    groundedsam(args)

    ### Merging instance mask 
    print('----------Instance mask merge----------')
    instance_merging(args)
    