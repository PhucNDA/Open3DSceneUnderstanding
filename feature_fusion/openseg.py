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
import tensorflow as tf2
import tensorflow.compat.v1 as tf
# Loader Point Cloud mapper, Dataloader, Nouns processing
from benchmark_scripts.opensun3d_challenge_utils import FrameReaderDataLoaderLowRes
from fusion_util import PointCloudToImageMapper, save_fused_feature, get_nouns, rotate_3db_feature_vector_anticlockwise_90, rotate_3d_feature_vector_anticlockwise_90
from fusion_util import PointCloudToImageMapper, save_fused_feature, NMS, mask_nms, extract_openseg_img_feature
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
def openseg_feat(args):
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
        ### Feature bank
        interval = args.interval # interval consecutive frames default 1: correct lifting
        feat_dim = args.feat_dim #bigG = 1280
        sum_features = torch.zeros((point.shape[0], feat_dim)).cuda()
        counter = torch.zeros((point.shape[0], )).cpu()
        openseg_model = tf2.saved_model.load('../../openseg_exported_clip',tags=[tf.saved_model.tag_constants.SERVING],)
        text_emb = tf.zeros([1, 1, 768])
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
                continue

            feat_2d = extract_openseg_img_feature(image_path, openseg_model, text_emb).to(device)
            feat_2d_3d = feat_2d[:, mapping[:, 1], mapping[:, 2]].permute(1, 0)
            counter[idx]+= 1
            sum_features[idx] += feat_2d_3d[idx]
        counter[counter==0] = 1e-5
        counter_reshaped = counter.unsqueeze(1).expand(-1, sum_features.size(1)).cuda()
        counter_reshaped = counter_reshaped.cuda()
        breakpoint()
        feat_bank = sum_features/counter_reshaped
        torch.save(feat_bank,  point_feature_path)
    
    single_process(point, color)
    # adapter, _, preprocess = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.clip_checkpoint)
    # point_features = torch.load(computed_feature)
    # class_preds = torch.zeros((point.shape[0], 1))
    # with torch.no_grad(), torch.cuda.amp.autocast():
    #     for ind in range(len(class_names)):
    #         txts = [class_names[ind], 'others']
    #         text = open_clip.tokenize(txts)
    #         text_features = adapter.encode_text(text.cuda())
    #         text_features /= text_features.norm(dim=-1, keepdim=True)
    #         class_preds[:, ind]=(100.0 * point_features.half() @ text_features.T).softmax(dim=-1)[:,0]
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
    print('---------- openseg_feat Features ----------')
    openseg_feat(args)
