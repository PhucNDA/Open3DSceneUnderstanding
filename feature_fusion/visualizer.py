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
import csv
import open_clip

# Point mapper
import pyviz3d.visualizer as viz
import open3d as o3d
from benchmark_scripts.opensun3d_challenge_utils import FrameReaderDataLoaderLowRes
from fusion_util import PointCloudToImageMapper, save_fused_feature, NMS, mask_nms
from PIL import Image, ImageDraw

from ovseg.open_vocab_seg import add_ovseg_config
from ovseg.open_vocab_seg.utils import VisualizationDemo

##############################################

def get_parser():
    parser = argparse.ArgumentParser("Visualization Demo", add_help=True)
    ### Features
    
    parser.add_argument("--type",type=str,required = True,help="development set or test set")
    parser.add_argument("--scene",type=str,required = True,help="scene id in challenge")
    parser.add_argument("--clip_checkpoint", type=str, required=True, help="path to CLIP checkpoint file")
    parser.add_argument("--exp", type=str, required=True, help="experiment workspace")
    parser.add_argument("--feat", type=str, required=True, help="feature type")
    return parser

if __name__ == "__main__":
    
    args = get_parser().parse_args()
    if args.type == 'test':
        csvf = 'queries_test_scenes.csv'
        data_type = 'ChallengeTestSet'
    else:
        csvf = 'queries_challenge_scenes.csv'
        data_type = 'ChallengeDevelopmentSet'
        
    scenes = []
    with open(csvf, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            scenes.append(row)
    scenes.pop(0)

    # Scene
    exp = args.exp
    inputdir = '../../Dataset/iccvw/' + data_type +'/' + exp +  '/' + args.feat
    arkitscenes_root_dir = "../../Dataset/iccvw" #"PATH/TO/ARKITSCENES/DOWNLOAD/DIR"
    scene_id = args.scene
    gt_data = '../../Dataset/iccvw/' + data_type + '/pcl/'+scene_id+'.pth'
    point_feature_path = inputdir+'/'+scene_id+'_grounded_ov.pt'

    target_index = next((index for index, row in enumerate(scenes) if row[0] == scene_id), None)
    text_prompt = scenes[target_index][5]
    class_names = text_prompt.split('.')
    class_names = [name.strip() for name in class_names]

    ### Set up dataloader
    data_root_dir = os.path.join(arkitscenes_root_dir, data_type)
    scene_dir = os.path.join(data_root_dir, scene_id)
    print(os.path.abspath(scene_dir))

    point, color = torch.load(gt_data)
    point_features = torch.load(point_feature_path)['feat']

    ### Point2Image mapper
    visibility_threshold = 0.03 # threshold for the visibility check
    img_dim=(256, 192)
    cut_num_pixel_boundary = 10 # do not use the features on the image boundary
    
    ### Features similarity
    # adapter, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained=args.clip_checkpoint)
    adapter, _, _ = open_clip.create_model_and_transforms('ViT-L-14-336', pretrained = 'openai')
    # adapter, _, preprocess = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k')
    predicted_class = torch.zeros((point.shape[0], len(class_names)))
    with torch.no_grad(), torch.cuda.amp.autocast():
        adapter.cuda()
        for ind in range(len(class_names)): 
            txts = [class_names[ind], 'other']
            text = open_clip.tokenize(txts)
            text_features = adapter.encode_text(text.cuda()).cuda()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            predicted_class[:, ind] = (100.0 * point_features.half().cuda() @ text_features.T).softmax(dim=-1)[:,0]
    if True: # Semantic Visualization
        v = viz.Visualizer()
        color = (color + 1) * 127.5
        # There are 2 query class
        v.add_points(f'pcl color', point, color, point_size=20, visible=True)
        color[(predicted_class[:,0]>0.7).cpu()]=np.array((255,0,0))
        v.add_points(f'pcl gt_label', point, color, point_size=20, visible=True)
        
        color[(predicted_class[:,0]>0.8).cpu()]=np.array((0,0,255))
        v.add_points(f'pcl gt_label1', point, color, point_size=20, visible=True)

        color[(predicted_class[:,0]>0.9).cpu()]=np.array((0,255,0))
        v.add_points(f'pcl gt_label2', point, color, point_size=20, visible=True)
        

        color[(predicted_class[:,0]>0.95).cpu()]=np.array((128,0,128))
        v.add_points(f'pcl gt_label3', point, color, point_size=20, visible=True)

        # color[(predicted_class[:,1]>0.8).cpu()]=np.array((0,255,0))
        # v.add_points(f'pcl gt_label1', point, color, point_size=20, visible=True)
        v.save('viz')
    


    

