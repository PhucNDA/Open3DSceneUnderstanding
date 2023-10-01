
import torch
import os
import numpy as np
import pyviz3d.visualizer as viz
import open3d as o3d
from benchmark_scripts.opensun3d_challenge_utils import FrameReaderDataLoaderLowRes
from fusion_util import PointCloudToImageMapper, save_fused_feature
from PIL import Image, ImageDraw

import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import matplotlib.pyplot as plt
from torch.nn import functional as F
from tqdm import tqdm, trange

from detectron2.config import get_cfg

from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from ovseg.open_vocab_seg import add_ovseg_config
from ovseg.open_vocab_seg.utils import VisualizationDemo

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


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for open vocabulary segmentation")
    parser.add_argument(
        "--config-file",
        default="configs/ovseg_swinB_vitL_demo.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        help="A list of user-defined class_names"
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def rotate_3d_feature_vector_anticlockwise_90(feature_vector):
    rotated_vector = feature_vector.permute(1, 0, 2)
    rotated_vector = torch.flip(rotated_vector, dims=(0,))

    return rotated_vector

def rotate_3d_feature_vector_anticlockwise_90_batch(feature_vector):
    rotated_vector = feature_vector.permute(0, 2, 1, 3)
    rotated_vector = torch.flip(rotated_vector, dims=(1,))

    return rotated_vector

if __name__=='__main__':

    outputdir = '../../Dataset/iccvw/ChallengeDevelopmentSet/computed_feature'

    arkitscenes_root_dir = "../../Dataset/iccvw" #"PATH/TO/ARKITSCENES/DOWNLOAD/DIR"
    data_type = "ChallengeDevelopmentSet" # or "ChallengeTestSet"
    scene_ids = ['42445173', '42445677', '42445935', '42446478', '42446588'] # Dev Scenes
    scene_id = scene_ids[3]
    img_folder = 'lowres_wide'
    gt_data = '../../Dataset/iccvw/ChallengeDevelopmentSet/pcl/'+scene_id+'.pth'
    
    ### Set up model
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)
    class_names = args.class_names

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

    ### Single Processing Test: passed
    def single_process():
        ### Feature bank
        feat_dim = 768
        sum_features = torch.zeros((point.shape[0], feat_dim))
        counter = torch.zeros((point.shape[0], ))
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
            # indices of projectable pixels
            idx = np.where(mapping[:,3] == 1)[0]
            # visualization of projectable pixels and points - Test: passed
            if False:
                img = Image.fromarray(image)
                for iter in mapping[idx]:
                    draw = ImageDraw.Draw(img)
                    draw.rectangle([iter[2]-2,iter[1]-2,iter[2]+2,iter[1]+2], outline='red', width=2)
                    img.save("test.jpg")
                
                v = viz.Visualizer()
                v.add_points(f'pcl color', point, (color+1)*127.5, point_size=20, visible=True)
                color[idx]=np.array((255,0,0))
                v.add_points(f'pcl gt_label', point, color, point_size=20, visible=True)
                v.save('viz')

            ### OVSeg image forwarding
            img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            predictions, adapter, visualized_output = demo.run_on_image(img, class_names, adapter_return=True)
                # Original image  
            coords = rotate_3d_feature_vector_anticlockwise_90(predictions['sem_seg'].permute(1, 2, 0)).permute(2, 0, 1)
            features = rotate_3d_feature_vector_anticlockwise_90(predictions['feat_seg'].permute(1, 2, 0)).permute(2, 0, 1)
            
            
            # visualization of related points - Test: passed
            if False:
                text_features = adapter.get_text_features(class_names)
                Image.fromarray(image).save("test.jpg")
                features = features.cpu()
                sum_features[idx] += features[:,mapping[idx][:,[1,2]][:,0], mapping[idx][:,[1,2]][:,1]].permute(1,0)
                predicted_class = F.softmax(adapter.get_sim_logits(text_features, sum_features.cuda())[...,:-1],dim=-1)
                v = viz.Visualizer()
                # There are 2 query class
                v.add_points(f'pcl color', point, (color+1)*127.5, point_size=20, visible=True)
                color[(predicted_class[:,0]>0.6).cpu()]=np.array((255,0,0))
                v.add_points(f'pcl gt_label', point, color, point_size=20, visible=True)
                color[(predicted_class[:,1]>0.6).cpu()]=np.array((0,255,0))
                v.add_points(f'pcl gt_label1', point, color, point_size=20, visible=True)
                breakpoint()
                visualized_output.save("test_predict.jpg")
                v.save('viz')
            else:
                features = features.cpu()
                sum_features[idx] += features[:,mapping[idx][:,[1,2]][:,0], mapping[idx][:,[1,2]][:,1]].permute(1,0)
                counter[idx] += 1

        print('Averaging features')
        counter[counter==0] = 1e-5 
        counter_reshaped = counter.unsqueeze(1).expand(-1, sum_features.size(1))
        feat_bank = sum_features/counter_reshaped
        breakpoint()
        text_features = adapter.get_text_features(class_names)
        predicted_class = F.softmax(adapter.get_sim_logits(text_features, feat_bank.cuda())[...,:-1],dim=-1)
        # final visualization of avegraged features of all views - Test: passed
        if False:
            v = viz.Visualizer()
            # There are 2 query class
            v.add_points(f'pcl color', point, (color+1)*127.5, point_size=20, visible=True)
            color[(predicted_class[:,0]>0.6).cpu()]=np.array((255,0,0))
            v.add_points(f'pcl gt_label', point, color, point_size=20, visible=True)
            color[(predicted_class[:,1]>0.8).cpu()]=np.array((0,255,0))
            v.add_points(f'pcl gt_label1', point, color, point_size=20, visible=True)
            breakpoint()
            visualized_output.save("test_predict.jpg")
            v.save('viz')
    ### Batch Processing Test: passed
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
            # indices of projectable pixels
            # idx = np.where(mapping[:,3] == 1)[0]
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
            #prediction, visualized_output = demo.run_on_image(images[start_idx:end_idx], class_names, adapter_return=False)
            # predictions[B][{semseg, feat_seg}]
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
        text_features = adapter.get_text_features(class_names)
        predicted_class = F.softmax(adapter.get_sim_logits(text_features, feat_bank.cuda())[...,:-1],dim=-1)
        # final visualization of avegraged features of all views - Test: passed
        if False:
            v = viz.Visualizer()
            color = (color + 1) * 127.5
            # There are 2 query class
            v.add_points(f'pcl color', point, color, point_size=20, visible=True)
            color[(predicted_class[:,0]>0.7).cpu()]=np.array((255,0,0))
            v.add_points(f'pcl gt_label', point, color, point_size=20, visible=True)
            color[(predicted_class[:,1]>0.7).cpu()]=np.array((0,255,0))
            v.add_points(f'pcl gt_label1', point, color, point_size=20, visible=True)
            # color[(predicted_class[:,2]>0.7).cpu()]=np.array((0,0,255))
            # v.add_points(f'pcl gt_label2', point, color, point_size=20, visible=True)
            # color[(predicted_class[:,3]>0.7).cpu()]=np.array((65,150,127))
            # v.add_points(f'pcl gt_label3', point, color, point_size=20, visible=True)
            breakpoint()
            v.save('viz')
        ### Save Fused Feature
        torch.save({"feat": feat_bank.half().cpu()}, os.path.join(outputdir, scene_id +'_ovseg.pt'))


    if batch_processing == False: 
        single_process()
    else:
        batch_process()