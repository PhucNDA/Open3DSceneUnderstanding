# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.nn import functional as F

from detectron2.config import get_cfg

from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from open_vocab_seg import add_ovseg_config

from open_vocab_seg.utils import VisualizationDemo

# constants
WINDOW_NAME = "Open vocabulary segmentation"


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
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        help="A list of user-defined class_names"
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
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

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    class_names = args.class_names
    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            ### Specific for the dataset Flipped
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

            start_time = time.time()
            predictions, adapter, visualized_output = demo.run_on_image(img, class_names, adapter_return=True)
            
            # Original image
            
            coords = rotate_3d_feature_vector_anticlockwise_90(predictions[0]['sem_seg'].permute(1, 2, 0)).permute(2, 0, 1)
            features = rotate_3d_feature_vector_anticlockwise_90(predictions[0]['feat_seg'].permute(1, 2, 0)).permute(2, 0, 1).cpu()
            ### Text Features
            text_features = adapter.get_text_features(class_names).cpu()
            tmp = torch.where(coords!=0)
            predicted_class = torch.argmax(F.softmax(adapter.get_sim_logits(text_features, features[:,tmp[1],tmp[2]].permute(1, 0)),
                                    dim=-1)[...,:-1], dim=-1)
            ### Visualize image - for debugging only 
            if False:
                img = read_image(path, format="BGR")
                fig, ax = plt.subplots()
                ax.imshow(img)
                num_classes = len(torch.unique(predicted_class))
                colors = plt.cm.rainbow(torch.linspace(0, 1, num_classes + 1))
                for i in range(num_classes):
                    mask = (predicted_class == i)
                    plt.scatter(tmp[2][mask].cpu(), tmp[1][mask].cpu(), color=colors[i + 1], label=f'Class {i}', alpha=0.7)
                plt.legend()
                plt.savefig('scatter_plot.png')

            


            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    else:
        raise NotImplementedError