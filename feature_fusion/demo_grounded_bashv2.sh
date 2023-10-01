#!/bin/bash

arg1=$1
arg2=$2
arg3=$3
arg4=$4
arg5=$5

export PYTHONPATH=/home/phucnda/open3d/iccvw_resource/ICCVW
CUDA_VISIBLE_DEVICES=0 python demo_groundedv2.py \
    --scene $arg1 \
    --classname $arg2 \
    --captions $arg3 \
    --rotate $arg4 \
    --distractor $arg5 \
    --config ../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --grounded_checkpoint ../../groundingdino_swint_ogc.pth \
    --sam_checkpoint ../../sam_vit_h_4b8939.pth \
    --box_threshold 0.6 \
    --text_threshold 0.25 \
    --device "cuda" \
    --voxel_size 0.05 \
    --clip_checkpoint ../../ovseg_clip_l_9a1909.pth

