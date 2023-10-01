#!/bin/bash

arg1=$1
arg2=$2
arg3=$3
arg4=$4
arg5=$5
arg6=$6
arg7=$7

export PYTHONPATH=/home/phucnda/open3d/iccvw_resource/ICCVW
CUDA_VISIBLE_DEVICES=0 python demo_groundedv3.py \
    --scene $arg1 \
    --classname $arg2 \
    --captions $arg3 \
    --rotate $arg4 \
    --distractor $arg5 \
    --clip_model ViT-L-14-336 \
    --clip_checkpoint openai \
    --feat_dim 768 \
    --config ../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --grounded_checkpoint ../../groundingdino_swint_ogc.pth \
    --sam_checkpoint ../../sam_vit_h_4b8939.pth \
    --box_threshold $arg6 \
    --text_threshold $arg7 \
    --device "cuda" \
    --voxel_size 0.05 \
    --interval 10 \
    --datapath '../../Dataset/iccvw/ChallengeTestSet' \
    --exp 'versionfinal'
    
#laion2b_s39b_b160k

