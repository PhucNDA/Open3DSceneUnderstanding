export PYTHONPATH=/home/phucnda/open3d/iccvw_resource//ICCVW
CUDA_VISIBLE_DEVICES=5 python demo_grounded.py \
    --scene '42446478' \
    --captions 'Horse picture' \
    --rotate 1 \
    --config ../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --grounded_checkpoint ../../groundingdino_swint_ogc.pth \
    --sam_checkpoint ../../sam_vit_h_4b8939.pth \
    --box_threshold 0.5 \
    --text_threshold 0.25 \
    --device "cuda" \
    --voxel_size 0.05 \
    --clip_checkpoint ../../ovseg_clip_l_9a1909.pth

