export PYTHONPATH=/home/tinhn/3D/ICCVW
CUDA_VISIBLE_DEVICES=6 python grounded_feat.py \
  --config ../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint ../../groundingdino_swint_ogc.pth \
  --sam_checkpoint ../../sam_vit_h_4b8939.pth \
  --box_threshold 0.5 \
  --text_threshold 0.25 \
  --text_prompt "Sofa. Presents." \
  --device "cuda" \
  --clip_checkpoint ../../ovseg_clip_l_9a1909.pth