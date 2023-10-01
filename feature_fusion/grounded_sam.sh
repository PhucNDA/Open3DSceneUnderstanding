export PYTHONPATH=/home/tinhn/3D/ICCVW
CUDA_VISIBLE_DEVICES=6 python grounded_sam.py \
  --config ../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint ../../groundingdino_swint_ogc.pth \
  --sam_checkpoint ../../sam_vit_h_4b8939.pth \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --text_prompt "Present. Sofa." \
  --device "cuda" \
  --config-file ../ovseg/configs/ovseg_swinB_vitL_demo.yaml \
  --opts MODEL.WEIGHTS ../../ovseg_swinbase_vitL14_ft_mpt.pth