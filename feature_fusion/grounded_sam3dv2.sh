export PYTHONPATH=/home/tinhn/3D/ICCVW
CUDA_VISIBLE_DEVICES=6 python grounded_sam3dv2.py \
  --config ../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint ../../groundingdino_swint_ogc.pth \
  --sam_checkpoint ../../sam_vit_h_4b8939.pth \
  --box_threshold 0.5 \
  --text_threshold 0.25 \
  --text_prompt "Sofa. Presents." \
  --device "cuda" \
  --voxel_size 0.05 \
  --config-file ../ovseg/configs/ovseg_swinB_vitL_demo.yaml \
  --opts MODEL.WEIGHTS ../../ovseg_swinbase_vitL14_ft_mpt.pth