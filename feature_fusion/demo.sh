export PYTHONPATH=/home/tinhn/3D/ICCVW
CUDA_VISIBLE_DEVICES=6 python demo.py \
    --config-file ../ovseg/configs/ovseg_swinB_vitL_demo.yaml \
    --scene '42446478' \
    --captions 'The sofa and presents' \
    --config ../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --grounded_checkpoint ../../groundingdino_swint_ogc.pth \
    --sam_checkpoint ../../sam_vit_h_4b8939.pth \
    --box_threshold 0.5 \
    --text_threshold 0.25 \
    --device "cuda" \
    --voxel_size 0.05 \
    --opts MODEL.WEIGHTS ../../ovseg_swinbase_vitL14_ft_mpt.pth

