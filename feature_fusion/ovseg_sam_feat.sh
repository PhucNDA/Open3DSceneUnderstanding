export PYTHONPATH=/home/tinhn/3D/ICCVW
CUDA_VISIBLE_DEVICES=6 python ovseg_sam_feat.py --config-file ../ovseg/configs/ovseg_swinB_vitL_demo.yaml --sam_checkpoint ../../sam_vit_h_4b8939.pth --class-names 'sofa' 'presents'  --clip_checkpoint ../../ovseg_clip_l_9a1909.pth

