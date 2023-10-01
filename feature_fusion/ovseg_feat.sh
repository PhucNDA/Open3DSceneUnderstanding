export PYTHONPATH=/home/tinhn/3D/ICCVW
CUDA_VISIBLE_DEVICES=5 python ovseg_feat.py --config-file ../ovseg/configs/ovseg_swinB_vitL_demo.yaml --class-names 'sofa' 'presents'  --opts MODEL.WEIGHTS ../../ovseg_swinbase_vitL14_ft_mpt.pth

