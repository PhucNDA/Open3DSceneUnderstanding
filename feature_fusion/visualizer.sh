export PYTHONPATH=/home/phucnda/open3d/iccvw_resource/ICCVW
CUDA_VISIBLE_DEVICES=0 python visualizer.py \
    --type 'test' \
    --scene 42446100 \
    --clip_checkpoint openai \
    --exp version8 \
    --feat computed_feature