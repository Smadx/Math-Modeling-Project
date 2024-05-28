CUDA_VISIBLE_DEVICES=7 \
nohup accelerate launch --config_file accelerate_config.yaml train.py > train.log 2>&1 &