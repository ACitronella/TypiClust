#! /bin/bash
gpu_id='0'
model_graveyard=../model_graveyard/

for fold_idx in 0 1 2 3
do
    # cfg_path="../configs/blink/al/fold0\ batch_size10\ probcover0.08_idx_standard_dist0.1.yaml"
    python train_al.py --cfg "../configs/blink/al/fold${fold_idx} batch_size10 probcover0.08_idx_standard_dist0.3.yaml" --gpu_id $gpu_id --model_graveyard $model_graveyard
done
