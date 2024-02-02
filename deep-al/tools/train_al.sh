#! /bin/bash
gpu_id='0'
model_graveyard=../model_graveyard/

for fold_idx in 0 1 2 3 4
do
    python train_al.py --cfg "../configs/blink2/fold${fold_idx} batch_size10 random.yaml" --gpu_id $gpu_id --model_graveyard $model_graveyard
done
