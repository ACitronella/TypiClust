#! /bin/bash
gpu_id='1'
model_graveyard=../model_graveyard_but_use_mse_as_best_indicator

for fold_idx in 0 1 2 3 4
do
    python train_al.py --cfg "../configs/blink3/random-finetune-batch_size10-fold${fold_idx}.yaml" --gpu_id $gpu_id --model_graveyard $model_graveyard
done
