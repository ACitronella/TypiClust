#! /bin/bash
gpu_id='0'
model_graveyard=../model_graveyard/

for fold_idx in 0 1 2 3 4
do
    python train_al.py --cfg "../configs/blink2/simclr128-lr0.04-temp0.01_probcover0.3-1epoch-then-emb-diff-finetune-batch_size10-fold${fold_idx}.yaml" --gpu_id $gpu_id --model_graveyard $model_graveyard
done
