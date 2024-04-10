#! /bin/bash
gpu_id='1'

# for fold_idx in 0 1 2 3 4
# do
    # python train_al.py --cfg "../configs/blink3/random-on-blinking-period-finetune-batch_size10-fold${fold_idx}-round2.yaml" --gpu_id $gpu_id --model_graveyard $model_graveyard
# done


model_graveyard=../model_graveyard_but_use_mse4

for fold_idx in 0 1 2 3 4
do
    python train_al.py --cfg "../configs/blink3/random-finetune-batch_size10-fold${fold_idx}.yaml" --gpu_id $gpu_id --model_graveyard $model_graveyard
done
