#! /bin/bash
gpu_id='1'

model_graveyard=../model_graveyard_rerunboth2

# for fold_idx in 0 1 2 3 4
# do
#     python train_al.py --cfg "../configs/blink3/random-finetune-batch_size10-fold${fold_idx}.yaml" --gpu_id $gpu_id --model_graveyard $model_graveyard
# done

python train_al_progressive_retrain_from_scratch.py --cfg "../configs/blinkprogressively_add_in_lset/simclr128_probcover.yaml" --gpu_id $gpu_id --model_graveyard $model_graveyard
python train_al_progressive_retrain_from_scratch.py --cfg "../configs/blinkprogressively_add_in_lset/simclr128_embdiff.yaml" --gpu_id $gpu_id --model_graveyard $model_graveyard
python train_al_progressive_retrain_from_scratch.py --cfg "../configs/blinkprogressively_add_in_lset/random score.yaml" --gpu_id $gpu_id --model_graveyard $model_graveyard
