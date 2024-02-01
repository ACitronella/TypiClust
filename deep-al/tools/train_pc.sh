#! /bin/bash
experiment_name=simclr_probcoveropt-batchsize10
al_name=probcover # typiclust_rp
rng_seed=2
# control seed of simclr, 1 is for non random batch training, 2 is for random batch training, 3 is for random batch + positional encoding at the end training.
mkdir ../model_graveyard/$experiment_name

for fold_idx in 0 1 2 3
do
    mkdir ../model_graveyard/${experiment_name}/stepsize10n_fold$fold_idx
    delta=$(python find_delta.py --seed $rng_seed --fold_idx $fold_idx)
    # echo $delta
    python train_al.py --cfg "../configs/blink/al/fold${fold_idx} batch_size10.yaml" --al ${al_name} --exp-name auto --initial_size 0 --budget 10 --delta $delta --seed $rng_seed
    mv ../model_graveyard/*.p* ../model_graveyard/${experiment_name}/stepsize10n_fold${fold_idx}
done
