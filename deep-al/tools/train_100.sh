#! /bin/bash
experiment_name=all
al_name=typiclust_rp # dont needed, since we use all dataset to train
rng_seed=2
# control seed of simclr, 1 is for non random batch training, 2 is for random batch training, 3 is for random batch + positional encoding at the end training.
mkdir ../model_graveyard/$experiment_name

for fold_idx in 0
do
    mkdir ../model_graveyard/${experiment_name}/stepsize10n_fold$fold_idx
    python train_al.py --cfg ../configs/blink/al/full_training_set_fold${fold_idx}.yaml --al ${al_name} --exp-name auto --initial_size all --budget 1 --seed $rng_seed
    mv ../model_graveyard/*.p* ../model_graveyard/${experiment_name}/stepsize10n_fold${fold_idx}
done
