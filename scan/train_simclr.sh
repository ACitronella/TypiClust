#! /bin/bash
seed=1
for fold in 0 1 2 3
do
	python blink_simclr.py --config_env configs/env.yml --config_exp configs/pretext/simclr64_blink_fold$fold.yml --seed $seed 
done

