#! /bin/bash
seed=132
for fold in 0 1 2 3 4
do
	python blink_simclr.py --config_env configs/env.yml --config_exp configs/pretext/simclr128_blink_fold${fold}.yml --seed $seed --gpu_id 1
done

