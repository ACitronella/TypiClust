#! /bin/bash
seed=132
gpu_id=1
# for pcode in A1115 A2111 A2211 A3102 A3104 C2104 C2201 C2202 C2204 C2205
# do
# 	python blink_simclr.py --config_env configs/env.yml --config_exp "configs/blinkleaveoneout/simclr128_blink_ex_${pcode} lr0.04 temp0.01.yml" --seed $seed --gpu_id $gpu_id
# 	# echo $pcode
# done

for fold_idx in 0 1 2 3 4
do 
	python blink_simclr.py --config_env configs/env.yml --config_exp "configs/blinkprogressive/simclr128 lr0.04 temp0.01 fold$fold_idx.yml" --seed $seed --gpu_id $gpu_id
done
