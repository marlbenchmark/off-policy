#!/bin/sh
env="StarCraft2"
map="3m"
algo="rmaddpg"
exp="debug"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in $(seq ${seed_max}); do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed ${seed} --actor_train_interval_step 1 --tau 0.005 --num_env_steps 10000000
    echo "training is done!"
done
