#!/bin/sh
env="BlueprintConstruction"
scenario_name="empty"
num_agents=2
num_boxes=4
floor_size=4.0
algo="qmix"
exp="rewnorm-bpc"
seed_max=1

echo "env is ${env}, scenario name is ${scenario_name}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in $(seq ${seed_max}); do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=6 python train/train_hns.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario_name} --num_agents ${num_agents} --num_boxes ${num_boxes} --floor_size ${floor_size} --seed ${seed} --episode_length 200 --batch_size 256 --buffer_size 5000 --lr 7e-4 --use_soft_update --hard_update_interval_episode 900 --num_env_steps 100000000 --eval_interval 50000 --num_eval_episodes 32 --use_reward_normlization
    echo "training is done!"
done
