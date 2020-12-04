#!/bin/sh
env="MPE"
scenario="simple_speaker_listener"
num_landmarks=3
num_agents=2
algo="mvdn"
exp="debug"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in $(seq ${seed_max}); do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=7 python train/train_mpe.py --share_policy --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --episode_length 25 --use_soft_update --lr 7e-4 --hard_update_interval_episode 500 --num_env_steps 10000000 --use_reward_normlization --use_wandb
    echo "training is done!"
done
