#!/bin/sh
env="Hanabi"
hanabi="Hanabi-Small"
num_agents=2
algo="vdn"
exp="debug"
seed_max=1

echo "env is ${env}, hanabi game is ${hanabi}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in $(seq ${seed_max}); do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=3 python train/train_hanabi.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --hanabi_name ${hanabi} --num_agents ${num_agents} --seed ${seed} --episode_length 80 --batch_size 64 --hidden_size 512 --layer_N 2 --lr 2.5e-5 --buffer_size 5000 --train_interval_episode 4 --use_soft_update --hard_update_interval_episode 500 --epsilon_anneal_time 80000 --num_env_steps 100000000 --eval_interval 50000 --num_eval_episodes 32 --use_value_active_masks --use_wandb
    echo "training is done!"
done
