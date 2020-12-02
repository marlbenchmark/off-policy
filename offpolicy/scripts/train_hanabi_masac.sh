#!/bin/sh
env="Hanabi"
hanabi="Hanabi-Small"
num_agents=2
algo="masac"
exp="non-dict"
seed_max=1

echo "env is ${env}, hanabi game is ${hanabi}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in $(seq ${seed_max}); do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=6 python train/train_hanabi.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --hanabi_name ${hanabi} --num_agents ${num_agents} --seed ${seed} --episode_length 40 --batch_size 64 --hidden_size 512 --layer_N 2 --lr 0.000025 --buffer_size 5000 --train_interval_episode 1 --actor_train_interval_step 1 --epsilon_anneal_time 80000 --target_entropy_coef 0.1 --num_env_steps 100000000
    echo "training is done!"
done
