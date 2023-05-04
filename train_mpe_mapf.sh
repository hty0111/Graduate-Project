#!/bin/sh
env="MPE"
scenario="mapf"
num_agents=10
num_obstacles=0
algo="mappo" #"rmappo" "ippo"
num_steps=50000000
n_rollout_threads=64
seed_max=7
version="v3.1"

exp="${version}_agents_${num_agents}_obs_${num_obstacles}_threads_${n_rollout_threads}"
echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./ python ./scripts/train_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_agents} --num_obstacles $[${num_obstacles}+5*${seed}] --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads ${n_rollout_threads} -num_mini_batch 1 --episode_length 30 --num_env_steps ${num_steps} \
    --ppo_epoch 10 --use_ReLU  --gain 0.01 --lr 7e-4 --critic_lr 7e-4
done

