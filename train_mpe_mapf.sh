#!/bin/sh
env="MPE"
scenario="mapf"
num_agents=5
num_obstacles=0
algo="mappo" #"rmappo" "ippo"
num_steps=1000000
n_rollout_threads=64
seed=1
agents_iter=4
obs_iter=3
version="v5.4"

#for agents_i in `seq ${agents_iter}`;
#do
#    num_a=$[${num_agents}+5*(${agents_i}-1)]
#    for obs_i in `seq ${obs_iter}`;
#    do
#        num_obs=$[${num_obstacles}+10*(${obs_i}-1)]
#        exp="${version}_agents_${num_a}_obs_${num_obs}_threads_${n_rollout_threads}"
#        echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
#        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./ python ./scripts/train_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
#        --scenario_name ${scenario} --num_agents ${num_a} --num_landmarks ${num_a} --num_obstacles ${num_obs} --seed ${seed} \
#        --n_training_threads 1 --n_rollout_threads ${n_rollout_threads} -num_mini_batch 1 --episode_length 30 --num_env_steps ${num_steps} \
#        --ppo_epoch 10 --use_ReLU  --gain 0.01 --lr 7e-4 --critic_lr 7e-4
#    done
#done

exp="${version}_agents_${num_agents}_obs_${num_obstacles}_threads_${n_rollout_threads}"
echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}"
echo "seed is ${seed}:"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./ python ./scripts/train_mapf.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
--scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_agents} --num_obstacles ${num_obstacles} --seed ${seed} \
--n_training_threads 1 --n_rollout_threads ${n_rollout_threads} -num_mini_batch 1 --episode_length 30 --num_env_steps ${num_steps} \
--ppo_epoch 10 --use_ReLU  --gain 0.01 --lr 7e-4 --critic_lr 7e-4

