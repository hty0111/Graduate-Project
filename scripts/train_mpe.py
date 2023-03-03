"""
Author: HTY
Email: 1044213317@qq.com
Date: 2023-02-19 22:00
Description:
"""

import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from config import get_config
from envs.env_wrappers import SubprocShareVecEnv, DummyShareVecEnv
from envs.mpe import mapf_v1

"""Train script for MPEs."""


def make_train_env(args):
    def get_env_fn(rank):
        def init_env():
            env = mapf_v1.parallel_env(num_agents=args.num_agents)
            env.reset(seed=args.seed + rank * 1000)
            return env
        return init_env

    if args.n_rollout_threads == 1:
        return DummyShareVecEnv([get_env_fn(0)])
    else:
        return SubprocShareVecEnv([get_env_fn(i) for i in range(args.n_rollout_threads)])


def make_eval_env(args):
    def get_env_fn(rank):
        def init_env():
            env = mapf_v1.parallel_env(num_agents=args.num_agents)
            env.reset(seed=args.seed * 50000 + rank * 10000)
            return env
        return init_env

    if args.n_rollout_threads == 1:
        return DummyShareVecEnv([get_env_fn(0)])
    else:
        return SubprocShareVecEnv([get_env_fn(i) for i in range(args.n_eval_rollout_threads)])


def main(argv):
    args = get_config(argv)

    if args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        args.use_recurrent_policy = True
        args.use_naive_recurrent_policy = False
    elif args.algorithm_name == "mappo":
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        args.use_recurrent_policy = False
        args.use_naive_recurrent_policy = False
    elif args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False")
        args.use_centralized_V = False
    else:
        raise NotImplementedError

    assert (args.share_policy is True and args.scenario_name == 'simple_speaker_listener') is False, (
        "The simple_speaker_listener scenario can not use shared policy. Please check the config.py.")

    # cuda
    if args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(args.n_training_threads)
        if args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(args.n_training_threads)

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") \
              / args.env_name / args.scenario_name / args.algorithm_name / args.experiment_name

    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if args.use_wandb:
        run = wandb.init(config=args,
                         project=args.env_name,
                         entity=args.user_name,
                         notes=socket.gethostname(),
                         name=str(args.algorithm_name) + "_" + str(args.experiment_name)
                              + "_seed" + str(args.seed),
                         group=args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exist_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                              str(folder.name).startswith('run')]
            if len(exist_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exist_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    # 设置进程名称
    setproctitle.setproctitle(str(args.algorithm_name) + "-" + str(args.env_name) + "-" +
                              str(args.experiment_name) + "@" + str(args.user_name))

    # seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # env init
    envs = make_train_env(args)
    eval_envs = make_eval_env(args) if args.use_eval else None

    config = {
        "args": args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": args.num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if args.share_policy:
        from runner.shared.mpe_runner import MPERunner as Runner
    else:
        from runner.separated.mpe_runner import MPERunner as Runner

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if args.use_wandb:
        run.finish()
    else:
        runner.writer.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writer.close()


if __name__ == "__main__":
    main(sys.argv[1:])
