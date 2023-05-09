"""
Author: HTY
Email: 1044213317@qq.com
Date: 2023-05-08 21:08
Description:
"""

import os
from pathlib import Path
import numpy as np
import scipy as sp
import pandas as pd
import json
from matplotlib import pyplot as plt


def read_data(version):
    prefix = '/home/hty/Desktop/Graduation-Project/results/MPE/mapf/mappo/'
    suffix = '/logs/'
    root_dir = prefix + version + suffix
    file_name = 'summary.json'
    data_path = root_dir + file_name
    if os.path.exists(data_path) is False:
        exit(0)
    with open(data_path) as f:
        data = json.load(f)

    # for key in data.keys():
    #     print(key)

    value_loss = 'value_loss/value_loss'
    policy_loss = 'policy_loss/policy_loss'
    dist_entropy = 'dist_entropy/dist_entropy'
    actor_grad_norm = 'actor_grad_norm/actor_grad_norm'
    critic_grad_norm = 'critic_grad_norm/critic_grad_norm'
    ratio = 'ratio/ratio'

    collision = 'rewards/collision/rewards/collision'
    path = 'rewards/path/rewards/path'
    goal = 'rewards/goal/rewards/goal'
    rewards = 'rewards/average_episode_rewards/rewards/average_episode_rewards'

    collision_early = 'rewards/early/collision/rewards/early/collision'
    path_early = 'rewards/early/path/rewards/early/path'
    goal_early = 'rewards/early/goal/rewards/early/goal'
    rewards_early = 'rewards/early/average_episode_rewards/rewards/early/average_episode_rewards'

    collision_single = 'agent0/collision/agent0/collision'
    path_single = 'agent0/path/agent0/path'
    goal_single = 'agent0/goal/agent0/goal'

    collision_early_single = 'agent0/early/collision/agent0/early/collision'
    path_early_single = 'agent0/early/path/agent0/early/path'
    goal_early_single = 'agent0/early/goal/agent0/early/goal'

    steps = 20

    data_type = 'path'

    # data_early = np.array(data[root_dir + collision_early])
    # data_later = np.array(data[root_dir + collision])

    data_early = np.array(data[root_dir + path_early])
    data_later = np.array(data[root_dir + path])

    # data_early = np.array(data[root_dir + goal_early])
    # data_later = np.array(data[root_dir + goal])

    data = np.concatenate((data_early[:4, 1:], data_later[:steps, 1:]))

    # data = np.concatenate((data_early[:, 1:], data_later[198:steps, 1:]))
    # data = data_later[:steps, 1:]
    # data = data_early[:150, 1:]

    x = data[:, 0] / 1e5
    y = data[:, 1]
    if data_type == 'collision' or data_type == 'path':
        y = -y
    elif data_type == 'goal':
        y = y / 2

    # y = sp.signal.savgol_filter(y, 20, 5)

    return x, y


def read_json():
    version = 'v5.1_agents_5_obs_0_threads_128/run1'
    x_5_0, y_5_0 = read_data(version)

    version = 'v5.1_agents_10_obs_0_threads_128/run1'
    x_10_0, y_10_0 = read_data(version)

    plt.figure()
    # plt.title('Collision')
    # plt.title('Reach Goal')
    # plt.title('Kinetic Model')
    plt.plot(x_5_0, y_5_0, label='5 agents & 0 obstacles')
    plt.plot(x_10_0, y_10_0, label='10 agents & 0 obstacles')
    plt.xlabel('Steps [x 1e5]')
    plt.ylabel('Collision Times')
    # plt.ylabel('Reach Goal Times')
    # plt.ylabel('Kinetic Failure Times')
    plt.grid(True, ls='--')
    plt.legend()
    plt.show()


def read_csv():
    data_file = 'average_reward.csv'
    data_path = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]) / 'data' / data_file
    if os.path.exists(data_path) is False:
        exit(0)
    data = pd.read_csv(data_path)
    data = data.iloc[1:, 1:, ].to_numpy()

    x = data[:, 0]
    y = data[:, 1]
    y = (y - y.min) / (y.max() - y.min())

    plt.figure()
    plt.title('Collision')
    plt.plot(x, y)
    plt.xlabel('steps')
    plt.ylabel('rewards')
    plt.grid(True, ls='--')
    plt.show()


if __name__ == "__main__":
    read_json()
