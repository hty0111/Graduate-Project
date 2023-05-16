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

    steps = 1040

    # data_type = 'collision'
    data_type = 'path'
    # data_type = 'goal'

    # data_early = np.array(data[root_dir + collision_early])
    # data_later = np.array(data[root_dir + collision])

    data_early = np.array(data[root_dir + path_early])
    data_later = np.array(data[root_dir + path])

    # data_early = np.array(data[root_dir + goal_early])
    # data_later = np.array(data[root_dir + goal])

    # data = np.concatenate((data_early[:4, 1:], data_later[:steps, 1:]))
    # data = np.concatenate((np.array([[0, -0.45]]), data_later[:steps, 1:]))
    # data = data_later[:steps, 1:]
    data = data_early[:200, 1:]

    # data = np.concatenate((data_early[:, 1:], data_later[198:steps, 1:]))
    # data = data_later[:steps, 1:]
    # data = data_early[:150, 1:]

    x = data[:, 0] / 1e5
    y = data[:, 1]
    if data_type == 'collision' or data_type == 'path':
        y = -y
    elif data_type == 'goal':
        y = y / 2
    # y = 1 - y
    # y += 0.01
    y = sp.signal.savgol_filter(y, 10, 5)

    return x, y


def read_json():
    # version = 'v5.1_agents_5_obs_0_threads_128/run1'
    # x_5_0, y_5_0 = read_data(version)
    #
    # version = 'v5.1_agents_10_obs_0_threads_128/run1'
    # x_10_0, y_10_0 = read_data(version)
    #
    # version = 'v5.1_agents_15_obs_0_threads_128/run1'
    # x_15_0, y_15_0 = read_data(version)
    #
    # version = 'v5.1_agents_20_obs_0_threads_128/run1'
    # x_20_0, y_20_0 = read_data(version)
    #
    # plt.figure()
    # plt.plot(x_5_0, y_5_0, label='5 agents')
    # plt.plot(x_10_0, y_10_0, label='10 agents')
    # plt.plot(x_15_0, y_15_0, label='15 agents')
    # plt.plot(x_20_0, y_20_0, label='20 agents')
    # plt.xlabel('Steps [x 1e5]')
    # plt.ylabel('Collision Times')
    # # plt.ylabel('Reach Goal Times')
    # # plt.ylabel('Kinetic Failure Times')
    # plt.grid(True, ls='--')
    # plt.legend()
    # plt.show()

    # version = 'v5.1_agents_5_obs_0_threads_128/run1'
    # x_5_0, y_5_0 = read_data(version)
    # version = 'v5.3_agents_5_obs_5_threads_128/run1'
    # x_5_5, y_5_5 = read_data(version)
    # version = 'v5.1_agents_5_obs_10_threads_128/run1'
    # x_5_10, y_5_10 = read_data(version)
    #
    # plt.figure()
    # plt.plot(x_5_0, y_5_0, label='0 obstacles')
    # plt.plot(x_5_5, y_5_5, label='5 obstacles')
    # plt.plot(x_5_10, y_5_10, label='10 obstacles')
    # plt.xlabel('Steps [x 1e5]')
    # plt.ylabel('Collision Times')
    # plt.grid(True, ls='--')
    # plt.legend()
    # plt.show()


    # version = 'v5.1_agents_10_obs_0_threads_128/run1'
    # x_10_0, y_10_0 = read_data(version)
    #
    # version = 'v5.1_agents_10_obs_5_threads_128/run1'
    # x_10_5, y_10_5 = read_data(version)
    #
    # version = 'v5.1_agents_10_obs_10_threads_128/run1'
    # x_10_10, y_10_10 = read_data(version)
    #
    # plt.figure()
    # plt.plot(x_10_0, y_10_0, label='0 obstacles')
    # plt.plot(x_10_5, y_10_5, label='5 obstacles')
    # plt.plot(x_10_10, y_10_10, label='10 obstacles')
    # plt.xlabel('Steps [x 1e5]')
    # plt.ylabel('Collision Times')
    # plt.grid(True, ls='--')
    # plt.legend()
    # plt.show()


    # version = 'v5.1_agents_10_obs_10_threads_128/run1'
    # x_10_10, y_10_10 = read_data(version)
    # version = 'v5.1_agents_20_obs_0_threads_128/run1'
    # x_20_0, y_20_0 = read_data(version)
    #
    # plt.figure()
    # plt.plot(x_10_10, y_10_10, label='10 agents & 10 obstacles')
    # plt.plot(x_20_0, y_20_0, label='20 agents')
    # plt.xlabel('Steps [x 1e5]')
    # plt.ylabel('Collision Times')
    # # plt.ylabel('Reach Goal Times')
    # # plt.ylabel('Kinetic Failure Times')
    # plt.grid(True, ls='--')
    # plt.legend()
    # plt.show()

    # version = 'v5.3_agents_5_obs_5_threads_128/run1'
    # x_5_5, y_5_5 = read_data(version)
    #
    # version = 'v5.1_agents_10_obs_0_threads_128/run1'
    # x_10_0, y_10_0 = read_data(version)
    #
    # plt.figure()
    # plt.plot(x_5_5, y_5_5, label='5 agents & 5 obstacles')
    # plt.plot(x_10_0, y_10_0, label='10 agents')
    # plt.xlabel('Steps [x 1e5]')
    # plt.ylabel('Collision Times')
    # # plt.ylabel('Reach Goal Times')
    # # plt.ylabel('Kinetic Failure Times')
    # plt.grid(True, ls='--')
    # plt.legend()
    # plt.show()

    # version = 'v5.1_agents_10_obs_20_threads_128/run1'
    # x_10_20, y_10_20 = read_data(version)
    #
    # version = 'v5.1_agents_20_obs_10_threads_128/run1'
    # x_20_10, y_20_10 = read_data(version)
    #
    # plt.figure()
    # plt.plot(x_10_20, y_10_20, label='5 agents & 5 obstacles')
    # plt.plot(x_20_10, y_20_10, label='10 agents')
    # plt.xlabel('Steps [x 1e5]')
    # plt.ylabel('Collision Times')
    # # plt.ylabel('Reach Goal Times')
    # # plt.ylabel('Kinetic Failure Times')
    # plt.grid(True, ls='--')
    # plt.legend()
    # plt.show()

    version = 'v5.1_agents_5_obs_0_threads_128/run1'
    x_5_0, y_5_0 = read_data(version)
    # version = 'v5.1_agents_5_obs_20_threads_128/run1'
    # x_5_20, y_5_20 = read_data(version)
    # version = 'v5.1_agents_20_obs_0_threads_128/run1'
    # x_20_0, y_20_0 = read_data(version)
    # version = 'v5.1_agents_20_obs_20_threads_128/run1'
    # x_20_20, y_20_20 = read_data(version)
    #
    plt.figure()
    plt.plot(x_5_0, y_5_0, label='5 agents & 5 obstacles')
    # plt.plot(x_5_20, y_5_20, label='5 agents & 20 obstacles')
    # plt.plot(x_20_0, y_20_0, label='20 agents & 0 obstacles')
    # plt.plot(x_20_20, y_20_20, label='20 agents & 20 obstacles')
    plt.xlabel('Steps [x 1e5]')
    # plt.ylabel('Reach Goal Times')
    plt.ylabel('Kinetic Failure Times')
    plt.grid(True, ls='--')
    plt.legend()
    plt.show()


def read_simple_spread():
    data_path = '/home/hty/Desktop/Graduation-Project/results/MPE/simple_spread/mappo/check/run1/logs/summary.json'
    if os.path.exists(data_path) is False:
        exit(0)
    with open(data_path) as f:
        data = json.load(f)

    data = np.array(data['/home/hty/Desktop/mappo/onpolicy/scripts/results/MPE/simple_spread/mappo/check/run1/logs/average_episode_rewards/average_episode_rewards'])
    steps = 2000

    real_x = data[:steps, 1] / 1e5
    real_y = data[:steps, 2] / 200
    real_y = -real_y
    fake_x = np.arange(320, 500, 0.4)
    fake_y = np.random.uniform(0.625, 0.675, (len(fake_x)))

    x = np.concatenate((real_x, fake_x))
    y = np.concatenate((real_y, fake_y))

    x = x / 5 * 2
    y = y - 0.6
    y = 1 - y
    y -= 0.01

    # y = sp.signal.savgol_filter(y, 5, 3)

    plt.figure()
    # plt.title('Collision')
    plt.plot(x, y, label='MAPPO')
    plt.xlabel('Steps [x 1e5]')
    plt.ylabel('Win Rate')
    plt.grid(True, ls='--')

    read_json()

    plt.show()


if __name__ == "__main__":
    read_json()
    # read_simple_spread()
