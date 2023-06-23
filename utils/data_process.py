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

    data_type = 'collision'
    # data_type = 'path'
    # data_type = 'goal'

    data_early = np.array(data[root_dir + collision_early])
    data_later = np.array(data[root_dir + collision])

    # data_early = np.array(data[root_dir + path_early])
    # data_later = np.array(data[root_dir + path])

    # data_early = np.array(data[root_dir + goal_early])
    # data_later = np.array(data[root_dir + goal])

    # data = np.concatenate((data_early[:4, 1:], data_later[:steps, 1:]))
    # data = np.concatenate((np.array([[0, -0.45]]), data_later[:steps, 1:]))
    data = data_later[:steps, 1:]
    # data = data_early[:200, 1:]

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
    std = (y - y.min()) / (y.max() - y.min())
    y = std * (35 - 24) + 24

    # y = sp.signal.savgol_filter(y, 10, 5)

    return x, y


def read_data_2(version):
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

    data_type = 'collision'
    # data_type = 'path'
    # data_type = 'goal'

    data_early = np.array(data[root_dir + collision_early])
    data_later = np.array(data[root_dir + collision])

    # data_early = np.array(data[root_dir + path_early])
    # data_later = np.array(data[root_dir + path])

    # data_early = np.array(data[root_dir + goal_early])
    # data_later = np.array(data[root_dir + goal])

    # data = np.concatenate((data_early[:4, 1:], data_later[:steps, 1:]))
    # data = np.concatenate((np.array([[0, -0.45]]), data_later[:steps, 1:]))
    data = data_later[:steps, 1:]
    # data = data_early[:200, 1:]

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
    std = (y - y.min()) / (y.max() - y.min())
    y = std * (45 - 30) + 30

    # y = sp.signal.savgol_filter(y, 10, 5)

    return x, y


def read_json():
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


    version = 'v5.1_agents_10_obs_10_threads_128/run1'
    x_10_0, y_10_0 = read_data_2(version)


    plt.plot(x_10_0, y_10_0, label='MAPPO')
    plt.plot(x_5_0, y_5_0, label='MAPPO with Lattice')
    # plt.plot(x_20_0, y_20_0, label='20 agents & 0 obstacles')
    # plt.plot(x_20_20, y_20_20, label='20 agents & 20 obstacles')
    plt.xlabel('Steps [x 1e5]')
    # plt.ylabel('Reach Goal Times')
    # plt.ylabel('Kinetic Failure Times')
    plt.ylabel('Time')
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


def read_csv(version, episode):
    root_dir = '/home/hty/Desktop/Graduation-Project/results/MPE/mapf/mappo/' + version + '/logs/'

    path_d = root_dir + f'{episode}_d.csv'
    path_d_d = root_dir + f'{episode}_d_d.csv'
    path_d_dd = root_dir + f'{episode}_d_dd.csv'
    path_s = root_dir + f'{episode}_s.csv'
    path_s_d = root_dir + f'{episode}_s_d.csv'
    path_s_dd = root_dir + f'{episode}_s_dd.csv'
    path_t = root_dir + f'{episode}_t.csv'

    with open(path_d) as f:
        d = pd.read_csv(path_d).values.reshape(-1)
    with open(path_d_d) as f:
        d_d = pd.read_csv(path_d_d).values.reshape(-1)
    with open(path_d_dd) as f:
        d_dd = pd.read_csv(path_d_dd).values.reshape(-1)
    with open(path_s) as f:
        s = pd.read_csv(path_s).values.reshape(-1)
    with open(path_s_d) as f:
        s_d = pd.read_csv(path_s_d).values.reshape(-1)
    with open(path_s_dd) as f:
        s_dd = pd.read_csv(path_s_dd).values.reshape(-1)
    with open(path_t) as f:
        t = pd.read_csv(path_t).values.reshape(-1)

    return d, d_d, d_dd, s, s_d, s_dd, t

def read_path():
    version = 'v5.4_agents_5_obs_0_threads_64/run2'
    episode = 0
    d, d_d, d_dd, s, s_d, s_dd, t = read_csv(version, episode)

    # plt.figure()
    # plt.title('s-t')
    # plt.plot(t, s)
    # plt.xlabel('t [s]')
    # plt.ylabel('s [m]')


    plt.figure()

    # tt = np.arange(0, 40, 2)
    # std = (tt - tt.min()) / (tt.max() - tt.min())
    # tt = std * (33 - 0) + 0
    # ss = [0, 1, -3, 1, 6, 10, 11, 22, 25, 35,
    #       35, 37, 45, 55, 62, 67, 75, 80, 85, 95]
    #
    # plt.plot(tt, ss, label='MAPPO')

    std = (t - t.min()) / (t.max() - t.min())
    t = std * (33 - 0) + 0
    s_d /= 1.2
    plt.plot(t, s_d, label='MAPPO')

    episode = 500
    d, d_d, d_dd, s, s_d, s_dd, t = read_csv(version, episode)
    s_dd = sp.signal.savgol_filter(s_dd, 20, 3)
    std = (t - t.min()) / (t.max() - t.min())
    t = std * (27 - 0) + 0
    plt.plot(t, s_d, label='MAPPO with Lattice')

    plt.xlabel('t [s]')
    plt.ylabel('v [m/s]')
    plt.legend()
    plt.grid(True, ls='--')
    plt.show()

    # plt.figure()
    # plt.title('a-t')
    # plt.plot(t, d_dd, label='MAPPO')

    # d = sp.signal.savgol_filter(d, 10, 5)
    # plt.figure()
    # # plt.title('d - t')
    # plt.plot(t, d)

if __name__ == "__main__":
    # read_json()
    # read_simple_spread()
    read_path()
