"""
Description: 
version: v1.0
Author: HTY
Date: 2023-02-07 19:57:55
"""

from envs.mpe import simple_spread_v2, mapf_v1
import random
import numpy as np
from pettingzoo.test import api_test, parallel_api_test


def random_demo(env, render=True, episodes=1):
    """Runs an env object with random actions."""
    total_reward = 0
    completed_episodes = 0

    while completed_episodes < episodes:
        observations = env.reset()     # simple_env.py/reset(), simple_spread.py/reset_world()

        if env.__class__.__name__ == "aec_to_parallel_wrapper":
            if render:
                env.render()    # simple_env.py/render(), simple_env.py/draw()
            actions = {agent: env.action_space(agent).sample() for agent in
                       env.agents}  # this is where you would insert your policy
            observations, rewards, terminations, truncations, infos = env.step(actions)

        else:
            for agent in env.agent_iter():
                if render:
                    env.render()    # simple_env.py/render(), simple_env.py/draw()

                # 返回当前智能体上一步执行完成时的累计奖励等
                obs, reward, termination, truncation, info = env.last() # simple_env.observe(), simple_spread.observation()
                total_reward += reward
                if termination or truncation:
                    action = None
                elif isinstance(obs, dict) and "action_mask" in obs:
                    action = random.choice(np.flatnonzero(obs["action_mask"]))
                else:
                    action = env.action_space(agent).sample()
                    # action = [0.5, 0.05, 0.05, 0.05, 0.05]

                # 最后一个智能体时，会更新整个地图
                env.step(action)    # simple_env.py/step()

        completed_episodes += 1

    if render:
        env.close()

    print("Average total reward", total_reward / episodes)

    return total_reward


if __name__ == "__main__":
    # env = mapf_v1.env(max_cycles=100, render_mode='human')  # 参数传给raw_env.__init__()
    env = mapf_v1.parallel_env(max_cycles=100, render_mode='human')
    # parallel_api_test(parallel_env, num_cycles=1000)
    random_demo(env, render=False, episodes=100)


