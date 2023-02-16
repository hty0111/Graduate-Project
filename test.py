"""
Description: 
version: v1.0
Author: HTY
Date: 2023-02-07 19:57:55
"""

from envs.mpe import simple_v2, simple_adversary_v2, simple_spread_v2
import random
import numpy as np

def random_demo(env, render=True, episodes=1):
    """Runs an env object with random actions."""
    total_reward = 0
    completed_episodes = 0

    while completed_episodes < episodes:
        env.reset() # simple_env.py/reset(), simple_spread.py/reset_world()
        for agent in env.agent_iter():
            if render:
                env.render()    # simple_env.py/render(), simple_env.py/draw()

            obs, reward, termination, truncation, _ = env.last()    # 返回当前智能体上一步执行完成时的累计奖励等
            total_reward += reward
            if termination or truncation:
                action = None
            elif isinstance(obs, dict) and "action_mask" in obs:
                action = random.choice(np.flatnonzero(obs["action_mask"]))
            else:
                # action = env.action_space(agent).sample()
                action = [0.5, 0.05, 0.05, 0.05, 0.05]

            # 最后一个智能体时，会更新整个地图
            env.step(action)    # simple_env.py/step()

        completed_episodes += 1

    if render:
        env.close()

    print("Average total reward", total_reward / episodes)

    return total_reward

if __name__ == "__main__":
    # env = simple_spread_v2.env(max_cycles=100, continuous_actions=True, render_mode='human')
    # # env = simple_adversary_v2.env(render_mode='human')
    # random_demo(env, render=True, episodes=100)
    import numpy as np

    for i in np.arange(0, 5, 1):
        print(i)

