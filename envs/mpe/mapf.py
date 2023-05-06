"""
Author: HTY
Email: 1044213317@qq.com
Date: 2023-02-17 14-10
Description: Scenario for Multi-Agent Path Finding

|--------------------|----------------------------------------|
| Actions            | Discrete                    |
| Parallel API       | Yes                                    |
| Manual Control     | No                                     |
| Agents             | `agents= [agent_0, agent_1, agent_2]`  |
| Agents             | 100                                    |
| Action Shape       | dim(t) * dim(v) * dim(s) * dim(d)      |
| Action Values      | Discrete(Action Shape)                 |
| Observation Shape  | (4)                                    |
| Observation Values | (-inf,inf)                             |
| State Shape        | (4,)                                   |
| State Values       | (-inf,inf)                             |

Observation space: `[self_vel, landmark_rel_position]`

### Arguments

``` python
mapf_v1.env(max_cycles=25, continuous_actions=False)
```

`max_cycles`:  number of frames (a step for each agent) until game terminates
`continuous_actions`: Whether agent action spaces are discrete(default) or continuous

"""

import numpy as np
from pettingzoo.utils.conversions import parallel_wrapper_fn

from .entities import Agent, Landmark, Obstacle, ReferenceLine, World
from .base_env import BaseEnv, make_env
from algorithms.lattice.cubic_spline import CubicSpline2D


class raw_env(BaseEnv):
    def __init__(self, max_cycles=25, num_agents=3, num_obstacles=10, render_mode='human'):
        scenario = Scenario()
        world = scenario.make_world(num_agents, num_obstacles)
        super().__init__(
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
        )
        self.metadata["name"] = "mapf"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario:
    def make_world(self, num_agents, num_obs):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = num_agents
        num_landmarks = num_agents
        num_reference_line = num_agents
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent{i}"
            agent.collide = True
            agent.silent = True
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = f"landmark{i}"
            landmark.collide = False
            landmark.movable = False
        world.reference_lines = [ReferenceLine for i in range(num_reference_line)]
        world.obstacles = [Obstacle() for i in range(num_obs)]
        return world

    def reset_world(self, world, np_random, width, height):
        num_agents = len(world.agents)
        random_index = np_random.permutation(range(0, num_agents))

        # set properties & states for agents & landmarks
        random_flag = True
        for i, (agent, landmark) in enumerate(zip(world.agents, world.landmarks)):
            # # 智能体起点按序均匀分布
            delta_x = width / (num_agents + 1)
            agent.pos = np.array([delta_x * (i + 1), agent.size])  # bottom

            # # 起点随机
            # while True:
            #     x = np_random.uniform(agent.size, width - agent.size)
            #     success = True
            #     for a in world.agents:
            #         if a.pos is not None and np.abs(x - a.pos[0]) < agent.size + a.size:
            #             success = False
            #             break
            #     if success is True:
            #         break
            # agent.pos = np.array([x, agent.size])  # bottom

            agent.vel = np.zeros(world.dim_p)
            agent.c = np.zeros(world.dim_c)
            agent.color = np_random.uniform(0, 255, size=3)

            # 避免终点重合

            # while True:
            #     x = np_random.uniform(landmark.size, width - landmark.size)
            #     success = True
            #     for l in world.landmarks:
            #         if l.pos is not None and np.abs(x - l.pos[0]) < landmark.size + l.size:
            #             success = False
            #             break
            #     if success is True:
            #         break
            # landmark.pos = np.array([x, height - landmark.size])  # top

            if random_flag is False:
                landmark.pos = np.array([delta_x * (random_index[i - 1] + 1), height - landmark.size])
                random_flag = True
            elif random_index[i] == i and i != num_agents - 1:
                landmark.pos = np.array([delta_x * (random_index[i + 1] + 1), height - landmark.size])
                random_flag = False
            elif i == num_agents - 1:
                landmark.pos = np.array([delta_x * (random_index[i] + 1) + 0.1, height - landmark.size])
            else:
                landmark.pos = np.array([delta_x * (random_index[i] + 1), height - landmark.size])

            landmark.vel = np.zeros(world.dim_p)
            landmark.color = agent.color

            # calculate reference lines
            world.reference_lines[i] = ReferenceLine(world.agents[i].pos, world.landmarks[i].pos)

            # set properties for obstacles
        for obstacle in world.obstacles:
            obstacle.pos = np.array([np_random.uniform(obstacle.size + world.landmarks[0].size,
                                                       width - obstacle.size - world.landmarks[0].size),
                                     np_random.uniform(obstacle.size + world.landmarks[0].size,
                                                       height - obstacle.size - world.landmarks[0].size)])
            obstacle.color = (0, 0, 0)
            obstacle.size = 0.5

    def benchmark_data(self, agent, world):
        reward = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for lm in world.landmarks:
            dists = [
                np.sqrt(np.sum(np.square(a.pos - lm.pos)))
                for a in world.agents
            ]
            min_dists += min(dists)
            reward -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    reward -= 1
                    collisions += 1
        return reward, collisions, min_dists, occupied_landmarks

    def is_collision(self, entity1, entity2):
        delta_pos = entity1.pos - entity2.pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = entity1.size + entity2.size
        return True if dist < dist_min else False

    def reward(self, agent: Agent, world: World, infos):
        """ collision; distance to goal; velocity; lateral offset; """
        rew = 0

        # collision
        if agent.collide:
            for a in world.agents:
                if a is not agent and infos[a.name]['done'] is False and self.is_collision(agent, a):
                    rew -= 0.5

            for obs in world.obstacles:
                if self.is_collision(agent, obs):
                    rew -= 1

        # distance to goal
        # if np.hypot(agent.pos[0] - landmark.pos[0], agent.pos[1] - landmark.pos[1]) < 5:
        #     rew += 10

        # velocity

        return rew

    def observation(self, agent: Agent, landmark: Landmark, world: World):
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            other_pos.append(other.pos - agent.pos)

        obstacles_pos = [obstacle.pos for obstacle in world.obstacles]
        return np.concatenate(
            [agent.vel] + [agent.pos] + [landmark.pos] + obstacles_pos
        )

    #     agent_vel = agent.vel / 5
    #     agent_pos = self.world_to_net(*agent.pos)
    #     landmark_pos = landmark.pos - agent.pos
    #     landmark_pos = self.world_to_net(*landmark_pos)
    #     obstacles_pos = [self.world_to_net(*obstacle.pos) for obstacle in world.obstacles]
    #
    #     return np.concatenate(
    #         [agent_vel] + [agent_pos] + [landmark_pos] + obstacles_pos
    #     )
    #
    # def world_to_net(self, world_x, world_y):
    #     net_x = (world_x - 50 / 2) / (50 / 2)
    #     net_y = (world_y - 100 / 2) / (100 / 2)
    #     return net_x, net_y