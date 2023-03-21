"""
Author: HTY
Email: 1044213317@qq.com
Date: 2023-02-17 14-10
Description: Multi-Agent Path Finding

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

from .entities import Agent, Landmark, ReferenceLine, World
from .base_env import BaseEnv, make_env
from algorithms.lattice.cubic_spline import CubicSpline2D


class raw_env(BaseEnv):
    def __init__(self, max_cycles=25, num_agents=3, render_mode='human'):
        scenario = Scenario()
        world = scenario.make_world(num_agents)
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
    def make_world(self, num=2):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = num
        num_landmarks = num
        num_reference_line = num
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
        return world

    def reset_world(self, world, np_random, width, height):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([255, 182, 193])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([135, 206, 250])
        # set random initial states
        for agent, landmark, reference_line in zip(world.agents, world.landmarks, world.reference_lines):
            agent.pos = np.array([np_random.uniform(agent.size, width - agent.size), agent.size])  # bottom
            agent.vel = np.zeros(world.dim_p)
            agent.c = np.zeros(world.dim_c)
            agent.color = np_random.uniform(0, 255, size=3)
            landmark.pos = np.array([np_random.uniform(landmark.size, width - landmark.size), height - landmark.size])  # top
            landmark.vel = np.zeros(world.dim_p)
            landmark.color = agent.color
        # add reference lines
        for i in range(len(world.agents)):
            # x_list = [world.agents[i].pos[0], world.landmarks[i].pos[0]]
            # y_list = [world.agents[i].pos[1], world.landmarks[i].pos[1]]
            world.reference_lines[i] = ReferenceLine(world.agents[i].pos, world.landmarks[i].pos)


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

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.pos - agent2.pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent: Agent, landmark: Landmark, world: World):
        """ collision; distance to goal; velocity; lateral offset; """
        rew = 0

        # collision
        if agent.collide:
            for a in world.agents:
                if a is not agent and self.is_collision(a, agent):
                    rew -= 1

        # distance to goal
        # if np.hypot(agent.pos[0] - landmark.pos[0], agent.pos[1] - landmark.pos[1]) < 5:
        #     rew += 10

        # velocity


        return rew


    def observation(self, agent: Agent, landmark: Landmark, world: World):
        # get positions of all entities in this agent's reference frame
        landmark_pos = landmark.pos - agent.pos
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        # TODO use_communication & add_obstacles
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.c)
            other_pos.append(other.pos - agent.pos)
        return np.concatenate(
            [agent.vel] + [agent.pos] + [landmark_pos]
        )
