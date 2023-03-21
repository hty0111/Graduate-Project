"""
Author: HTY
Email: 1044213317@qq.com
Date: 2023-02-19 21:53
Description: Modified from MPE to work with tasks for multi-agent path finding
"""

from algorithms.lattice.frenet_lattice import LatticePlanner

import os
import numpy as np
import pygame
import gymnasium
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector

import matplotlib.pyplot as plt


def make_env(raw_env):
    def env_fn(**kwargs):
        env = raw_env(**kwargs)
        env = wrappers.AssertOutOfBoundsWrapper(env)
        env = wrappers.OrderEnforcingWrapper(env)
        return env

    return env_fn


class BaseEnv(AECEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
        "render_fps": 10,
    }

    def __init__(
        self,
        scenario,
        world,
        max_cycles,
        render_mode=None,
    ):
        super().__init__()  # AECEnv类的初始化，无作用

        self.render_mode = render_mode
        pygame.init()
        self.viewer = None
        self.width = 50
        self.height = 100
        self.canvas_scale = 10
        self.screen = pygame.Surface([self.width * self.canvas_scale, self.height * self.canvas_scale])
        self.game_font = pygame.freetype.Font(os.path.join(os.path.dirname(__file__), "secrcode.ttf"), 15)

        # Set up the drawing window
        self.renderOn = False
        self.seed()

        self.max_cycles = max_cycles
        self.scenario = scenario
        self.world = world
        self.scenario.reset_world(self.world, self.np_random, self.width, self.height)
        # self.agent_landmark = {agent: landmark for agent, landmark in zip(self.world.agents, self.world.landmarks)}

        self.agents = [agent.name for agent in self.world.agents]
        self.possible_agents = self.agents[:]
        self._index_map = {
            agent.name: idx for idx, agent in enumerate(self.world.agents)
        }
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.planner = LatticePlanner()
        self.step_dt = self.planner.MIN_T   # [s]

        # set spaces
        self.action_spaces = dict()
        self.observation_spaces = dict()
        state_dim = 0
        for agent in self.world.agents:
            action_dim = self.planner.sample_dim

            obs_dim = len(self.scenario.observation(agent, self.world.landmarks[self._index_map[agent.name]], self.world))
            state_dim += obs_dim
            self.action_spaces[agent.name] = spaces.Discrete(action_dim)
            self.observation_spaces[agent.name] = spaces.Box(
                low=-np.float32(np.inf),
                high=+np.float32(np.inf),
                shape=(obs_dim,),
                dtype=np.float32,
            )

        # state是每个agent的observation
        self.state_space = spaces.Box(
            low=-np.float32(np.inf),
            high=+np.float32(np.inf),
            shape=(state_dim,),
            dtype=np.float32,
        )

        self.steps = 0
        self.current_actions = [None] * self.num_agents

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    def seed(self, seed=None):
        self.np_random, _ = seeding.np_random(seed)

    def observe(self, agent: str):
        return self.scenario.observation(
            self.world.agents[self._index_map[agent]], self.world.landmarks[self._index_map[agent]], self.world
        ).astype(np.float32)

    def state(self):
        states = tuple(
            self.scenario.observation(
                self.world.agents[self._index_map[agent]], self.world.landmarks[self._index_map[agent]], self.world
            ).astype(np.float32)
            for agent in self.possible_agents
        )
        return np.concatenate(states, axis=None)

    def reset(self, seed=None, return_info=False, options=None):
        if seed is not None:
            self.seed(seed=seed)
        self.scenario.reset_world(self.world, self.np_random, self.width, self.height)

        self.agents = self.possible_agents[:]
        self.rewards = {name: 0.0 for name in self.agents}
        self._cumulative_rewards = {name: 0.0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: False for name in self.agents}  # 用info存储done，绕过termination和truncation

        self.agent_selection = self._agent_selector.reset()
        self.steps = 0

        self.current_actions = [None] * self.num_agents

    def _execute_world_step(self):
        # set action for each agent
        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i]
            reference_line = self.world.reference_lines[i]
            plt.plot(reference_line.x, reference_line.y)
            plt.plot(agent.pos[0], agent.pos[1], "gx")
            plt.xlim(0, self.width)
            plt.ylim(0, self.height)
            if agent.movable is True and action is not None:
                agent.pos, agent.vel = self._set_action(agent.pos, agent.vel, action, reference_line, self.step_dt)
            reward = float(self.scenario.reward(agent, self.world.landmarks[self._index_map[agent.name]], self.world))
            self.rewards[agent.name] = reward if action is not None else 0  # 如果done就把reward设为0
            self.infos[agent.name] = self.done(agent)

            plt.plot(agent.pos[0], agent.pos[1], "rx")


        # plt.show()
        # self.world.step()

    def done(self, agent) -> bool:
        landmark = self.world.landmarks[self._index_map[agent.name]]
        if np.hypot(agent.pos[0] - landmark.pos[0], agent.pos[1] - landmark.pos[1]) < 5:    # close enough to goal
            return True
        elif agent.pos[0] > self.width or agent.pos[1] > self.height:   # out of bounds
            return True
        else:
            return False

    # set env action for a particular agent
    def _set_action(self, pos, vel, action, reference_line, dt) -> tuple:
        s, d, s_d, d_d = self.planner.cartesian_to_frenet(reference_line, *pos, *vel)
        paths = self.planner.calc_frenet_paths(s, s_d, 0, d, d_d, 0)

        path = paths[action]
        s_step = path.lon_traj.calc_point(dt)
        s_d_step = path.lon_traj.calc_first_derivative(dt)
        d_step = path.lat_traj.calc_point(dt)
        d_d_step = path.lat_traj.calc_first_derivative(dt)
        pos_x, pos_y, vel_x, vel_y = self.planner.frenet_to_cartesian(reference_line, s_step, d_step, s_d_step, d_d_step)

        return np.array([pos_x, pos_y]), np.array([vel_x, vel_y])

    def step(self, action):
        if self.infos[self.agent_selection] is True:
            action = None

        current_idx = self._index_map[self.agent_selection]
        self.current_actions[current_idx] = action

        if self._agent_selector.is_last():  # update the whole world
            self._execute_world_step()
            self.steps += 1
            if self.steps >= self.max_cycles:
                for a in self.agents:
                    self.infos[a] = True
        else:
            self._clear_rewards()

        if self.render_mode == "human":
            self.render()

        self.agent_selection = self._agent_selector.next()

    def enable_render(self, mode="human"):
        if not self.renderOn and mode == "human":
            self.screen = pygame.display.set_mode(self.screen.get_size())
            self.renderOn = True

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        self.enable_render(self.render_mode)

        observation = np.array(pygame.surfarray.pixels3d(self.screen))
        if self.render_mode == "human":
            self.draw()
            pygame.display.flip()
        return (
            np.transpose(observation, axes=(1, 0, 2))
            if self.render_mode == "rgb_array"
            else None
        )

    def draw(self):
        # clear screen
        self.screen.fill((255, 255, 255))

        # update geometry and text positions
        for agent, landmark, reference_line in zip(self.world.agents, self.world.landmarks, self.world.reference_lines):
            # agent
            x, y = self.world2map(*agent.pos)
            pygame.draw.circle(self.screen, agent.color, (x, y), agent.size * self.canvas_scale)
            pygame.draw.circle(self.screen, (0, 0, 0), (x, y), agent.size * self.canvas_scale, 1)  # borders

            # landmark
            x, y = self.world2map(*landmark.pos)
            pygame.draw.circle(self.screen, landmark.color, (x, y), landmark.size * self.canvas_scale)
            pygame.draw.circle(self.screen, (0, 0, 0), (x, y), landmark.size * self.canvas_scale, 1)  # borders

            # reference line
            points = [self.world2map(x, y) for x, y in zip(reference_line.x, reference_line.y)]
            pygame.draw.lines(self.screen, agent.color, False, points)

        for obstacle in self.world.obstacles:
            x, y = self.world2map(*obstacle.pos)
            pygame.draw.circle(self.screen, obstacle.color, (x, y), obstacle.size * self.canvas_scale)
            pygame.draw.circle(self.screen, (0, 0, 0), (x, y), obstacle.size * self.canvas_scale, 1)

            # text
            # self.game_font.render_to(
            #     surf=self.screen, dest=(x, y), text=entity.name, fgcolor=(0, 0, 0)
            # )

    def close(self):
        if self.renderOn:
            pygame.event.pump()
            pygame.display.quit()
            self.renderOn = False

    def world2map(self, world_x, world_y):
        x = world_x
        y = self.height - world_y  # this makes the display mimic the old pyglet setup (i.e. flips image)
        x *= self.canvas_scale
        y *= self.canvas_scale
        return x, y
