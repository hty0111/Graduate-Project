"""
Author: HTY
Email: 1044213317@qq.com
Date: 2023-03-03 15:59
Description: Modified from OpenAI Baselines to work with multi-agent envs under PettingZoo framework
"""

import numpy as np
from multiprocessing import Process, Pipe
from abc import ABC, abstractmethod
from baselines.common.vec_env import VecEnvWrapper, CloudpickleWrapper

from utils.util import tile_images
from utils.typecasting import dict_to_array


class ShareVecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes a batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """
    closed = False
    viewer = None
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, observation_spaces, state_space, action_spaces):
        self.observation_spaces = observation_spaces    # List[gym.spaces]
        self.state_space = state_space                  # gym.spaces
        self.action_spaces = action_spaces              # List[gym.spaces]

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode='human'):
        imgs = self.get_images()
        bigimg = tile_images(imgs)
        if mode == 'human':
            self.get_viewer().imshow(bigimg)
            return self.get_viewer().isopen
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        """
        Return RGB images from each environment
        """
        raise NotImplementedError

    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

    def get_viewer(self):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.SimpleImageViewer()
        return self.viewer


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            # transform action matrix into action dict for pettingzoo step
            actions_dict = {}
            for i in range(len(data)):
                actions_dict[env.agents[i]] = data[i][0]
            ob, reward, termination, truncation, info = map(dict_to_array, env.step(actions_dict))

            done = [b1 or b2 for b1, b2 in zip(termination, truncation)]    # done if termination or truncation
            if 'bool' in done.__class__.__name__:   # for one agent
                if done:
                    ob = dict_to_array(env.reset())
            else:
                if np.all(done):
                    ob = dict_to_array(env.reset())

            remote.send((ob, reward, termination, truncation, info))
        elif cmd == 'reset':
            ob = dict_to_array(env.reset())
            remote.send(ob)
        elif cmd == 'render':
            if data == "rgb_array":
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == "human":
                env.render(mode=data)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((dict_to_array(env.observation_spaces), env.state_space, dict_to_array(env.action_spaces)))
        else:
            raise NotImplementedError


class SubprocShareVecEnv(ShareVecEnv, ABC):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_spaces, state_space, action_spaces = self.remotes[0].recv()
        ShareVecEnv.__init__(self, observation_spaces, state_space, action_spaces)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, terminations, truncations, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(terminations), np.stack(truncations), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def render(self, mode="rgb_array"):
        for remote in self.remotes:
            remote.send(('render', mode))
        if mode == "rgb_array":   
            frame = [remote.recv() for remote in self.remotes]
            return np.stack(frame) 


# single env
class DummyShareVecEnv(ShareVecEnv, ABC):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(self,
                             dict_to_array(env.observation_spaces),
                             env.state_space,
                             dict_to_array(env.action_spaces))
        self.actions = []

    def step_async(self, actions: np.ndarray):
        action_dict = {}
        for i in range(actions.shape[1]):
            action_dict[self.envs[0].agents[i]] = actions[0][i][0]
        self.actions.append(action_dict)

    def step_wait(self):
        results = [tuple(map(dict_to_array, env.step(a))) for (a, env) in zip(self.actions, self.envs)]
        obs, rews, terminations, truncations, infos = map(np.array, zip(*results))
        dones = np.array([b1 or b2 for t1, t2 in zip(terminations, truncations) for b1, b2 in zip(t1, t2)]).reshape(
            terminations.shape)
        for (i, done) in enumerate(dones):
            if 'bool' in done.__class__.__name__:
                if done:
                    obs[i] = dict_to_array(self.envs[i].reset())
            else:
                if np.all(done):
                    obs[i] = dict_to_array(self.envs[i].reset())

        self.actions = []
        return obs, rews, terminations, truncations, infos

    def reset(self):
        obs = np.array([dict_to_array(env.reset()) for env in self.envs])
        return obs

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError
