"""
Author: HTY
Email: 1044213317@qq.com
Date: 2023-02-21 14:13
Description: 
"""

import time
import numpy as np
import torch
from runner.shared.base_runner import Runner
from utils.typecasting import t2n
import wandb
import imageio


class MPERunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""

    def __init__(self, config):
        super(MPERunner, self).__init__(config)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # take actions by policy
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)

                # step: mapf.py/observation() & base_env.step(), shape: (num_envs * num_agents * dim)
                observations, rewards, terminations, truncations, infos = self.envs.step(actions)

                # dones = np.array(infos)
                dones = np.zeros((self.n_rollout_threads, self.num_agents))
                for i_env in np.arange(self.n_rollout_threads):
                    for i_agent in np.arange(self.num_agents):
                        dones[i_env][i_agent] = infos[i_env][i_agent]['done']

                # insert data into buffer
                data = observations, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic
                self.insert(data)

                if np.all(dones):
                    break

            self.buffer.step = 0
            # compute return and update network
            self.compute()
            train_infos = self.train()
            self.envs.reset()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            # log information
            if (episode + 1) % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num time steps {}/{}, FPS {}.\n"
                      .format(self.args.scenario_name,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                # env_infos = {}
                # for agent_id in range(self.num_agents):
                #     idv_rews = []
                #     for info in infos:
                #         if 'reward' in info[agent_id].keys():
                #             idv_rews.append(info[agent_id]['reward']['collision'])
                #     agent_k = 'agent%i/rewards/' % agent_id
                #     env_infos[agent_k] = idv_rews

                env_infos = {}
                for agent_i, info in enumerate(infos[0]):   # 在
                    for type, reward in info['reward'].items():
                        env_infos[f'agent{agent_i}/reward/{type}'] = reward

                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        observations = self.envs.reset()  # (num_envs, num_agents, obs_dim)

        # replay buffer
        if self.use_centralized_V:
            self_state_dim = observations.shape[2] - 2 * self.num_obstacles
            self_state, obstacles = observations[:, :, :self_state_dim], observations[:, :, self_state_dim:]
            state = self_state.reshape(self.n_rollout_threads, -1)
            state = np.expand_dims(state, 1).repeat(self.num_agents, axis=1)
            # (num_envs, num_agents, state_dim)
            state = np.concatenate((state, obstacles), axis=2)
            # print("self state: ", self_state)
            # print("obstacles", obstacles)
            # print("state: ", state)
        else:
            state = observations

        # 在buffer中存入第一组数据
        self.buffer.state[0] = state.copy()
        self.buffer.observations[0] = observations.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        # concatenate: (2, 2, 10) --> (4, 10), (threads, num_agents, obs_dim)
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.state[step]),
                                              np.concatenate(self.buffer.observations[step]),
                                              np.concatenate(self.buffer.rnn_states[step]),
                                              np.concatenate(self.buffer.rnn_states_critic[step]),
                                              np.concatenate(self.buffer.masks[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(t2n(value), self.n_rollout_threads))  # (4, 10) --> (2, 2, 10)
        actions = np.array(np.split(t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(t2n(rnn_states_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        observations, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        # expand dim
        rewards = np.expand_dims(rewards, axis=-1)

        if self.use_centralized_V:
            self_state_dim = observations.shape[2] - 2 * self.num_obstacles
            self_state, obstacles = observations[:, :, :self_state_dim], observations[:, :, self_state_dim:]
            state = self_state.reshape(self.n_rollout_threads, -1)
            state = np.expand_dims(state, 1).repeat(self.num_agents, axis=1)
            # (num_envs, num_agents, state_dim)
            state = np.concatenate((state, obstacles), axis=2)
        else:
            state = observations

        self.buffer.insert(state, observations, rnn_states, rnn_states_critic, actions, action_log_probs, values,
                           rewards, masks)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                                   np.concatenate(eval_rnn_states),
                                                                   np.concatenate(eval_masks),
                                                                   deterministic=True)
            eval_actions = np.array(np.split(t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(t2n(eval_rnn_states), self.n_eval_rollout_threads))

            if self.eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i] + 1)[eval_actions[:, :, i]]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
            elif self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
            else:
                raise NotImplementedError

            # Observ reward and next observations
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
        eval_average_episode_rewards = np.mean(eval_env_infos['eval_average_episode_rewards'])
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)
