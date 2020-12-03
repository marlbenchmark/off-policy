import os
import numpy as np
from itertools import chain
import wandb
import torch
from tensorboardX import SummaryWriter
import time

from offpolicy.utils.rec_buffer import RecReplayBuffer, PrioritizedRecReplayBuffer
from offpolicy.utils.util import is_discrete, is_multidiscrete, DecayThenFlatSchedule

from offpolicy.runner.rnn.base_runner import RecRunner

class MPERunner(RecRunner):
    def __init__(self, config):
        super(MPERunner, self).__init__(config)
        self.collecter = self.shared_collect_rollout if self.share_policy else self.separated_collect_rollout
        # fill replay buffer with random actions
        num_warmup_episodes = max((self.batch_size, self.args.num_random_episodes))
        self.warmup(num_warmup_episodes)
        self.start = time.time()
        self.log_clear()
    
    @torch.no_grad()
    def eval(self):
        self.trainer.prep_rollout()
        eval_infos = {}
        eval_infos['average_episode_rewards'] = []

        for _ in range(self.args.num_eval_episodes):
            env_info = self.collecter( explore=False, training_episode=False, warmup=False)
            for k, v in env_info.items():
                eval_infos[k].append(v)

        self.log_env(eval_infos, suffix="eval_")
      
    # for mpe-simple_spread and mpe-simple_reference
    
    def shared_collect_rollout(self, explore=True, training_episode=True, warmup=False):
        env_info = {}
        p_id = "policy_0"
        policy = self.policies[p_id]

        env = self.env if training_episode or warmup else self.eval_env

        obs = env.reset()
        share_obs = obs.reshape(self.num_envs, -1)

        rnn_states_batch = np.zeros((self.num_envs * len(self.policy_agents[p_id]), self.hidden_size), dtype=np.float32)
        if is_multidiscrete(self.policy_info[p_id]['act_space']):
            self.sum_act_dim = int(np.sum(self.policy_act_dim[p_id]))
        else:
            self.sum_act_dim = self.policy_act_dim[p_id]

        last_acts_batch = np.zeros((self.num_envs * len(self.policy_agents[p_id]), self.sum_act_dim), dtype=np.float32)

        # init
        episode_obs = {}
        episode_share_obs = {}
        episode_acts = {}
        episode_rewards = {}
        episode_next_obs = {}
        episode_next_share_obs = {}
        episode_dones = {}
        episode_dones_env = {}
        episode_avail_acts = {}
        episode_next_avail_acts = {}
        accumulated_rewards = {}

        episode_obs[p_id] = np.zeros((self.episode_length, *obs.shape), dtype=np.float32)
        episode_share_obs[p_id] = np.zeros((self.episode_length, *share_obs.shape), dtype=np.float32)
        episode_acts[p_id] = np.zeros((self.episode_length, self.num_envs, len(self.policy_agents[p_id]), self.sum_act_dim), dtype=np.float32)
        episode_rewards[p_id] = np.zeros((self.episode_length, self.num_envs, len(self.policy_agents[p_id]), 1), dtype=np.float32)
        episode_next_obs[p_id] = np.zeros_like((episode_obs[p_id]))
        episode_next_share_obs[p_id] = np.zeros_like((episode_share_obs[p_id]))
        episode_dones[p_id] = np.ones((self.episode_length, self.num_envs, len(self.policy_agents[p_id]), 1), dtype=np.float32)
        episode_dones_env[p_id] = np.ones((self.episode_length, self.num_envs, 1), dtype=np.float32)
        episode_avail_acts[p_id] = None
        episode_next_avail_acts[p_id] = None
        accumulated_rewards[p_id] = []

        t = 0
        episode_step = 0
        while t < self.episode_length:
            obs_batch = np.concatenate(obs)
            # get actions for all agents to step the env
            if warmup:
                # completely random actions in pre-training warmup phase
                acts_batch = policy.get_random_actions(obs_batch)
                # get new rnn hidden state
                _, rnn_states_batch, _ = policy.get_actions(obs_batch,
                                                            last_acts_batch,
                                                            rnn_states_batch)
            else:
                # get actions with exploration noise (eps-greedy/Gaussian)
                if self.algorithm_name == "rmasac":
                    acts_batch, rnn_states_batch, _ = policy.get_actions(obs_batch,
                                                                        last_acts_batch,
                                                                        rnn_states_batch,
                                                                        sample=explore)
                else:
                    acts_batch, rnn_states_batch, _ = policy.get_actions(obs_batch,
                                                                        last_acts_batch,
                                                                        rnn_states_batch,
                                                                        t_env=self.total_env_steps,
                                                                        explore=explore,
                                                                        use_target=False,
                                                                        use_gumbel=False)
            # update rnn hidden state
            rnn_states_batch = rnn_states_batch.detach().numpy()
            if not isinstance(acts_batch, np.ndarray):
                acts_batch = acts_batch.detach().numpy()
            last_acts_batch = acts_batch
            env_acts = np.split(acts_batch, self.num_envs)

            # env step and store the relevant episode information
            next_obs, rewards, dones, infos = env.step(env_acts)
            next_share_obs = next_obs.reshape(self.num_envs, -1)
            t += 1
            if training_episode:
                self.total_env_steps += self.num_envs

            dones_env = np.all(dones, axis=1)
            terminate_episodes = np.any(dones_env) or t == self.episode_length

            episode_obs[p_id][episode_step] = obs
            episode_share_obs[p_id][episode_step] = share_obs
            episode_acts[p_id][episode_step] = env_acts
            episode_rewards[p_id][episode_step] = rewards
            accumulated_rewards[p_id].append(rewards.copy())
            episode_next_obs[p_id][episode_step] = next_obs
            episode_next_share_obs[p_id][episode_step] = next_share_obs
            episode_dones[p_id][episode_step] = dones
            episode_dones_env[p_id][episode_step] = dones_env
            episode_step += 1

            obs = next_obs
            share_obs = next_share_obs

            assert self.num_envs == 1, ("only one env is support here.")
            if terminate_episodes:
                break

        if explore:
            self.num_episodes_collected += self.num_envs
            # push all episodes collected in this rollout step to the buffer
            self.buffer.insert(self.num_envs,
                               episode_obs,
                               episode_share_obs,
                               episode_acts,
                               episode_rewards,
                               episode_next_obs,
                               episode_next_share_obs,
                               episode_dones,
                               episode_dones_env,
                               episode_avail_acts,
                               episode_next_avail_acts)

        average_episode_rewards = np.mean(np.sum(accumulated_rewards[p_id], axis=0))
        env_info['average_episode_rewards'] = average_episode_rewards

        return env_info
    
    # for mpe-simple_speaker_listener
    
    def separated_collect_rollout(self, explore=True, training_episode=True, warmup=False):
        env_info = {}
        env = self.env if training_episode or warmup else self.eval_env

        obs = env.reset()
        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        agent_obs = []
        for agent_id in range(self.num_agents):
            env_obs = []
            for o in obs:
                env_obs.append(o[agent_id])
            env_obs = np.array(env_obs)
            agent_obs.append(env_obs)

        rnn_states = np.zeros((self.num_agents, self.num_envs, self.hidden_size), dtype=np.float32)

        # [agents, parallel envs, dim]
        last_acts = []
        episode_obs = {}
        episode_share_obs = {}
        episode_acts = {}
        episode_rewards = {}
        episode_next_obs = {}
        episode_next_share_obs = {}
        episode_dones = {}
        episode_dones_env = {}
        episode_avail_acts = {}
        episode_next_avail_acts = {}
        accumulated_rewards = {}

        for i, p_id in enumerate(self.policy_ids):
            if is_multidiscrete(self.policy_info[p_id]['act_space']):
                self.sum_act_dim = int(np.sum(self.policy_act_dim[p_id]))
            else:
                self.sum_act_dim = self.policy_act_dim[p_id]
            last_act = np.zeros((self.num_envs, self.sum_act_dim))
            last_acts.append(last_act)

            # init
            episode_obs[p_id] = np.zeros((self.episode_length, self.num_envs, 1, agent_obs[i].shape[-1]), dtype=np.float32)
            episode_share_obs[p_id] = np.zeros((self.episode_length, self.num_envs, 1, share_obs.shape[-1]), dtype=np.float32)
            episode_acts[p_id] = np.zeros((self.episode_length, self.num_envs, len(self.policy_agents[p_id]), self.sum_act_dim), dtype=np.float32)
            episode_rewards[p_id] = np.zeros((self.episode_length, self.num_envs, len(self.policy_agents[p_id]), 1), dtype=np.float32)
            accumulated_rewards[p_id] = []
            episode_next_obs[p_id] = np.zeros_like(episode_obs[p_id])
            episode_next_share_obs[p_id] = np.zeros_like(episode_share_obs[p_id])
            episode_dones[p_id] = np.ones((self.episode_length, self.num_envs, len(self.policy_agents[p_id]), 1), dtype=np.float32)
            episode_dones_env[p_id] = np.ones((self.episode_length, self.num_envs, 1), dtype=np.float32)
            episode_avail_acts[p_id] = None
            episode_next_avail_acts[p_id] = None

        t = 0
        episode_step = 0
        while t < self.episode_length:
            for agent_id, p_id in zip(self.agent_ids, self.policy_ids):
                policy = self.policies[p_id]
                # get actions for all agents to step the env
                if warmup:
                    # completely random actions in pre-training warmup phase
                    # [parallel envs, agents, dim]
                    act = policy.get_random_actions(agent_obs[agent_id])
                    # get new rnn hidden state
                    _, rnn_state, _ = policy.get_actions(agent_obs[agent_id],
                                                        last_acts[agent_id],
                                                        rnn_states[agent_id])
                else:
                    # get actions with exploration noise (eps-greedy/Gaussian)
                    if self.algorithm_name == "rmasac":
                        act, rnn_state, _ = policy.get_actions(agent_obs[agent_id],
                                                                last_acts[agent_id],
                                                                rnn_states[agent_id],
                                                                sample=explore)
                    else:
                        act, rnn_state, _ = policy.get_actions(agent_obs[agent_id],
                                                                last_acts[agent_id],
                                                                rnn_states[agent_id],
                                                                t_env=self.total_env_steps,
                                                                explore=explore,
                                                                use_target=False,
                                                                use_gumbel=False)
                # update rnn hidden state
                rnn_states[agent_id] = rnn_state.detach().numpy()

                if not isinstance(act, np.ndarray):
                    act = act.detach().numpy()
                last_acts[agent_id] = act

            env_acts = []
            for i in range(self.num_envs):
                env_act = []
                for agent_id in range(self.num_agents):
                    env_act.append(last_acts[agent_id][i])
                env_acts.append(env_act)

            # env step and store the relevant episode information
            next_obs, rewards, dones, infos = env.step(env_acts)

            t += 1
            
            if training_episode:
                self.total_env_steps += self.num_envs
            next_share_obs = []
            for no in next_obs:
                next_share_obs.append(list(chain(*no)))
            next_share_obs = np.array(next_share_obs)

            next_agent_obs = []
            for agent_id in range(self.num_agents):
                next_env_obs = []
                for no in next_obs:
                    next_env_obs.append(no[agent_id])
                next_env_obs = np.array(next_env_obs)
                next_agent_obs.append(next_env_obs)

            dones_env = np.all(dones, axis=1)
            terminate_episodes = np.any(dones_env) or t == self.episode_length
            if terminate_episodes:
                dones_env = np.ones_like(dones_env).astype(bool)

            for agent_id, p_id in zip(self.agent_ids, self.policy_ids):
                episode_obs[p_id][episode_step] = np.expand_dims(agent_obs[agent_id], axis=1)
                episode_share_obs[p_id][episode_step] = share_obs
                episode_acts[p_id][episode_step] = np.expand_dims(last_acts[agent_id], axis=1)
                episode_rewards[p_id][episode_step] = np.expand_dims(rewards[:, agent_id], axis=1)
                episode_next_obs[p_id][episode_step] = np.expand_dims(next_agent_obs[agent_id], axis=1)
                episode_next_share_obs[p_id][episode_step] = next_share_obs
                episode_dones[p_id][episode_step] = np.expand_dims(dones[:, agent_id], axis=1)
                episode_dones_env[p_id][episode_step] = dones_env

                accumulated_rewards[p_id].append(rewards[:, agent_id])

            episode_step += 1
            agent_obs = next_agent_obs
            share_obs = next_share_obs

            assert self.num_envs == 1, ("only one env is support here.")
            if terminate_episodes:
                break

        if explore:
            self.num_episodes_collected += self.num_envs
            # push all episodes collected in this rollout step to the buffer
            self.buffer.insert(self.num_envs,
                               episode_obs,
                               episode_share_obs,
                               episode_acts,
                               episode_rewards,
                               episode_next_obs,
                               episode_next_share_obs,
                               episode_dones,
                               episode_dones_env,
                               episode_avail_acts,
                               episode_next_avail_acts)

        average_episode_rewards = []
        for p_id in self.policy_ids:
            average_episode_rewards.append(np.mean(np.sum(accumulated_rewards[p_id], axis=0)))
        
        env_info['average_episode_rewards'] = np.mean(average_episode_rewards)

        return env_info

    def log(self):
        end = time.time()
        print("\n Env {} Algo {} Exp {} runs total num timesteps {}/{}, FPS {}.\n"
              .format(self.args.scenario_name,
                      self.algorithm_name,
                      self.args.experiment_name,
                      self.total_env_steps,
                      self.num_env_steps,
                      int(self.total_env_steps / (end - self.start))))
        for p_id, train_info in zip(self.policy_ids, self.train_infos):
            self.log_train(p_id, train_info)

        self.log_env(self.env_infos)
        self.log_clear()

    def log_clear(self):
        self.env_infos = {}

        self.env_infos['average_episode_rewards'] = []
