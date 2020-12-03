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

class SMACRunner(RecRunner):
    def __init__(self, config):
        super(SMACRunner, self).__init__(config)
        # fill replay buffer with random actions
        num_warmup_episodes = max((self.batch_size, self.args.num_random_episodes))
        self.warmup(num_warmup_episodes)
        self.start = time.time()
        self.log_clear()
    
    @torch.no_grad()
    def eval(self):
        self.trainer.prep_rollout()

        eval_infos = {}
        eval_infos['win_rate'] = []
        eval_infos['average_episode_rewards'] = []

        for _ in range(self.args.num_eval_episodes):
            env_info = self.collecter(explore=False, training_episode=False, warmup=False)
            
            for k, v in env_info.items():
                eval_infos[k].append(v)

        self.log_env(eval_infos, suffix="eval_")
    
    def collect_rollout(self, explore=True, training_episode=True, warmup=False):
        env_info = {}
        p_id = "policy_0"
        policy = self.policies[p_id]

        env = self.env if training_episode or warmup else self.eval_env

        obs, share_obs, avail_acts = env.reset()

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
        episode_acts[p_id] = np.zeros((self.episode_length, *avail_acts.shape), dtype=np.float32)
        episode_rewards[p_id] = np.zeros((self.episode_length, self.num_envs, len(self.policy_agents[p_id]), 1), dtype=np.float32)
        episode_next_obs[p_id] = np.zeros_like(episode_obs[p_id])
        episode_next_share_obs[p_id] = np.zeros_like(episode_share_obs[p_id])
        episode_dones[p_id] = np.ones((self.episode_length, self.num_envs, len(self.policy_agents[p_id]), 1), dtype=np.float32)
        episode_dones_env[p_id] = np.ones((self.episode_length, self.num_envs, 1), dtype=np.float32)
        episode_avail_acts[p_id] = np.ones((self.episode_length, *avail_acts.shape), dtype=np.float32)
        episode_next_avail_acts[p_id] = np.ones_like(episode_avail_acts[p_id])
        accumulated_rewards[p_id] = []

        t = 0
        episode_step = 0
        while t < self.episode_length:
            obs_batch = np.concatenate(obs)
            avail_acts_batch = np.concatenate(avail_acts)
            # get actions for all agents to step the env
            if warmup:
                # completely random actions in pre-training warmup phase
                acts_batch = policy.get_random_actions(obs_batch, avail_acts_batch)
                # get new rnn hidden state
                _, rnn_states_batch, _ = policy.get_actions(obs_batch,
                                                            last_acts_batch,
                                                            rnn_states_batch,
                                                            avail_acts_batch)

            else:
                # get actions with exploration noise (eps-greedy/Gaussian)
                if self.algorithm_name == "rmasac":
                    acts_batch, rnn_states_batch, _ = policy.get_actions(obs_batch,
                                                                        last_acts_batch,
                                                                        rnn_states_batch,
                                                                        avail_acts_batch,
                                                                        sample=explore)
                else:
                    acts_batch, rnn_states_batch, _ = policy.get_actions(obs_batch,
                                                                        last_acts_batch,
                                                                        rnn_states_batch,
                                                                        avail_acts_batch,
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
            next_obs, next_share_obs, rewards, dones, infos, next_avail_acts = env.step(env_acts)
            t += 1
            if training_episode:
                self.total_env_steps += self.num_envs

            dones_env = np.all(dones, axis=1)
            terminate_episodes = np.any(dones_env) or t == self.episode_length
            # if terminate_episodes:
            #    dones_env = np.ones_like(dones_env).astype(bool)

            episode_obs[p_id][episode_step] = obs
            episode_share_obs[p_id][episode_step] = share_obs
            episode_acts[p_id][episode_step] = env_acts
            episode_rewards[p_id][episode_step] = rewards
            accumulated_rewards[p_id].append(rewards.copy())
            episode_next_obs[p_id][episode_step] = next_obs
            episode_next_share_obs[p_id][episode_step] = next_share_obs
            # here dones store agent done flag of the next step
            episode_dones[p_id][episode_step] = dones
            episode_dones_env[p_id][episode_step] = dones_env
            episode_avail_acts[p_id][episode_step] = avail_acts
            episode_next_avail_acts[p_id][episode_step] = next_avail_acts
            episode_step += 1

            obs = next_obs
            share_obs = next_share_obs
            avail_acts = next_avail_acts

            assert self.num_envs == 1, ("only one env is support here.")
            if terminate_episodes:
                for i in range(self.num_envs):
                    if 'won' in infos[i][0].keys():
                        # take one agent
                        env_info['win_rate'] = 1 if infos[i][0]['won'] else 0
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

        env_info['average_episode_rewards'] = np.mean(np.sum(accumulated_rewards[p_id], axis=0))

        return env_info

    def log(self):
        end = time.time()
        print("\n Env {} Map {} Algo {} Exp {} runs total num timesteps {}/{}, FPS {}. \n"
              .format(self.env_name,
                      self.args.map_name,
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
        self.env_infos['win_rate'] = []
