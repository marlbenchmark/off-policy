import os
import wandb
import numpy as np
from itertools import chain
from tensorboardX import SummaryWriter
import torch
import time

from offpolicy.utils.mlp_buffer import MlpReplayBuffer, PrioritizedMlpReplayBuffer
from offpolicy.utils.util import is_discrete, is_multidiscrete, DecayThenFlatSchedule

from offpolicy.runner.mlp.base_runner import MlpRunner

class HNSRunner(MlpRunner):
    def __init__(self, config):
        super(HNSRunner, self).__init__(config)
        # fill replay buffer with random actions
        self.finish_first_train_reset = False
        num_warmup_episodes = max((self.batch_size/self.episode_length, self.args.num_random_episodes))
        self.warmup(num_warmup_episodes)
        self.start = time.time()
        self.log_clear()

    @torch.no_grad()
    def eval(self):
        self.trainer.prep_rollout()

        eval_infos = {}
        eval_infos['success_rate'] = []
        eval_infos['average_step_rewards'] = []

        for _ in range(self.args.num_eval_episodes):
            env_info = self.collecter( explore=False, training_episode=False, warmup=False)
            
            for k, v in env_info.items():
                eval_infos[k].append(v)

        self.log_env(eval_infos, suffix="eval_")

    def collect_rollout(self, explore=True, training_episode=True, warmup=False):
        env_info = {}
        p_id = "policy_0"
        policy = self.policies[p_id]

        env = self.env if explore else self.eval_env
        n_rollout_threads = self.num_envs if explore else self.num_eval_envs

        need_to_reset = True

        while need_to_reset:
            success = 0.0
            trials = 0.0
            discard_episode = 0

            if not explore:
                obs, share_obs, _ = env.reset()
            else:
                need_to_reset = False
                if self.finish_first_train_reset:
                    obs = self.obs
                    share_obs = self.share_obs
                else:
                    obs, share_obs, _ = env.reset()
                    self.finish_first_train_reset = True

            # init
            episode_rewards = []
            step_obs = {}
            step_share_obs = {}
            step_acts = {}
            step_rewards = {}
            step_next_obs = {}
            step_next_share_obs = {}
            step_dones = {}
            step_dones_env = {}
            step_avail_acts = {}
            step_next_avail_acts = {}

            for step in range(self.episode_length):
                obs_batch = np.concatenate(obs)
                # get actions for all agents to step the env
                if warmup:
                    # completely random actions in pre-training warmup phase
                    acts_batch = policy.get_random_actions(obs_batch)
                else:
                    # get actions with exploration noise (eps-greedy/Gaussian)
                    acts_batch, _ = policy.get_actions(obs_batch,
                                                        t_env=self.total_env_steps,
                                                        explore=explore)
                # update rnn hidden state
                if not isinstance(acts_batch, np.ndarray):
                    acts_batch = acts_batch.cpu().detach().numpy()
                env_acts = np.split(acts_batch, n_rollout_threads)

                # env step and store the relevant episode information
                next_obs, next_share_obs, rewards, dones, infos, _ = env.step(env_acts)
                episode_rewards.append(rewards)
                dones_env = np.all(dones, axis=1)

                for i in range(n_rollout_threads):
                    if dones_env[i]:
                        if 'discard_episode' in infos[i].keys():
                            if infos[i]['discard_episode']:
                                discard_episode += 1
                            else:
                                trials += 1
                        else:
                            trials += 1
                        if 'success' in infos[i].keys():
                            if infos[i]['success']:
                                success += 1

                if explore and n_rollout_threads == 1 and np.all(dones_env):
                    next_obs, next_share_obs, _ = env.reset()

                if not explore and np.any(dones_env):

                    assert n_rollout_threads == 1, ("only support one env for the evaluation in hide and seek domain. ")

                    if trials < 1:
                        need_to_reset = True
                        break
                    else:
                        need_to_reset = False
                        env_info['success_rate'] = success
                        env_info['average_step_rewards'] = np.mean(episode_rewards)
                        return env_info

                step_obs[p_id] = obs
                step_share_obs[p_id] = share_obs
                step_acts[p_id] = env_acts
                step_rewards[p_id] = rewards
                step_next_obs[p_id] = next_obs
                step_next_share_obs[p_id] = next_share_obs
                step_dones[p_id] = np.zeros_like(dones)
                step_dones_env[p_id] = dones_env
                step_avail_acts[p_id] = None
                step_next_avail_acts[p_id] = None

                obs = next_obs
                share_obs = next_share_obs

                if explore:
                    self.obs = obs
                    self.share_obs = share_obs
                    # push all episodes collected in this rollout step to the buffer
                    self.buffer.insert(n_rollout_threads,
                                       step_obs,
                                       step_share_obs,
                                       step_acts,
                                       step_rewards,
                                       step_next_obs,
                                       step_next_share_obs,
                                       step_dones,
                                       step_dones_env,
                                       step_avail_acts,
                                       step_next_avail_acts)
                # train
                if training_episode:
                    self.total_env_steps += n_rollout_threads
                    if (self.last_train_T == 0 or ((self.total_env_steps - self.last_train_T) / self.train_interval) >= 1):
                        self.train()
                        self.total_train_steps += 1
                        self.last_train_T = self.total_env_steps

        env_info['success_rate'] = success/trials if trials > 0 else 0.0
        env_info['average_step_rewards'] = np.mean(episode_rewards)

        return env_info

    def log(self):
        end = time.time()
        print("\n Env {} Algo {} Exp {} runs total num timesteps {}/{}, FPS{}.\n"
              .format(self.env_name,
                      self.algorithm_name,
                      self.args.experiment_name,
                      self.total_env_steps,
                      self.num_env_steps,
                      int(self.total_env_steps / (end - self.start))))
        for p_id, train_info in zip(self.policy_ids, self.train_infos):
            self.log_train(p_id, train_info)

        self.log_env(self.env_infos)
        self.log_clear()

    def log_env(self, env_info, suffix=None):
        if self.env_name == "BoxLocking" or self.env_name == "BlueprintConstruction":
            for k, v in env_info:
                if len(v) > 0:
                    v = np.mean(v)
                    suffix_k = k if suffix is None else suffix + k 
                    print(suffix_k + " is " + str(v))
                    if self.use_wandb:
                        wandb.log({suffix_k: v}, step=self.total_env_steps)
                    else:
                        self.writter.add_scalars(suffix_k, {suffix_k: v}, self.total_env_steps)

    def log_clear(self):
        self.env_infos = {}

        self.env_infos['average_step_rewards'] = []
        self.env_infos['success_rate'] = []