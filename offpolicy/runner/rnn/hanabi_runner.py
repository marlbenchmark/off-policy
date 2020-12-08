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

class HanabiRunner(RecRunner):
    def __init__(self, config):
        super(HanabiRunner, self).__init__(config)
        # fill replay buffer with random actions
        num_warmup_episodes = max((self.batch_size, self.args.num_random_episodes))
        self.warmup(num_warmup_episodes)
        self.start = time.time()
        self.log_clear()

    def eval(self):
        self.trainer.prep_rollout()

        eval_infos = {}
        eval_infos['average_score'] = []
        eval_infos['average_episode_rewards'] = []

        for _ in range(self.args.num_eval_episodes):
            env_info = self.collecter( explore=False, training_episode=False, warmup=False)
            
            for k, v in env_info.items():
                eval_infos[k].append(v)

        self.log_env(eval_infos, suffix="eval_")

    @torch.no_grad()
    def collect_rollout(self, explore=True, training_episode=True, warmup=False):
        env_info = {}
        p_id = 'policy_0'
        policy = self.policies[p_id]

        env = self.env if training_episode or warmup else self.eval_env
        obs, share_obs, avail_acts = env.reset()

        t = 0
        score = 0
        episode_step = 0
        terminate_episodes = False

        rnn_states = torch.zeros((self.num_envs, len(self.policy_agents[p_id]), self.hidden_size))
        if is_multidiscrete(self.policy_info[p_id]['act_space']):
            self.sum_act_dim = int(np.sum(self.policy_act_dim[p_id]))
        else:
            self.sum_act_dim = self.policy_act_dim[p_id]

        last_acts = np.zeros((self.num_envs, len(self.policy_agents[p_id]), self.sum_act_dim))

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

        episode_obs[p_id] = np.zeros((self.episode_length, self.num_envs, len(self.policy_agents[p_id]), *obs.shape[1:]), dtype=np.float32)
        episode_share_obs[p_id] = np.zeros((self.episode_length, self.num_envs, len(self.policy_agents[p_id]), *share_obs.shape[1:]), dtype=np.float32)
        episode_avail_acts[p_id] = np.ones((self.episode_length, self.num_envs, len(self.policy_agents[p_id]), *avail_acts.shape[1:]), dtype=np.float32)
        episode_acts[p_id] = np.zeros((self.episode_length, *last_acts.shape), dtype=np.float32)
        episode_rewards[p_id] = np.zeros((self.episode_length, self.num_envs, len(self.policy_agents[p_id]), 1), dtype=np.float32)
        episode_dones[p_id] = np.ones((self.episode_length, self.num_envs, len(self.policy_agents[p_id]), 1), dtype=np.float32)
        episode_dones_env[p_id] = np.ones((self.episode_length, self.num_envs, 1), dtype=np.float32)
        accumulated_rewards[p_id] = []

        # obs, share_obs, avail_acts, action, rewards, dones, dones_env
        turn_rewards_since_last_action = np.zeros((self.num_envs, len(self.policy_agents[p_id]), 1), dtype=np.float32)
        env_acts = np.zeros_like(last_acts)
        turn_obs = np.zeros((self.num_envs, len(self.policy_agents[p_id]), *obs.shape[1:]), dtype=np.float32)
        turn_share_obs = np.zeros((self.num_envs, len(self.policy_agents[p_id]), *share_obs.shape[1:]), dtype=np.float32)
        turn_avail_acts = np.zeros((self.num_envs, len(self.policy_agents[p_id]), *avail_acts.shape[1:]), dtype=np.float32)
        turn_acts = np.zeros_like(last_acts)
        turn_rewards = np.zeros_like(turn_rewards_since_last_action)
        turn_dones = np.zeros_like(turn_rewards_since_last_action)
        turn_dones_env = np.zeros((self.num_envs, 1), dtype=np.float32)

        while t < self.episode_length:
            # get actions for all agents to step the env
            for agent_id in range(len(self.policy_agents[p_id])):
                if warmup:
                    # completely random actions in pre-training warmup phase
                    act = policy.get_random_actions(obs, avail_acts)
                    # get new rnn hidden state
                    _, rnn_state, _ = policy.get_actions(obs,
                                                        last_acts[:,agent_id],
                                                        rnn_states[:,agent_id],
                                                        avail_acts)
                else:
                    # get actions with exploration noise (eps-greedy/Gaussian)
                    act, rnn_state, _ = policy.get_actions(obs,
                                                            last_acts[:,agent_id],
                                                            rnn_states[:,agent_id],
                                                            avail_acts,
                                                            t_env=self.total_env_steps,
                                                            explore=explore)
                rnn_states[:, agent_id] = rnn_state.detach()
                
                if not isinstance(act, np.ndarray):
                    act = act.cpu().detach().numpy()
                last_acts[:, agent_id] = act

                # [obs, share_obs, avail_acts, actions] - > current agent id
                turn_acts[:, agent_id] = act
                turn_obs[:, agent_id] = obs.copy()
                turn_share_obs[:, agent_id] = share_obs.copy()
                turn_avail_acts[:, agent_id] = avail_acts.copy()

                # env step and store the relevant episode information
                next_obs, next_share_obs, rewards, dones, infos, next_avail_acts = env.step(act)

                t += 1
                if training_episode:
                    self.total_env_steps += self.num_envs
                dones_env = np.all(dones, axis=1)

                # [rewards] - > current agent id
                turn_rewards_since_last_action += rewards
                turn_rewards[:, agent_id] = turn_rewards_since_last_action[:, agent_id].copy()
                turn_rewards_since_last_action[:, agent_id] = 0.0

                for i in range(self.num_envs):
                    # done==True env
                    if (dones_env[i] or (t == self.episode_length)):
                        # episode is done
                        # [dones_env] deal with all agents - > all agents
                        turn_dones_env = dones_env.copy()

                        # [dones] deal with current agent - > current agent id
                        current_agent_id = agent_id
                        turn_dones[i, current_agent_id] = False

                        # deal with left_agent of this turn - > left agents
                        # [obs, share obs, avail_acts, actions, rewards, dones]
                        left_agent_id = current_agent_id + 1
                        
                        # [rewards]   must use the right value
                        turn_rewards[i, left_agent_id:] = turn_rewards_since_last_action[i, left_agent_id:].copy()
                        turn_rewards_since_last_action[i, left_agent_id:] = 0.0
                        
                        # [dones]
                        turn_dones[i, left_agent_id:] = True

                        # [obs, share_obs, avail_acts, actions]
                        turn_acts[i, left_agent_id] = 0.0
                        turn_obs[i, left_agent_id] = 0.0
                        turn_share_obs[i, left_agent_id] = 0.0
                        turn_avail_acts[i, left_agent_id] = 1.0

                        # deal with previous agents of this turn -> previous agents
                        turn_rewards[i, 0:current_agent_id] += turn_rewards_since_last_action[i, 0:current_agent_id].copy()
                        turn_rewards_since_last_action[i, 0:current_agent_id] = 0.0

                        if 'score' in infos[i].keys():
                            env_info['average_score'] = infos[i]['score']

                        terminate_episodes = True
                        break
                    # done=False env
                    else:
                        # [dones, dones_env]
                        turn_dones[i, agent_id] = dones[i, agent_id].copy()  # current step
                        turn_dones_env = dones_env.copy()

                if terminate_episodes:
                    break

                obs = next_obs
                share_obs = next_share_obs
                avail_acts = next_avail_acts

            episode_obs[p_id][episode_step] = turn_obs.copy()
            episode_share_obs[p_id][episode_step] = turn_share_obs.copy()
            episode_acts[p_id][episode_step] = turn_acts.copy()
            episode_rewards[p_id][episode_step] = turn_rewards.copy()
            accumulated_rewards[p_id].append(turn_rewards.copy())
            episode_dones[p_id][episode_step] = turn_dones.copy()
            episode_dones_env[p_id][episode_step] = turn_dones_env.copy()
            episode_avail_acts[p_id][episode_step] = turn_avail_acts.copy()
            episode_step += 1

            if terminate_episodes:
                break

        # the current step of episode data is ready, calculate the next step.

        episode_next_obs[p_id] = np.zeros_like(episode_obs[p_id])
        episode_next_share_obs[p_id] = np.zeros_like(episode_share_obs[p_id])
        episode_next_avail_acts[p_id] = np.ones_like(episode_avail_acts[p_id])

        for current_step in range(episode_step - 1):
            episode_next_obs[p_id][current_step] = episode_obs[p_id][current_step + 1].copy()
            episode_next_share_obs[p_id][current_step] = episode_share_obs[p_id][current_step + 1].copy()
            episode_next_avail_acts[p_id][current_step] = episode_avail_acts[p_id][current_step + 1].copy()

        if explore:
            # push all episodes collected in this rollout step to the buffer
            self.num_episodes_collected += self.num_envs
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

    def log(self):
        end = time.time()
        print("\n Env {} Algo {} Exp {} runs total num timesteps {}/{}, FPS{}.\n"
              .format(self.args.hanabi_name,
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
        self.env_infos['average_score'] = []