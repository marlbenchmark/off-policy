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

class HanabiRunner(MlpRunner):

    def __init__(self, config):
        super(HanabiRunner, self).__init__(config)
        # fill replay buffer with random actions
        self.finish_first_train_reset = False
        self.turn_count = 0
        num_warmup_episodes = max((self.batch_size/self.episode_length, self.args.num_random_episodes))
        self.warmup(num_warmup_episodes)
        self.start = time.time()
        self.log_clear()
        

    def turn_init(self, p_id, n_rollout_threads, obs, share_obs, avail_acts):
        self.obs = obs
        self.share_obs = share_obs
        self.avail_acts = avail_acts

        if is_multidiscrete(self.policy_info[p_id]['act_space']):
            self.sum_act_dim = int(np.sum(self.policy_act_dim[p_id]))
        else:
            self.sum_act_dim = self.policy_act_dim[p_id]

        self.turn_rewards_since_last_action = np.zeros((n_rollout_threads, len(self.policy_agents[p_id]), 1), dtype=np.float32)
        self.turn_rewards_last = np.zeros_like(self.turn_rewards_since_last_action)
        self.turn_rewards = np.zeros_like(self.turn_rewards_since_last_action)

        self.turn_acts_last = self.turn_acts = np.zeros((n_rollout_threads, len(self.policy_agents[p_id]), self.sum_act_dim), dtype=np.float32)
        
        self.turn_obs = np.zeros((n_rollout_threads, len(self.policy_agents[p_id]), *obs.shape[1:]), dtype=np.float32)
        self.turn_next_obs_last = np.zeros_like(self.turn_obs)
        self.turn_obs_last = np.zeros_like(self.turn_obs)

        self.turn_share_obs = np.zeros((n_rollout_threads, len(self.policy_agents[p_id]), *share_obs.shape[1:]), dtype=np.float32)
        self.turn_next_share_obs_last = np.zeros_like(self.turn_share_obs)
        self.turn_share_obs_last = np.zeros_like(self.turn_share_obs)
               
        self.turn_avail_acts = np.zeros((n_rollout_threads, len(self.policy_agents[p_id]), *avail_acts.shape[1:]), dtype=np.float32)
        self.turn_next_avail_acts_last = np.ones_like(self.turn_avail_acts)
        self.turn_avail_acts_last = np.ones_like(self.turn_avail_acts)

        self.turn_dones_last = self.turn_dones = np.zeros_like(self.turn_rewards_since_last_action)
        self.turn_dones_env_last = np.zeros((n_rollout_threads, 1), dtype=np.float32)
        self.turn_dones_env = np.zeros((n_rollout_threads, 1), dtype=np.float32)
        
        self.check_avail_acts = np.ones_like(avail_acts)

    @torch.no_grad()
    def eval(self):
        self.trainer.prep_rollout()

        eval_infos = {}
        eval_infos['average_score'] = []
        eval_infos['average_step_rewards'] = []

        for _ in range(self.args.num_eval_episodes):
            env_info = self.eval_collect_rollout()
            
            for k, v in env_info.items():
                eval_infos[k].append(v)

        self.log_env(eval_infos, suffix="eval_")

    def collect_rollout(self, explore=True, training_episode=True, warmup=False):
        env_info = {}
        p_id = 'policy_0'
        policy = self.policies[p_id]
        env = self.env
        n_rollout_threads = self.num_envs

        score = []

        if not self.finish_first_train_reset:
            self.reset_choose = np.ones(n_rollout_threads) == 1.0
            obs, share_obs, avail_acts = env.reset(self.reset_choose)
            self.finish_first_train_reset = True
            self.turn_init(p_id, n_rollout_threads, obs, share_obs, avail_acts)

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
            #self.reset_choose = np.zeros(n_rollout_threads) == 1.0
            # get actions for all agents to step the env
            for current_agent_id in range(len(self.policy_agents[p_id])):
                env_acts = np.ones((n_rollout_threads, self.sum_act_dim), dtype=np.float32) * (-1.0)
                choose = np.any(self.check_avail_acts == 1, axis=1)
                if ~np.any(choose):
                    self.reset_choose = np.ones(n_rollout_threads) == 1.0
                    break
                if warmup:
                    # completely random actions in pre-training warmup phase
                    act = policy.get_random_actions(self.obs[choose], 
                                                    self.avail_acts[choose])
                else:
                    act, _ = policy.get_actions(self.obs[choose],
                                                self.avail_acts[choose],
                                                t_env=self.total_env_steps,
                                                explore=explore)
                if not isinstance(act, np.ndarray):
                    act = act.cpu().detach().numpy()

                env_acts[choose] = act.copy()

                # unpack actions to format needed to step env (list of dicts, dict mapping agent_id to action)
                #if step == 0:
                #    pass
                #else:
                self.turn_acts_last[:, current_agent_id] = self.turn_acts[:, current_agent_id]
                self.turn_obs_last[:, current_agent_id] = self.turn_obs[:, current_agent_id]
                self.turn_share_obs_last[:,current_agent_id] = self.turn_share_obs[:, current_agent_id]
                self.turn_avail_acts_last[:,current_agent_id] = self.turn_avail_acts[:, current_agent_id]
                self.turn_rewards_last[:,current_agent_id] = self.turn_rewards[:, current_agent_id]
                self.turn_dones_last[:, current_agent_id] = self.turn_dones[:, current_agent_id]
                self.turn_dones_env_last = self.turn_dones_env.copy()
                self.turn_next_obs_last[:,current_agent_id] = self.obs[:, current_agent_id]
                self.turn_next_share_obs_last[:,current_agent_id] = self.share_obs[:, current_agent_id]
                self.turn_next_avail_acts_last[:,current_agent_id] = self.avail_acts[:, current_agent_id]

                self.turn_acts[choose, current_agent_id] = act
                self.turn_obs[choose, current_agent_id] = self.obs[choose]
                self.turn_share_obs[choose, current_agent_id] = self.share_obs[choose]
                self.turn_avail_acts[choose, current_agent_id] = self.avail_acts[choose]

                # env step and store the relevant episode information
                next_obs, next_share_obs, rewards, dones_original, infos, next_avail_acts = env.step(env_acts)
                
                self.obs = next_obs
                self.share_obs = next_share_obs
                self.avail_acts = next_avail_acts

                dones_env_original = np.all(dones_original, axis=1)
                dones = dones_original[:,:,0]
                dones_env = np.all(dones, axis=1)
                
                episode_rewards.append(rewards)

                for i in range(n_rollout_threads):
                    if dones_env[i]:
                        if 'score' in infos[i].keys():
                            score.append(infos[i]['score'])

                self.turn_rewards_since_last_action[choose] += rewards[choose]
                self.turn_rewards[choose, current_agent_id] = self.turn_rewards_since_last_action[choose, current_agent_id].copy()
                self.turn_rewards_since_last_action[choose, current_agent_id] = 0.0

                # done=True env
                self.reset_choose[dones_env == True] = True

                # deal with all agents
                self.turn_dones_env[dones_env ==True] = dones_env_original[dones_env == True].copy()
                self.avail_acts[dones_env == True] = np.ones(((dones_env == True).sum(), self.sum_act_dim)).astype(np.float32)
                self.check_avail_acts[dones_env == True] = np.zeros(((dones_env == True).sum(), self.sum_act_dim)).astype(np.float32)

                # deal with current agent
                self.turn_dones[dones_env == True, current_agent_id] = np.zeros(((dones_env == True).sum(), 1)).astype(np.float32)

                # deal with left agents
                left_agent_id = current_agent_id + 1
                left_agents_num = self.num_agents - left_agent_id

                self.turn_dones[dones_env == True, left_agent_id:] = np.ones(((dones_env == True).sum(), left_agents_num, 1)).astype(np.float32)

                # must use the right value
                self.turn_rewards[dones_env == True, left_agent_id:] = self.turn_rewards_since_last_action[dones_env == True, left_agent_id:].copy()
                self.turn_rewards_since_last_action[dones_env ==True, left_agent_id:] = 0.0

                # use any value is okay
                self.turn_acts[dones_env == True, left_agent_id:] = np.zeros(((dones_env == True).sum(), left_agents_num, policy.act_dim))
                self.turn_obs[dones_env == True, left_agent_id:] = np.zeros(((dones_env == True).sum(), left_agents_num, policy.obs_dim))
                self.turn_share_obs[dones_env == True, left_agent_id:] = np.zeros(((dones_env == True).sum(), left_agents_num, policy.central_obs_dim))
                self.turn_avail_acts[dones_env == True, left_agent_id:] = np.ones(((dones_env == True).sum(), left_agents_num, policy.act_dim))

                # deal with previous agents
                self.turn_rewards[dones_env == True, 0:current_agent_id] = self.turn_rewards_since_last_action[dones_env == True, 0:current_agent_id].copy()
                self.turn_rewards_since_last_action[dones_env ==True, 0:current_agent_id] = 0.0

                # done==False env
                # deal with current agent
                self.turn_dones[dones_env == False, current_agent_id] = dones_original[dones_env == False, current_agent_id].copy()
                self.turn_dones_env[dones_env ==False] = dones_env_original[dones_env == False].copy()

                # done==None
                # pass

            new_next_obs, new_next_share_obs, new_next_avail_acts = env.reset(self.reset_choose)

            self.obs[self.reset_choose] = new_next_obs[self.reset_choose].copy()
            self.share_obs[self.reset_choose] = new_next_share_obs[self.reset_choose].copy()
            self.avail_acts[self.reset_choose] = new_next_avail_acts[self.reset_choose].copy()
            self.check_avail_acts[self.reset_choose] = new_next_avail_acts[self.reset_choose].copy()

            if self.turn_count > 0:
                step_obs[p_id] = self.turn_obs_last
                step_share_obs[p_id] = self.turn_share_obs_last
                step_acts[p_id] = self.turn_acts_last
                step_rewards[p_id] = self.turn_rewards_last
                step_next_obs[p_id] = self.turn_next_obs_last
                step_next_share_obs[p_id] = self.turn_next_share_obs_last
                step_dones[p_id] = self.turn_dones_last
                step_dones_env[p_id] = self.turn_dones_env_last
                step_avail_acts[p_id] = self.turn_avail_acts_last
                step_next_avail_acts[p_id] = self.turn_next_avail_acts_last

                if explore:
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
            
            self.turn_count += 1
            # train
            if training_episode:
                self.total_env_steps += n_rollout_threads
                if (self.last_train_T == 0 or ((self.total_env_steps - self.last_train_T) / self.train_interval) >= 1):
                    self.train()
                    self.total_train_steps += 1
                    self.last_train_T = self.total_env_steps

        env_info['average_step_rewards'] = np.mean(episode_rewards)
        env_info['average_score'] = np.mean(score)

        return env_info
    
    def eval_collect_rollout(self):
        env_info = {}
        p_id = 'policy_0'
        policy = self.policies[p_id]

        env = self.eval_env
        n_rollout_threads = self.num_eval_envs
        assert n_rollout_threads == 1, ("only support one env for evaluation in hanabi")

        reset_choose = np.ones(n_rollout_threads) == 1.0
        obs, share_obs, avail_acts = env.reset(reset_choose)

        if is_multidiscrete(self.policy_info[p_id]['act_space']):
            self.sum_act_dim = int(np.sum(self.policy_act_dim[p_id]))
        else:
            self.sum_act_dim = self.policy_act_dim[p_id]

        self.check_avail_acts = np.ones_like(avail_acts)

        # init
        episode_rewards = []

        while True:
            # get actions for all agents to step the env
            for current_agent_id in range(len(self.policy_agents[p_id])):
                env_acts = np.ones((n_rollout_threads, self.sum_act_dim), dtype=np.float32) * (-1.0)
                act, _ = policy.get_actions(obs[0],
                                            avail_acts[0],
                                            t_env=self.total_env_steps)
                if not isinstance(act, np.ndarray):
                    act = act.cpu().detach().numpy()

                env_acts[0] = act.copy()

                # env step and store the relevant episode information
                next_obs, next_share_obs, rewards, dones_original, infos, next_avail_acts = env.step(env_acts)
                
                obs = next_obs
                share_obs = next_share_obs
                avail_acts = next_avail_acts

                dones = dones_original[:,:,0]
                dones_env = np.all(dones, axis=1)
                
                episode_rewards.append(rewards)

                if dones_env[0]:
                    if 'score' in infos[0].keys():
                        env_info['average_score'] = infos[0]['score'] 
                    env_info['average_step_rewards'] = np.mean(episode_rewards)             
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

        self.env_infos['average_step_rewards'] = []
        self.env_infos['average_score'] = []