import os
import numpy as np
from itertools import chain
import wandb
import torch
from tensorboardX import SummaryWriter
import time

from offpolicy.utils.rec_buffer import RecReplayBuffer, PrioritizedRecReplayBuffer
from offpolicy.utils.util import is_discrete, is_multidiscrete, DecayThenFlatSchedule


class RecRunner(object):

    def __init__(self, config):
        # non-tunable hyperparameters are in args
        self.args = config["args"]
        self.device = config["device"]
        self.q_learning = ["qmix","vdn"]

        # set tunable hyperparameters
        self.share_policy = self.args.share_policy
        self.algorithm_name = self.args.algorithm_name
        self.env_name = self.args.env_name
        self.num_env_steps = self.args.num_env_steps
        self.use_wandb = self.args.use_wandb
        self.use_reward_normalization = self.args.use_reward_normalization
        self.use_popart = self.args.use_popart
        self.use_per = self.args.use_per
        self.per_alpha = self.args.per_alpha
        self.per_beta_start = self.args.per_beta_start
        self.buffer_size = self.args.buffer_size
        self.batch_size = self.args.batch_size
        self.hidden_size = self.args.hidden_size
        self.use_soft_update = self.args.use_soft_update
        self.hard_update_interval_episode = self.args.hard_update_interval_episode
        self.popart_update_interval_step = self.args.popart_update_interval_step
        self.actor_train_interval_step = self.args.actor_train_interval_step
        self.train_interval_episode = self.args.train_interval_episode
        self.train_interval = self.args.train_interval
        self.use_eval = self.args.use_eval
        self.eval_interval = self.args.eval_interval
        self.save_interval = self.args.save_interval
        self.log_interval = self.args.log_interval

        self.total_env_steps = 0  # total environment interactions collected during training
        self.num_episodes_collected = 0  # total episodes collected during training
        self.total_train_steps = 0  # number of gradient updates performed
        # last episode after which a gradient update was performed
        self.last_train_episode = 0
        self.last_train_T = 0
        self.last_eval_T = 0  # last episode after which a eval run was conducted
        self.last_save_T = 0  # last epsiode after which the models were saved
        self.last_log_T = 0
        self.last_hard_update_episode = 0

        if config.__contains__("take_turn"):
            self.take_turn = config["take_turn"]
        else:
            self.take_turn = False

        if config.__contains__("use_same_share_obs"):
            self.use_same_share_obs = config["use_same_share_obs"]
        else:
            self.use_same_share_obs = False

        if config.__contains__("use_available_actions"):
            self.use_avail_acts = config["use_available_actions"]
        else:
            self.use_avail_acts = False

        if config.__contains__("buffer_length"):
            self.episode_length = config["buffer_length"]
            if self.args.naive_recurrent_policy:
                self.data_chunk_length = config["buffer_length"]
            else:
                self.data_chunk_length = self.args.data_chunk_length
        else:
            self.episode_length = self.args.episode_length
            if self.args.naive_recurrent_policy:
                self.data_chunk_length = self.args.episode_length
            else:
                self.data_chunk_length = self.args.data_chunk_length

        self.policy_info = config["policy_info"]
        self.policy_ids = sorted(list(self.policy_info.keys()))
        self.policy_mapping_fn = config["policy_mapping_fn"]

        self.num_agents = config["num_agents"]
        self.agent_ids = [i for i in range(self.num_agents)]

        self.env = config["env"]
        self.eval_env = config["eval_env"]
        self.num_envs = 1

        # dir
        self.model_dir = self.args.model_dir
        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        # initialize all the policies and organize the agents corresponding to each policy
        if self.algorithm_name == "rmatd3":
            from offpolicy.algorithms.r_matd3.algorithm.rMATD3Policy import R_MATD3Policy as Policy
            from offpolicy.algorithms.r_matd3.r_matd3 import R_MATD3 as TrainAlgo
        elif self.algorithm_name == "rmaddpg":
            assert self.actor_train_interval_step == 1, (
                "rmaddpg only support actor_train_interval_step=1.")
            from offpolicy.algorithms.r_maddpg.algorithm.rMADDPGPolicy import R_MADDPGPolicy as Policy
            from offpolicy.algorithms.r_maddpg.r_maddpg import R_MADDPG as TrainAlgo
        elif self.algorithm_name == "rmasac":
            assert self.actor_train_interval_step == 1, (
                "rmasac only support actor_train_interval_step=1.")
            from offpolicy.algorithms.r_masac.algorithm.rMASACPolicy import R_MASACPolicy as Policy
            from offpolicy.algorithms.r_masac.r_masac import R_MASAC as TrainAlgo
        elif self.algorithm_name == "qmix":
            from offpolicy.algorithms.qmix.algorithm.QMixPolicy import QMixPolicy as Policy
            from offpolicy.algorithms.qmix.qmix import QMix as TrainAlgo
        elif self.algorithm_name == "vdn":
            from offpolicy.algorithms.vdn.algorithm.VDNPolicy import VDNPolicy as Policy
            from offpolicy.algorithms.vdn.vdn import VDN as TrainAlgo
        else:
            raise NotImplementedError

        self.collecter = self.collect_rollout
        self.saver = self.save_q if self.algorithm_name in self.q_learning else self.save
        self.train = self.batch_train_q if self.algorithm_name in self.q_learning else self.batch_train
        self.restorer = self.restore_q if self.algorithm_name in self.q_learning else self.restore

        self.policies = {p_id: Policy(config, self.policy_info[p_id]) for p_id in self.policy_ids}

        if self.model_dir is not None:
            self.restorer()

        # initialize rmaddpg class for updating policies
        self.trainer = TrainAlgo(self.args, self.num_agents, self.policies, self.policy_mapping_fn,
                                 device=self.device, episode_length=self.episode_length)

        self.policy_agents = {policy_id: sorted(
            [agent_id for agent_id in self.agent_ids if self.policy_mapping_fn(agent_id) == policy_id]) for policy_id in
            self.policies.keys()}

        self.policy_obs_dim = {
            policy_id: self.policies[policy_id].obs_dim for policy_id in self.policy_ids}
        self.policy_act_dim = {
            policy_id: self.policies[policy_id].act_dim for policy_id in self.policy_ids}
        self.policy_central_obs_dim = {
            policy_id: self.policies[policy_id].central_obs_dim for policy_id in self.policy_ids}

        num_train_episodes = (self.num_env_steps / self.episode_length) / (self.train_interval_episode)
        self.beta_anneal = DecayThenFlatSchedule(self.per_beta_start, 1.0, num_train_episodes, decay="linear")

        if self.use_per:
            self.buffer = PrioritizedRecReplayBuffer(self.per_alpha,
                                                     self.policy_info,
                                                     self.policy_agents,
                                                     self.buffer_size,
                                                     self.episode_length,
                                                     self.use_same_share_obs,
                                                     self.use_avail_acts,
                                                     self.use_reward_normalization)
        else:
            self.buffer = RecReplayBuffer(self.policy_info,
                                          self.policy_agents,
                                          self.buffer_size,
                                          self.episode_length,
                                          self.use_same_share_obs,
                                          self.use_avail_acts,
                                          self.use_reward_normalization)

        # fill replay buffer with random actions
        num_warmup_episodes = max((self.batch_size, self.args.num_random_episodes))
        self.warmup(num_warmup_episodes)
        self.start = time.time()
        self.log_clear()

    def run(self):
        # collect data
        self.trainer.prep_rollout()
        train_episode_reward, train_metric = self.collecter(explore=True, training_episode=True, warmup=False)

        self.train_episode_rewards.append(train_episode_reward)
        self.train_metrics.append(train_metric)

        # train
        if ((self.num_episodes_collected - self.last_train_episode) / self.train_interval_episode) >= 1 or self.last_train_episode == 0:
            self.train()
            self.total_train_steps += 1
            self.last_train_episode = self.num_episodes_collected

        # save
        if (self.total_env_steps - self.last_save_T) / self.save_interval >= 1:
            self.saver()
            self.last_save_T = self.total_env_steps

        # log
        if ((self.total_env_steps - self.last_log_T) / self.log_interval) >= 1:
            self.log()
            self.last_log_T = self.total_env_steps

        # eval
        if self.use_eval and ((self.total_env_steps - self.last_eval_T) / self.eval_interval) >= 1:
            self.eval()
            self.last_eval_T = self.total_env_steps

        return self.total_env_steps

    def batch_train(self):
        self.trainer.prep_training()
        # do a gradient update if the number of episodes collected since the last training update exceeds the specified amount
        update_actor = ((self.total_train_steps % self.actor_train_interval_step) == 0)
        # gradient updates
        self.train_infos = []
        for p_id in self.policy_ids:
            if self.use_per:
                beta = self.beta_anneal.eval(self.total_train_steps)
                sample = self.buffer.sample(self.batch_size, beta, p_id)
            else:
                sample = self.buffer.sample(self.batch_size)

            update = self.trainer.shared_train_policy_on_batch if self.use_same_share_obs else self.trainer.cent_train_policy_on_batch
            
            train_info, new_priorities, idxes = update(p_id, sample, update_actor)

            if self.use_per:
                self.buffer.update_priorities(idxes, new_priorities, p_id)

            self.train_infos.append(train_info)

        if self.use_soft_update and update_actor:
            for pid in self.policy_ids:
                self.policies[pid].soft_target_updates()
        else:
            if ((self.num_episodes_collected - self.last_hard_update_episode) / self.hard_update_interval_episode) >= 1:
                for pid in self.policy_ids:
                    self.policies[pid].hard_target_updates()
                self.last_hard_update_episode = self.num_episodes_collected

    def batch_train_q(self):
        self.trainer.prep_training()
        update_popart = ((self.total_train_steps % self.popart_update_interval_step) == 0)
        # gradient updates
        self.train_infos = []

        for p_id in self.policy_ids:
            if self.use_per:
                beta = self.beta_anneal.eval(self.total_train_steps)
                sample = self.buffer.sample(self.batch_size, beta, p_id)
            else:
                sample = self.buffer.sample(self.batch_size)

            train_info, new_priorities, idxes = self.trainer.train_policy_on_batch(
                sample, self.use_same_share_obs, update_popart)

            if self.use_per:
                self.buffer.update_priorities(idxes, new_priorities, p_id)

            self.train_infos.append(train_info)

        if self.use_soft_update:
            self.trainer.soft_target_updates()
        else:
            if (self.num_episodes_collected - self.last_hard_update_episode) / self.hard_update_interval_episode >= 1:
                self.trainer.hard_target_updates()
                self.last_hard_update_episode = self.num_episodes_collected

    def save(self):
        for pid in self.policy_ids:
            policy_critic = self.policies[pid].critic
            critic_save_path = self.save_dir + '/' + str(pid)
            if not os.path.exists(critic_save_path):
                os.makedirs(critic_save_path)
            torch.save(policy_critic.state_dict(),
                       critic_save_path + '/critic.pt')

            policy_actor = self.policies[pid].actor
            actor_save_path = self.save_dir + '/' + str(pid)
            if not os.path.exists(actor_save_path):
                os.makedirs(actor_save_path)
            torch.save(policy_actor.state_dict(),
                       actor_save_path + '/actor.pt')

    def save_q(self):
        for pid in self.policy_ids:
            policy_Q = self.policies[pid].q_network
            p_save_path = self.save_dir + '/' + str(pid)
            if not os.path.exists(p_save_path):
                os.makedirs(p_save_path)
            torch.save(policy_Q.state_dict(), p_save_path + '/q_network.pt')

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        torch.save(self.trainer.mixer.state_dict(),
                   self.save_dir + '/mixer.pt')

    def restore(self):
        for pid in self.policy_ids:
            path = str(self.model_dir) + str(pid)
            print("load the pretrained model from {}".format(path))
            policy_critic_state_dict = torch.load(path + '/critic.pt')
            policy_actor_state_dict = torch.load(path + '/actor.pt')

            self.policies[pid].critic.load_state_dict(policy_critic_state_dict)
            self.policies[pid].actor.load_state_dict(policy_actor_state_dict)

    def restore_q(self):
        for pid in self.policy_ids:
            path = str(self.model_dir) + str(pid)
            print("load the pretrained model from {}".format(path))
            policy_q_state_dict = torch.load(path + '/q_network.pt')           
            self.policies[pid].q_network.load_state_dict(policy_q_state_dict)
            
        policy_mixer_state_dict = torch.load(str(self.model_dir) + '/mixer.pt')
        self.trainer.mixer.load_state_dict(policy_mixer_state_dict)

    def warmup(self, num_warmup_episodes):
        # fill replay buffer with enough episodes to begin training
        self.trainer.prep_rollout()
        warmup_rewards = []
        print("warm up...")
        for _ in range((num_warmup_episodes // self.num_envs) + 1):
            reward, _ = self.collecter(
                explore=True, training_episode=False, warmup=True)
            warmup_rewards.append(reward)
        warmup_reward = np.mean(warmup_rewards)
        print("warmup average episode rewards: ", warmup_reward)

    def eval(self):
        self.trainer.prep_rollout()

        eval_episode_rewards = []
        eval_metrics = []

        for _ in range(self.args.num_eval_episodes):
            eval_episode_reward, eval_metric = self.collecter(
                explore=False, training_episode=False, warmup=False)
            eval_episode_rewards.append(eval_episode_reward)
            eval_metrics.append(eval_metric)

        average_episode_reward = np.mean(eval_episode_rewards)
        print("eval average episode rewards is " + str(average_episode_reward))
        if self.use_wandb:
            wandb.log({'eval_average_episode_rewards': average_episode_reward},
                      step=self.total_env_steps)
        else:
            self.writter.add_scalars("eval_average_episode_rewards", {
                                     'eval_average_episode_rewards': average_episode_reward}, self.total_env_steps)

        self.log_env(eval_metrics, suffix="eval")

    def collect_rollout(self, explore=True, training_episode=True, warmup=False):
        p_id = "policy_0"
        policy = self.policies[p_id]

        env = self.env if training_episode or warmup else self.eval_env

        battles_won = 0

        obs, share_obs, avail_acts = env.reset()

        recurrent_hidden_states_batch = np.zeros(
            (self.num_envs * len(self.policy_agents[p_id]), self.hidden_size))
        if is_multidiscrete(self.policy_info[p_id]['act_space']):
            self.sum_act_dim = int(np.sum(self.policy_act_dim[p_id]))
        else:
            self.sum_act_dim = self.policy_act_dim[p_id]

        last_acts_batch = np.zeros(
            (self.num_envs * len(self.policy_agents[p_id]), self.sum_act_dim))

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

        episode_obs[p_id] = np.zeros((self.episode_length, *obs.shape))
        episode_share_obs[p_id] = np.zeros(
            (self.episode_length, *share_obs.shape))
        episode_acts[p_id] = np.zeros((self.episode_length, *avail_acts.shape))
        episode_rewards[p_id] = np.zeros(
            (self.episode_length, self.num_envs, len(self.policy_agents[p_id]), 1))
        episode_next_obs[p_id] = np.zeros_like(episode_obs[p_id])
        episode_next_share_obs[p_id] = np.zeros_like(episode_share_obs[p_id])
        episode_dones[p_id] = np.ones(
            (self.episode_length, self.num_envs, len(self.policy_agents[p_id]), 1))
        episode_dones_env[p_id] = np.ones(
            (self.episode_length, self.num_envs, 1))
        episode_avail_acts[p_id] = np.ones(
            (self.episode_length, *avail_acts.shape))
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
                acts_batch = policy.get_random_actions(
                    obs_batch, avail_acts_batch)
                # get new rnn hidden state
                _, recurrent_hidden_states_batch, _ = policy.get_actions(obs_batch,
                                                                         last_acts_batch,
                                                                         recurrent_hidden_states_batch,
                                                                         avail_acts_batch)

            else:
                # get actions with exploration noise (eps-greedy/Gaussian)
                if self.algorithm_name == "rmasac":
                    acts_batch, recurrent_hidden_states_batch, _ = policy.get_actions(obs_batch,
                                                                                      last_acts_batch,
                                                                                      recurrent_hidden_states_batch,
                                                                                      avail_acts_batch,
                                                                                      sample=explore)
                else:
                    acts_batch, recurrent_hidden_states_batch, _ = policy.get_actions(obs_batch,
                                                                                      last_acts_batch,
                                                                                      recurrent_hidden_states_batch,
                                                                                      avail_acts_batch,
                                                                                      t_env=self.total_env_steps,
                                                                                      explore=explore,
                                                                                      use_target=False,
                                                                                      use_gumbel=False)
            # update rnn hidden state
            recurrent_hidden_states_batch = recurrent_hidden_states_batch.detach().numpy()
            if not isinstance(acts_batch, np.ndarray):
                acts_batch = acts_batch.detach().numpy()
            last_acts_batch = acts_batch
            env_acts = np.split(acts_batch, self.num_envs)

            # env step and store the relevant episode information
            next_obs, next_share_obs, rewards, dones, infos, next_avail_acts = env.step(
                env_acts)
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
                        if infos[i][0]['won']:  # take one agent
                            battles_won += 1
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

        if ((self.total_env_steps - self.last_log_T) / self.log_interval) >= 1:
            wandb.log({str(p_id) + '/min_episode_share_obs': np.array(
                episode_share_obs[p_id]).min()}, step=self.total_env_steps)
            wandb.log({str(p_id) + '/max_episode_share_obs': np.array(
                episode_share_obs[p_id]).max()}, step=self.total_env_steps)
            wandb.log({str(p_id) + '/mean_episode_share_obs': np.array(
                episode_share_obs[p_id]).mean()}, step=self.total_env_steps)

        average_episode_reward = np.mean(
            np.sum(accumulated_rewards[p_id], axis=0))

        return average_episode_reward, battles_won

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

        average_episode_reward = np.mean(self.train_episode_rewards)
        print("train average episode rewards is " + str(average_episode_reward))
        if self.use_wandb:
            wandb.log({'train_average_episode_rewards': average_episode_reward},
                      step=self.total_env_steps)
        else:
            self.writter.add_scalars("train_average_episode_rewards", {
                                     'train_average_episode_rewards': average_episode_reward}, self.total_env_steps)

        self.log_env(self.train_metrics)
        self.log_clear()

    def log_env(self, metrics, suffix="train"):
        if len(metrics) > 0:
            metric = np.mean(metrics)
            print(suffix + " win rate is " + str(metric))
            if self.use_wandb:
                wandb.log({suffix + '_win_rate': metric}, step=self.total_env_steps)
            else:
                self.writter.add_scalars(suffix + '_win_rate', {suffix + '_win_rate': metric}, self.total_env_steps)

    def log_clear(self):
        self.train_episode_rewards = []
        self.train_metrics = []

    def log_train(self, policy_id, train_info):
        for k, v in train_info.items():
            policy_k = str(policy_id) + '/' + k
            if self.use_wandb:
                wandb.log({policy_k: v}, step=self.total_env_steps)
            else:
                self.writter.add_scalars(policy_k, {policy_k: v}, self.total_env_steps)