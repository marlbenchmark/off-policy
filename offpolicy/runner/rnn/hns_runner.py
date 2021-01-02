import numpy as np
import wandb
import torch
import time
from offpolicy.utils.util import is_multidiscrete
from offpolicy.runner.rnn.base_runner import RecRunner

class HNSRunner(RecRunner):
    def __init__(self, config):
        """Runner class for the Hide and Seek environment. See parent class for more information."""
        super(HNSRunner, self).__init__(config)    
        # fill replay buffer with random actions
        num_warmup_episodes = max((self.batch_size, self.args.num_random_episodes))
        self.warmup(num_warmup_episodes)
        self.start = time.time()
        self.log_clear()

    def eval(self):
        """Collect episodes to evaluate the policy."""
        self.trainer.prep_rollout()

        eval_infos = {}
        eval_infos['success_rate'] = []
        eval_infos['average_episode_rewards'] = []

        for _ in range(self.args.num_eval_episodes):
            env_info = self.collecter( explore=False, training_episode=False, warmup=False)
            
            for k, v in env_info.items():
                eval_infos[k].append(v)

        self.log_env(eval_infos, suffix="eval_")

    @torch.no_grad()    
    def collect_rollout(self, explore=True, training_episode=True, warmup=False):
        """
        Collect a rollout and store it in the buffer. All agents share a single policy.
        :param explore: (bool) whether to use an exploration strategy when collecting the episoide.
        :param training_episode: (bool) whether this episode is used for evaluation or training.
        :param warmup: (bool) whether this episode is being collected during warmup phase.

        :return env_info: (dict) contains information about the rollout (total rewards, etc).
        """
        env_info = {}
        p_id = "policy_0"
        policy = self.policies[p_id]

        env = self.env if training_episode or warmup else self.eval_env

        success_to_collect_one_episode = False
        while not success_to_collect_one_episode:
            discard_episode = 0

            obs, share_obs, _ = env.reset()

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
            episode_next_obs[p_id] = np.zeros_like(episode_obs[p_id])
            episode_next_share_obs[p_id] = np.zeros_like(episode_share_obs[p_id])
            episode_dones[p_id] = np.ones((self.episode_length, self.num_envs, len(self.policy_agents[p_id]), 1), dtype=np.float32)
            episode_dones_env[p_id] = np.ones((self.episode_length, self.num_envs, 1), dtype=np.float32)
            episode_avail_acts[p_id] = None
            episode_next_avail_acts[p_id] = None
            accumulated_rewards[p_id] = []

            t = 0
            env_t = 0
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
                    acts_batch, rnn_states_batch, _ = policy.get_actions(obs_batch,
                                                                        last_acts_batch,
                                                                        rnn_states_batch,
                                                                        t_env=self.total_env_steps,
                                                                        explore=explore)
                # update rnn hidden state
                rnn_states_batch = rnn_states_batch.detach()
                last_acts_batch = acts_batch
                if not isinstance(acts_batch, np.ndarray):
                    acts_batch = acts_batch.cpu().detach().numpy()
                
                env_acts = np.split(acts_batch, self.num_envs)

                # env step and store the relevant episode information
                next_obs, next_share_obs, rewards, dones, infos, _ = env.step(env_acts)
                t += 1
                assert self.num_envs == 1, ("only one env is support here.")
                env_t += self.num_envs

                dones_env = np.all(dones, axis=1)
                terminate_episodes = np.any(dones_env) or t == self.episode_length

                for i in range(self.num_envs):
                    if 'discard_episode' in infos[i].keys():
                        if infos[i]['discard_episode']:  # take one agent
                            discard_episode += 1
                if discard_episode > 0:
                    break

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

                if terminate_episodes:
                    for i in range(self.num_envs):
                        if 'success' in infos[i].keys():
                            env_info['success_rate'] = 1 if infos[i]['success'] else 0
                    break

            if discard_episode == 0:
                if explore:
                    success_to_collect_one_episode = True
                    if training_episode:
                        self.total_env_steps += env_t
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
                else:
                    success_to_collect_one_episode = True

        average_episode_rewards = np.mean(np.sum(accumulated_rewards[p_id], axis=0))
        env_info['average_episode_rewards'] = average_episode_rewards

        return env_info

    def log(self):
        end = time.time()
        print("\n Env {} Algo {} Exp {} runs total num timesteps {}/{}, FPS {}.\n"
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
        """
        Log information related to the environment.
        :param env_info: (dict) contains information about the environment.
        :param suffix: (str) optional string to add to end of keys contained in env_info.
        """
        if self.env_name == "BoxLocking" or self.env_name == "BlueprintConstruction":
            for k, v in env_info.items():
                if len(v) > 0:
                    v = np.mean(v)
                    suffix_k = k if suffix is None else suffix + k 
                    print(suffix_k + " is " + str(v))
                    if self.use_wandb:
                        wandb.log({suffix_k: v}, step=self.total_env_steps)
                    else:
                        self.writter.add_scalars(suffix_k, {suffix_k: v}, self.total_env_steps)

    def log_clear(self):
        """See parent class."""
        self.env_infos = {}

        self.env_infos['average_episode_rewards'] = []
        self.env_infos['success_rate'] = []
