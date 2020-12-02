import numpy as np
import torch
import random
from offpolicy.utils.util import get_dim_from_space
from offpolicy.utils.segment_tree import SumSegmentTree, MinSegmentTree


def _cast(x):
    return x.transpose(2, 0, 1, 3)


class RecReplayBuffer(object):
    def __init__(self, policy_info, policy_agents, buffer_size, episode_length, use_same_share_obs, use_avail_acts, use_reward_normalization=False):
        self.policy_info = policy_info

        self.policy_buffers = {p_id: RecPolicyBuffer(buffer_size,
                                                     episode_length,
                                                     len(policy_agents[p_id]),
                                                     self.policy_info[p_id]['obs_space'],
                                                     self.policy_info[p_id]['share_obs_space'],
                                                     self.policy_info[p_id]['act_space'],
                                                     use_same_share_obs,
                                                     use_avail_acts,
                                                     use_reward_normalization)
                               for p_id in self.policy_info.keys()}

    def __len__(self):
        return self.policy_buffers['policy_0'].filled_i

    def insert(self, num_insert_episodes, obs, share_obs, acts, rewards,
               next_obs, next_share_obs, dones, dones_env,
               avail_acts, next_avail_acts):

        for p_id in self.policy_info.keys():
            idx_range = self.policy_buffers[p_id].insert(num_insert_episodes,
                                                         np.array(obs[p_id]), np.array(share_obs[p_id]), np.array(
                                                             acts[p_id]), np.array(rewards[p_id]),
                                                         np.array(next_obs[p_id]), np.array(next_share_obs[p_id]), np.array(
                                                             dones[p_id]), np.array(dones_env[p_id]),
                                                         np.array(avail_acts[p_id]), np.array(next_avail_acts[p_id]))
        return idx_range

    def sample(self, batch_size):
        inds = np.random.choice(self.__len__(), batch_size)
        obs, share_obs, acts, rewards, next_obs, next_share_obs, dones, dones_env, avail_acts, next_avail_acts = {
        }, {}, {}, {}, {}, {}, {}, {}, {}, {}
        for p_id in self.policy_info.keys():
            obs[p_id], share_obs[p_id], acts[p_id], rewards[p_id], next_obs[p_id], next_share_obs[p_id], dones[
                p_id], dones_env[p_id], avail_acts[p_id], next_avail_acts[p_id] = self.policy_buffers[p_id].sample_inds(inds)

        return obs, share_obs, acts, rewards, next_obs, next_share_obs, dones, dones_env, avail_acts, next_avail_acts, None, None


class RecPolicyBuffer(object):
    def __init__(self, buffer_size, episode_length, num_agents, obs_space, share_obs_space, act_space, use_same_share_obs, use_avail_acts, use_reward_normalization=False):

        self.buffer_size = buffer_size
        self.episode_length = episode_length
        self.num_agents = num_agents
        self.use_same_share_obs = use_same_share_obs
        self.use_avail_acts = use_avail_acts
        self.use_reward_normalization = use_reward_normalization
        self.filled_i = 0
        self.current_i = 0

        # obs
        if obs_space.__class__.__name__ == 'Box':
            obs_shape = obs_space.shape
            share_obs_shape = share_obs_space.shape
        elif obs_space.__class__.__name__ == 'list':
            obs_shape = obs_space
            share_obs_shape = share_obs_space
        else:
            raise NotImplementedError

        self.obs = np.zeros((self.episode_length, self.buffer_size,
                             self.num_agents, obs_shape[0]), dtype=np.float32)

        if self.use_same_share_obs:
            self.share_obs = np.zeros(
                (self.episode_length, self.buffer_size, share_obs_shape[0]), dtype=np.float32)
        else:
            self.share_obs = np.zeros(
                (self.episode_length, self.buffer_size, self.num_agents, share_obs_shape[0]), dtype=np.float32)

        self.next_obs = np.zeros_like(self.obs, dtype=np.float32)
        self.next_share_obs = np.zeros_like(self.share_obs, dtype=np.float32)

        # action
        act_dim = np.sum(get_dim_from_space(act_space))
        self.acts = np.zeros((self.episode_length, self.buffer_size,
                              self.num_agents, act_dim), dtype=np.float32)
        if self.use_avail_acts:
            self.avail_acts = np.ones_like(self.acts, dtype=np.float32)
            self.next_avail_acts = np.ones_like(
                self.avail_acts, dtype=np.float32)

        # rewards
        self.rewards = np.zeros(
            (self.episode_length, self.buffer_size, self.num_agents, 1), dtype=np.float32)

        # default to done being True
        self.dones = np.ones_like(self.rewards, dtype=np.float32)
        self.dones_env = np.ones(
            (self.episode_length, self.buffer_size, 1), dtype=np.float32)

    def __len__(self):
        return self.filled_i

    def insert(self, num_insert_episodes, obs, share_obs, acts, rewards,
               next_obs, next_share_obs, dones, dones_env,
               avail_acts=None, next_avail_acts=None):
        # obs: [step, episode, agent, dim]
        episode_length = obs.shape[0]
        assert episode_length == self.episode_length, ("different dimension!")
        left_size = self.buffer_size - self.current_i  # 100 - 99 - 1 = 0
        if left_size >= num_insert_episodes:
            self.obs[:, self.current_i:(
                self.current_i + num_insert_episodes)] = obs.copy()  # 98 + 1 = 99
            self.share_obs[:, self.current_i:(
                self.current_i + num_insert_episodes)] = share_obs.copy()
            self.acts[:, self.current_i:(
                self.current_i + num_insert_episodes)] = acts.copy()
            self.rewards[:, self.current_i:(
                self.current_i + num_insert_episodes)] = rewards.copy()
            self.next_obs[:, self.current_i:(
                self.current_i + num_insert_episodes)] = next_obs.copy()
            self.next_share_obs[:, self.current_i:(
                self.current_i + num_insert_episodes)] = next_share_obs.copy()
            self.dones[:, self.current_i:(
                self.current_i + num_insert_episodes)] = dones.copy()
            self.dones_env[:, self.current_i:(
                self.current_i + num_insert_episodes)] = dones_env.copy()
            if self.use_avail_acts:
                self.avail_acts[:, self.current_i:(
                    self.current_i + num_insert_episodes)] = avail_acts.copy()
                self.next_avail_acts[:, self.current_i:(
                    self.current_i + num_insert_episodes)] = next_avail_acts.copy()

            idx_range = (self.current_i, self.current_i + num_insert_episodes)
            self.current_i += num_insert_episodes  # 99

        else:
            self.obs[:, self.current_i:(
                self.current_i + left_size)] = obs[:, 0:left_size].copy()
            self.share_obs[:, self.current_i:(
                self.current_i + left_size)] = share_obs[:, 0:left_size].copy()
            self.acts[:, self.current_i:(
                self.current_i + left_size)] = acts[:, 0:left_size].copy()
            self.rewards[:, self.current_i:(
                self.current_i + left_size)] = rewards[:, 0:left_size].copy()
            self.next_obs[:, self.current_i:(
                self.current_i + left_size)] = next_obs[:, 0:left_size].copy()
            self.next_share_obs[:, self.current_i:(
                self.current_i + left_size)] = next_share_obs[:, 0:left_size].copy()
            self.dones[:, self.current_i:(
                self.current_i + left_size)] = dones[:, 0:left_size].copy()
            self.dones_env[:, self.current_i:(
                self.current_i + left_size)] = dones_env[:, 0:left_size].copy()
            if self.use_avail_acts:
                self.avail_acts[:, self.current_i:(
                    self.current_i + left_size)] = avail_acts[:, 0:left_size].copy()
                self.next_avail_acts[:, self.current_i:(
                    self.current_i + left_size)] = next_avail_acts[:, 0:left_size].copy()

            self.current_i = 0
            self.filled_i = self.buffer_size

            self.obs[:, self.current_i:(
                self.current_i + num_insert_episodes - left_size)] = obs[:, left_size:].copy()
            self.share_obs[:, self.current_i:(
                self.current_i + num_insert_episodes - left_size)] = share_obs[:, left_size:].copy()
            self.acts[:, self.current_i:(
                self.current_i + num_insert_episodes - left_size)] = acts[:, left_size:].copy()
            self.rewards[:, self.current_i:(
                self.current_i + num_insert_episodes - left_size)] = rewards[:, left_size:].copy()
            self.next_obs[:, self.current_i:(
                self.current_i + num_insert_episodes - left_size)] = next_obs[:, left_size:].copy()
            self.next_share_obs[:, self.current_i:(
                self.current_i + num_insert_episodes - left_size)] = next_share_obs[:, left_size:].copy()
            self.dones[:, self.current_i:(
                self.current_i + num_insert_episodes - left_size)] = dones[:, left_size:].copy()
            self.dones_env[:, self.current_i:(
                self.current_i + num_insert_episodes - left_size)] = dones_env[:, left_size:].copy()
            if self.use_avail_acts:
                self.avail_acts[:, self.current_i:(
                    self.current_i + num_insert_episodes - left_size)] = avail_acts[:, left_size:].copy()
                self.next_avail_acts[:, self.current_i:(
                    self.current_i + num_insert_episodes - left_size)] = next_avail_acts[:, left_size:].copy()
            self.current_i = num_insert_episodes - left_size

            idx_range = (-left_size, -left_size + num_insert_episodes)

        if self.filled_i < self.buffer_size:
            self.filled_i += num_insert_episodes

        return idx_range

    def sample_inds(self, sample_inds):

        obs = _cast(self.obs[:, sample_inds])
        acts = _cast(self.acts[:, sample_inds])
        if self.use_reward_normalization:
            # mean std
            # [length, envs, agents, 1]
            # [length, envs, 1]
            all_dones_env = np.tile(np.expand_dims(
                self.dones_env[:, :self.filled_i], -1), (1, 1, self.num_agents, 1))
            first_step_dones_env = np.zeros(
                (1, self.filled_i, self.num_agents, 1))
            curr_dones_env = np.concatenate(
                (first_step_dones_env, all_dones_env[:self.episode_length-1]))
            temp_rewards = self.rewards[:, :self.filled_i].copy()
            temp_rewards[curr_dones_env == 1.0] = np.nan

            mean_reward = np.nanmean(temp_rewards)
            std_reward = np.nanstd(temp_rewards)
            rewards = _cast(
                (self.rewards[:, sample_inds] - mean_reward) / std_reward)
        else:
            rewards = _cast(self.rewards[:, sample_inds])
        next_obs = _cast(self.next_obs[:, sample_inds])

        if self.use_same_share_obs:
            share_obs = self.share_obs[:, sample_inds]
            next_share_obs = self.next_share_obs[:, sample_inds]
        else:
            share_obs = _cast(self.share_obs[:, sample_inds])
            next_share_obs = _cast(self.next_share_obs[:, sample_inds])

        dones = _cast(self.dones[:, sample_inds])
        dones_env = self.dones_env[:, sample_inds]

        if self.use_avail_acts:
            avail_acts = _cast(self.avail_acts[:, sample_inds])
            next_avail_acts = _cast(self.next_avail_acts[:, sample_inds])
        else:
            avail_acts = None
            next_avail_acts = None

        return obs, share_obs, acts, rewards, next_obs, next_share_obs, dones, dones_env, avail_acts, next_avail_acts


class PrioritizedRecReplayBuffer(RecReplayBuffer):
    def __init__(self, alpha, policy_info, policy_agents, buffer_size, episode_length, use_same_share_obs, use_avail_acts, use_reward_normalization=False):
        super(PrioritizedRecReplayBuffer, self).__init__(policy_info, policy_agents, buffer_size,
                                                         episode_length, use_same_share_obs, use_avail_acts, use_reward_normalization)
        self.alpha = alpha
        self.policy_info = policy_info
        it_capacity = 1
        while it_capacity < buffer_size:
            it_capacity *= 2

        self._it_sums = {p_id: SumSegmentTree(
            it_capacity) for p_id in self.policy_info.keys()}
        self._it_mins = {p_id: MinSegmentTree(
            it_capacity) for p_id in self.policy_info.keys()}
        self.max_priorities = {p_id: 1.0 for p_id in self.policy_info.keys()}

    def insert(self, num_insert_episodes, obs, share_obs, acts, rewards,
               next_obs, next_share_obs, dones, dones_env,
               avail_acts=None, next_avail_acts=None):
        idx_range = super().insert(num_insert_episodes, obs, share_obs, acts, rewards,
                                   next_obs, next_share_obs, dones, dones_env,
                                   avail_acts, next_avail_acts)
        for idx in range(idx_range[0], idx_range[1]):
            for p_id in self.policy_info.keys():
                self._it_sums[p_id][idx] = self.max_priorities[p_id] ** self.alpha
                self._it_mins[p_id][idx] = self.max_priorities[p_id] ** self.alpha

        return idx_range

    def _sample_proportional(self, batch_size, p_id=None):
        mass = []
        total = self._it_sums[p_id].sum(0, len(self) - 1)
        mass = np.random.random(size=batch_size) * total
        idx = self._it_sums[p_id].find_prefixsum_idx(mass)
        return idx

    def sample(self, batch_size, beta=0, p_id=None):
        assert len(
            self) > batch_size, "Cannot sample with no completed episodes in the buffer!"
        assert beta > 0

        batch_inds = self._sample_proportional(batch_size, p_id)

        weights = []
        p_min = self._it_mins[p_id].min() / self._it_sums[p_id].sum()
        max_weight = (p_min * len(self)) ** (-beta)
        p_sample = self._it_sums[p_id][batch_inds] / self._it_sums[p_id].sum()
        weights = (p_sample * len(self)) ** (-beta) / max_weight

        obs, share_obs, acts, rewards, next_obs, next_share_obs, dones, dones_env, avail_acts, next_avail_acts = {
        }, {}, {}, {}, {}, {}, {}, {}, {}, {}
        for p_id in self.policy_info.keys():
            p_buffer = self.policy_buffers[p_id]
            obs[p_id], share_obs[p_id], acts[p_id], rewards[p_id], next_obs[p_id], next_share_obs[p_id], dones[
                p_id], dones_env[p_id], avail_acts[p_id], next_avail_acts[p_id] = p_buffer.sample_inds(batch_inds)

        return obs, share_obs, acts, rewards, next_obs, next_share_obs, dones, dones_env, avail_acts, next_avail_acts, weights, batch_inds

    def update_priorities(self, idxes, priorities, p_id=None):
        """
        Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        :param idxes: ([int]) List of idxes of sampled transitions
        :param priorities: ([float]) List of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        assert np.min(priorities) > 0
        assert np.min(idxes) >= 0
        assert np.max(idxes) < len(self)

        self._it_sums[p_id][idxes] = priorities ** self.alpha
        self._it_mins[p_id][idxes] = priorities ** self.alpha

        self.max_priorities[p_id] = max(
            self.max_priorities[p_id], np.max(priorities))
