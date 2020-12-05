import numpy as np
import torch
import random
from offpolicy.utils.util import get_dim_from_space
from offpolicy.utils.segment_tree import SumSegmentTree, MinSegmentTree


def _cast(x):
    return x.transpose(1, 0, 2)


class MlpReplayBuffer(object):
    def __init__(self, policy_info, policy_agents, buffer_size, use_same_share_obs, use_avail_acts, use_reward_normalization=False):

        self.policy_info = policy_info

        self.policy_buffers = {p_id: MlpPolicyBuffer(buffer_size,
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

    def insert(self, num_insert_steps, obs, share_obs, acts, rewards,
               next_obs, next_share_obs, dones, dones_env,
               avail_acts, next_avail_acts):

        for p_id in self.policy_info.keys():
            idx_range = self.policy_buffers[p_id].insert(num_insert_steps,
                                                         np.array(obs[p_id]), np.array(share_obs[p_id]), np.array(
                                                             acts[p_id]), np.array(rewards[p_id]),
                                                         np.array(next_obs[p_id]), np.array(next_share_obs[p_id]), np.array(
                                                             dones[p_id]), np.array(dones_env[p_id]),
                                                         np.array(avail_acts[p_id]), np.array(next_avail_acts[p_id]))
        return idx_range

    def sample(self, batch_size):
        inds = np.random.choice(len(self), batch_size)
        obs, share_obs, acts, rewards, next_obs, next_share_obs, dones, dones_env, avail_acts, next_avail_acts = {
        }, {}, {}, {}, {}, {}, {}, {}, {}, {}
        for p_id in self.policy_info.keys():
            obs[p_id], share_obs[p_id], acts[p_id], rewards[p_id], next_obs[p_id], next_share_obs[p_id], dones[
                p_id], dones_env[p_id], avail_acts[p_id], next_avail_acts[p_id] = self.policy_buffers[p_id].sample_inds(inds)

        return obs, share_obs, acts, rewards, next_obs, next_share_obs, dones, dones_env, avail_acts, next_avail_acts, None, None


class MlpPolicyBuffer(object):
    def __init__(self, buffer_size, num_agents, obs_space, share_obs_space, act_space, use_same_share_obs, use_avail_acts, use_reward_normalization=False):

        self.buffer_size = buffer_size
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

        self.obs = np.zeros(
            (self.buffer_size, self.num_agents, obs_shape[0]), dtype=np.float32)

        if self.use_same_share_obs:
            self.share_obs = np.zeros(
                (self.buffer_size, share_obs_shape[0]), dtype=np.float32)
        else:
            self.share_obs = np.zeros(
                (self.buffer_size, self.num_agents, share_obs_shape[0]), dtype=np.float32)

        self.next_obs = np.zeros_like(self.obs, dtype=np.float32)
        self.next_share_obs = np.zeros_like(self.share_obs, dtype=np.float32)

        # action
        act_dim = np.sum(get_dim_from_space(act_space))
        self.acts = np.zeros(
            (self.buffer_size, self.num_agents, act_dim), dtype=np.float32)
        if self.use_avail_acts:
            self.avail_acts = np.ones_like(self.acts, dtype=np.float32)
            self.next_avail_acts = np.ones_like(
                self.avail_acts, dtype=np.float32)

        # rewards
        self.rewards = np.zeros(
            (self.buffer_size, self.num_agents, 1), dtype=np.float32)

        # default to done being True
        self.dones = np.ones_like(self.rewards, dtype=np.float32)
        self.dones_env = np.ones((self.buffer_size, 1), dtype=np.float32)

    def __len__(self):
        return self.filled_i

    def insert(self, num_insert_steps, obs, share_obs, acts, rewards,
               next_obs, next_share_obs, dones, dones_env,
               avail_acts=None, next_avail_acts=None):
        # obs: [step, episode, agent, dim]
        episode_length = obs.shape[0]
        assert episode_length == num_insert_steps, ("different size!")

        if self.current_i + num_insert_steps <= self.buffer_size:
            idx_range = np.arange(self.current_i, self.current_i + num_insert_steps)       
        else:
            num_left_steps = self.current_i + num_insert_steps - self.buffer_size
            idx_range = np.concatenate((np.arange(self.current_i, self.buffer_size), np.arange(num_left_steps)))

        self.obs[idx_range] = obs.copy()
        self.share_obs[idx_range] = share_obs.copy()
        self.acts[idx_range] = acts.copy()
        self.rewards[idx_range] = rewards.copy()
        self.next_obs[idx_range] = next_obs.copy()
        self.next_share_obs[idx_range] = next_share_obs.copy()
        self.dones[idx_range] = dones.copy()
        self.dones_env[idx_range] = dones_env.copy()
        if self.use_avail_acts:
            self.avail_acts[idx_range] = avail_acts.copy()
            self.next_avail_acts[idx_range] = next_avail_acts.copy()

        self.current_i = idx_range[-1] + 1 
        self.filled_i = min(self.filled_i + len(idx_range), self.buffer_size)

        return idx_range

    def sample_inds(self, sample_inds):

        obs = _cast(self.obs[sample_inds])
        acts = _cast(self.acts[sample_inds])
        rewards = _cast(self.rewards[sample_inds])
        if self.use_reward_normalization:
            mean_reward = self.rewards[:self.filled_i].mean()
            std_reward = self.rewards[:self.filled_i].std()
            rewards = _cast(
                (self.rewards[sample_inds] - mean_reward) / std_reward)
        else:
            rewards = _cast(self.rewards[sample_inds])

        next_obs = _cast(self.next_obs[sample_inds])

        if self.use_same_share_obs:
            share_obs = self.share_obs[sample_inds]
            next_share_obs = self.next_share_obs[sample_inds]
        else:
            share_obs = _cast(self.share_obs[sample_inds])
            next_share_obs = _cast(self.next_share_obs[sample_inds])

        dones = _cast(self.dones[sample_inds])
        dones_env = self.dones_env[sample_inds]

        if self.use_avail_acts:
            avail_acts = _cast(self.avail_acts[sample_inds])
            next_avail_acts = _cast(self.next_avail_acts[sample_inds])
        else:
            avail_acts = None
            next_avail_acts = None

        return obs, share_obs, acts, rewards, next_obs, next_share_obs, dones, dones_env, avail_acts, next_avail_acts


class PrioritizedMlpReplayBuffer(MlpReplayBuffer):
    def __init__(self, alpha, policy_info, policy_agents, buffer_size, use_same_share_obs, use_avail_acts, use_reward_normalization=False):
        super(PrioritizedMlpReplayBuffer, self).__init__(policy_info, policy_agents,
                                                         buffer_size, use_same_share_obs, use_avail_acts, use_reward_normalization)
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

    def insert(self, num_insert_steps, obs, share_obs, acts, rewards,
               next_obs, next_share_obs, dones, dones_env,
               avail_acts=None, next_avail_acts=None):
        idx_range = super().insert(num_insert_steps, obs, share_obs, acts, rewards,
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
