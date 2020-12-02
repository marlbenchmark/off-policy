from .envs.box_locking import BoxLockingEnv
from .envs.blueprint_construction import BlueprintConstructionEnv
from .envs.hide_and_seek import HideAndSeekEnv
import numpy as np
from offpolicy.utils.util import MultiDiscrete
from functools import reduce


class HNSEnv(object):

    def __init__(
            self,
            args
    ):
        self.obs_instead_of_state = args.use_obs_instead_of_state
        if args.env_name == "BoxLocking":
            self.num_agents = args.num_agents
            self.env = BoxLockingEnv(args)
            self.order_obs = ['agent_qpos_qvel',
                              'box_obs', 'ramp_obs', 'observation_self']
            self.mask_order_obs = ['mask_aa_obs',
                                   'mask_ab_obs', 'mask_ar_obs', None]
        elif args.env_name == "BlueprintConstruction":
            self.num_agents = args.num_agents
            self.env = BlueprintConstructionEnv(args)
            self.order_obs = ['agent_qpos_qvel', 'box_obs',
                              'ramp_obs', 'construction_site_obs', 'observation_self']
            self.mask_order_obs = [None, None, None, None, None]
        elif args.env_name == "HideAndSeek":
            self.env = HideAndSeekEnv(args)
            self.num_seekers = args.num_seekers
            self.num_hiders = args.num_hiders
            self.num_agents = self.num_seekers + self.num_hiders
            self.order_obs = ['agent_qpos_qvel', 'box_obs',
                              'ramp_obs', 'food_obs', 'observation_self']
            self.mask_order_obs = [
                'mask_aa_obs', 'mask_ab_obs', 'mask_ar_obs', 'mask_af_obs', None]
        else:
            raise NotImplementedError

        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        self.action_movement_dim = []

        for agent_id in range(self.num_agents):
            # deal with dict action space
            self.action_movement = self.env.action_space['action_movement'][agent_id].nvec
            self.action_movement_dim.append(len(self.action_movement))
            action_glueall = self.env.action_space['action_glueall'][agent_id].n
            action_vec = np.append(self.action_movement, action_glueall)
            if 'action_pull' in self.env.action_space.spaces.keys():
                action_pull = self.env.action_space['action_pull'][agent_id].n
                action_vec = np.append(action_vec, action_pull)
            action_space = MultiDiscrete([[0, vec-1] for vec in action_vec])
            self.action_space.append(action_space)
            # deal with dict obs space
            obs_space = []
            obs_dim = 0
            for key in self.order_obs:
                if key in self.env.observation_space.spaces.keys():
                    space = list(self.env.observation_space[key].shape)
                    if len(space) < 2:
                        space.insert(0, 1)
                    obs_space.append(space)
                    obs_dim += reduce(lambda x, y: x*y, space)
            obs_space.insert(0, obs_dim)
            self.observation_space.append(obs_space)
            if self.obs_instead_of_state:
                self.share_observation_space.append([obs_space[0] * self.num_agents, [self.num_agents, obs_space[0]]])
            else:
                self.share_observation_space.append(obs_space)

    def seed(self, seed=None):
        if seed is None:
            self.env.seed(1)
        else:
            self.env.seed(seed)

    def reset(self):
        dict_obs = self.env.reset()

        for i, key in enumerate(self.order_obs):
            if key in self.env.observation_space.spaces.keys():
                if self.mask_order_obs[i] == None:
                    temp_share_obs = dict_obs[key].reshape(
                        self.num_agents, -1).copy()
                    temp_obs = temp_share_obs.copy()
                else:
                    temp_share_obs = dict_obs[key].reshape(
                        self.num_agents, -1).copy()
                    temp_mask = dict_obs[self.mask_order_obs[i]].copy()
                    temp_obs = dict_obs[key].copy()
                    temp_mask = temp_mask.astype(bool)
                    mins_temp_mask = ~temp_mask
                    temp_obs[mins_temp_mask] = np.zeros(
                        (mins_temp_mask.sum(), temp_obs.shape[2]))
                    temp_obs = temp_obs.reshape(self.num_agents, -1)
                if i == 0:
                    obs = temp_obs.copy()
                    share_obs = temp_share_obs.copy()
                else:
                    obs = np.concatenate((obs, temp_obs), axis=1)
                    share_obs = np.concatenate(
                        (share_obs, temp_share_obs), axis=1)
        if self.obs_instead_of_state:
        concat_obs = np.concatenate(obs, axis=0)
        share_obs = np.expand_dims(concat_obs, 0).repeat(
            self.num_agents, axis=0)

        return obs, share_obs, None

    def step(self, actions):

        action_movement = []
        action_pull = []
        action_glueall = []
        for agent_id in range(self.num_agents):
            temp_action_movement = np.zeros_like(self.action_movement)
            for k, movement_dim in enumerate(self.action_movement_dim):
                temp_action_movement[k] = np.argmax(
                    actions[agent_id][k * movement_dim:(k + 1) * movement_dim - 1])
            action_movement.append(temp_action_movement)
            glueall_dim_start = np.sum(self.action_movement_dim)
            action_glueall.append(
                int(np.argmax(actions[agent_id][glueall_dim_start:glueall_dim_start + 2])))

            if 'action_pull' in self.env.action_space.spaces.keys():
                action_pull.append(int(np.argmax(actions[agent_id][-2:])))
        action_movement = np.stack(action_movement, axis=0)
        action_glueall = np.stack(action_glueall, axis=0)
        if 'action_pull' in self.env.action_space.spaces.keys():
            action_pull = np.stack(action_pull, axis=0)
        env_actions = {'action_movement': action_movement,
                       'action_pull': action_pull,
                       'action_glueall': action_glueall}

        dict_obs, rewards, done, infos = self.env.step(env_actions)
        if len(rewards.shape) < 2:
            rewards = rewards[:, np.newaxis]

        dones = [[done]] * self.num_agents

        if 'discard_episode' in infos.keys():
            if infos['discard_episode']:
                obs = None
                share_obs = None
            else:
                for i, key in enumerate(self.order_obs):
                    if key in self.env.observation_space.spaces.keys():
                        if self.mask_order_obs[i] == None:
                            temp_share_obs = dict_obs[key].reshape(
                                self.num_agents, -1).copy()
                            temp_obs = temp_share_obs.copy()
                        else:
                            temp_share_obs = dict_obs[key].reshape(
                                self.num_agents, -1).copy()
                            temp_mask = dict_obs[self.mask_order_obs[i]].copy()
                            temp_obs = dict_obs[key].copy()
                            temp_mask = temp_mask.astype(bool)
                            mins_temp_mask = ~temp_mask
                            temp_obs[mins_temp_mask] = np.zeros(
                                (mins_temp_mask.sum(), temp_obs.shape[2]))
                            temp_obs = temp_obs.reshape(self.num_agents, -1)
                        if i == 0:
                            obs = temp_obs.copy()
                            share_obs = temp_share_obs.copy()
                        else:
                            obs = np.concatenate((obs, temp_obs), axis=1)
                            share_obs = np.concatenate(
                                (share_obs, temp_share_obs), axis=1)
                if self.obs_instead_of_state:
                    concat_obs = np.concatenate(obs, axis=0)
                    share_obs = np.expand_dims(concat_obs, 0).repeat(
                        self.num_agents, axis=0)
        return obs, share_obs, rewards, dones, infos, None

    def render(self):
        self.env.render()
