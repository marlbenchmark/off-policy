import numpy as np
import gym
from envs.hns.wrappers.multi_agent import (SplitMultiAgentActions, SplitObservations,
                                           SelectKeysWrapper)
from envs.hns.wrappers.util import (DiscretizeActionWrapper, MaskActionWrapper,
                                    DiscardMujocoExceptionEpisodes, SpoofEntityWrapper,
                                    AddConstantObservationsWrapper,
                                    ConcatenateObsWrapper, NumpyArrayRewardWrapper)
from envs.hns.wrappers.manipulation import (GrabObjWrapper, GrabClosestWrapper, TimeWrapper,
                                            LockObjWrapper, LockAllWrapper)
from envs.hns.wrappers.lidar import Lidar
from envs.hns.wrappers.team import TeamMembership
from envs.hns.wrappers.line_of_sight import AgentAgentObsMask2D, AgentGeomObsMask2D
from envs.hns.envs.base import Base
from envs.hns.modules.agents import Agents, AgentManipulation
from envs.hns.modules.construction_sites import ConstructionSites
from envs.hns.modules.walls import WallScenarios, RandomWalls
from envs.hns.modules.objects import Boxes, LidarSites
from envs.hns.modules.world import FloorAttributes, WorldConstants
from envs.hns.modules.util import (uniform_placement, center_placement,
                                   uniform_placement_middle)


class ConstructionDistancesWrapper(gym.ObservationWrapper):
    '''
        Calculates the distance between every pair of boxes, between boxes and
        construction sites, and between box corners and construction site corners.
        This wrapper should be only be applied if the both the Boxes module (with
        mark_box_corners set to True) and the ConstructionSites module have been
        added to the environment.
    '''

    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        box_xpos = obs['box_xpos']
        boxcorner_pos = obs['box_corner_pos']
        site_pos = obs['construction_site_pos']
        sitecorner_pos = obs['construction_site_corner_pos']

        box_box_dist = np.linalg.norm(
            box_xpos[..., None] - box_xpos.T[None, ...], axis=1)
        box_site_dist = np.linalg.norm(
            box_xpos[..., None] - site_pos.T[None, ...], axis=1)
        boxcorner_sitecorner_dist = (
            np.linalg.norm(boxcorner_pos[..., None] - sitecorner_pos.T[None, ...], axis=1))

        obs.update({'box_box_dist': box_box_dist,
                    'box_site_dist': box_site_dist,
                    'boxcorner_sitecorner_dist': boxcorner_sitecorner_dist})

        return obs


class ConstructionDenseRewardWrapper(gym.Wrapper):
    '''
        Adds a dense reward for placing the boxes at the construction site locations.
        Reward is based on the smoothmin distance between each site and all the boxes.
        Args:
            use_corners (bool): Whether to calculate reward based solely on the distances
                between box centers and site centers, or also based on the distances
                between box corners and site corners.
            alpha (float): Smoothing parameter. Should be nonpositive.
            reward_scale (float): scales the reward by this factor
    '''

    def __init__(self, env, use_corners=False, alpha=-1.5, reward_scale=0.05):
        super().__init__(env)
        assert alpha < 0, 'alpha must be negative for the SmoothMin function to work'
        self.alpha = alpha
        self.reward_scale = reward_scale
        self.use_corners = use_corners

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        box_site_dist = (obs['boxcorner_sitecorner_dist']
                         if self.use_corners
                         else obs['box_site_dist'])
        scaling_factors = np.exp(self.alpha * box_site_dist)
        site_box_smoothmin_dists = (np.sum(box_site_dist * scaling_factors, axis=0) /
                                    np.sum(scaling_factors, axis=0))
        rew -= np.mean(site_box_smoothmin_dists) * self.reward_scale
        return obs, rew, done, info


class ConstructionCompletedRewardWrapper(gym.Wrapper):
    '''
        Adds a sparse reward and ends the episode after all construction sites have been
        'activated' by having a box within a certain distance of them. The reward is based
        on the number of construction sites in the episode.
        Args:
            use_corners (bool): Whether to calculate if construction is finished based
                solely on the distances between box centers and site centers, or also
                based on the distances between box corners and site corners.
            site_activation_radius (float): a site is considered 'activated' if there is
                at least one box within the site activation radius.
            reward_scale (float): scales the reward by this factor
    '''

    def __init__(self, env, use_corners=False, site_activation_radius=0.5, reward_scale=0.1):
        super().__init__(env)
        self.n_sites = self.metadata['curr_n_sites']
        self.n_agents = self.metadata['n_actors']
        self.site_activation_radius = site_activation_radius
        self.reward_scale = reward_scale
        self.use_corners = use_corners
        self.success = False

    def reset(self):
        obs = self.env.reset()
        self.n_sites = self.metadata['curr_n_sites']
        self.n_agents = self.metadata['n_actors']
        self.success = False
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        site_dist_to_closest_box = obs['box_site_dist'].min(axis=0)
        sitecorner_dist_to_closest_boxcorner = obs['boxcorner_sitecorner_dist'].min(
            axis=0)
        activated_sites = site_dist_to_closest_box < self.site_activation_radius
        aligned_corners = sitecorner_dist_to_closest_boxcorner < self.site_activation_radius
        # print(activated_sites)
        # print(np.sum(activated_sites))

        all_sites_activated = np.all(activated_sites)
        all_corners_aligned = np.all(aligned_corners)
        construction_completed = ((all_sites_activated and not self.use_corners) or
                                  (all_sites_activated and all_corners_aligned))
        if self.n_agents > 1:
            activated_sites_num = np.sum(activated_sites)
            rew += activated_sites_num * self.reward_scale

        if construction_completed:
            if self.n_agents == 1:
                rew += self.n_sites * self.reward_scale
            self.success = True
            # done = True
        info['success'] = self.success
        info['activated_sites'] = activated_sites_num

        return obs, rew, done, info


def make_env(args):
    return BlueprintConstructionEnv(args)


def BlueprintConstructionEnv(args, n_substeps=15, horizon=200, deterministic_mode=True,
                             floor_size=4.0, grid_size=30,
                             n_agents=2,
                             n_rooms=2, random_room_number=False, scenario='empty', door_size=2,
                             n_sites=2, n_elongated_sites=0, site_placement='uniform_away_from_walls',
                             reward_infos=[{'type': 'construction_dense'}, {
                                 'type': 'construction_completed'}],
                             n_boxes=4, n_elongated_boxes=0,
                             n_min_boxes=None, box_size=0.5, box_only_z_rot=False,
                             lock_box=True, grab_box=True, grab_selective=False, lock_grab_radius=0.25,
                             lock_type='all_lock_team_specific', grab_exclusive=False,
                             grab_out_of_vision=True, lock_out_of_vision=True,
                             box_floor_friction=0.2, other_friction=0.01, gravity=[0, 0, -50],
                             action_lims=(-0.9, 0.9), polar_obs=True,
                             n_lidar_per_agent=0, visualize_lidar=False, compress_lidar_scale=None,
                             boxid_obs=False, boxsize_obs=False, team_size_obs=False, additional_obs={}):

    scenario = args.scenario_name
    n_agents = args.num_agents
    n_boxes = args.num_boxes
    floor_size = args.floor_size

    grab_radius_multiplier = lock_grab_radius / box_size
    lock_radius_multiplier = lock_grab_radius / box_size

    if type(n_sites) not in [list, np.ndarray]:
        n_sites = [n_sites, n_sites]

    env = Base(n_agents=n_agents, n_substeps=n_substeps, horizon=horizon,
               floor_size=floor_size, grid_size=grid_size,
               action_lims=action_lims, deterministic_mode=deterministic_mode)

    if scenario == 'randomwalls':
        env.add_module(RandomWalls(n_agents=n_agents, grid_size=grid_size, num_rooms=n_rooms,
                                   random_room_number=random_room_number, min_room_size=6,
                                   door_size=door_size, gen_door_obs=False))
    elif scenario == 'empty':
        env.add_module(WallScenarios(n_agents=n_agents, grid_size=grid_size, door_size=door_size,
                                     scenario='empty',
                                     friction=other_friction))
    else:
        raise ValueError(f"Scenario {scenario} not supported.")

    env.add_module(Agents(n_agents,
                          placement_fn=uniform_placement,
                          color=[
                              np.array((66., 235., 244., 255.)) / 255] * n_agents,
                          friction=other_friction,
                          polar_obs=polar_obs))
    if np.max(n_boxes) > 0:
        env.add_module(Boxes(n_boxes=n_boxes, placement_fn=uniform_placement,
                             friction=box_floor_friction, polar_obs=polar_obs,
                             n_elongated_boxes=n_elongated_boxes,
                             boxid_obs=boxid_obs, boxsize_obs=boxsize_obs,
                             box_size=box_size,
                             box_only_z_rot=box_only_z_rot,
                             mark_box_corners=True))
    if n_sites[1] > 0:
        if site_placement == 'center':
            site_placement_fn = center_placement
        elif site_placement == 'uniform':
            site_placement_fn = uniform_placement
        elif site_placement == 'uniform_away_from_walls':
            site_placement_fn = uniform_placement_middle(0.85)
        else:
            raise ValueError(f'Site placement option: {site_placement} not implemented.'
                             ' Please choose from center, uniform and uniform_away_from_walls.')

        env.add_module(ConstructionSites(n_sites, placement_fn=site_placement_fn,
                                         site_size=box_size, site_height=box_size / 2,
                                         n_elongated_sites=n_elongated_sites))
    if n_lidar_per_agent > 0 and visualize_lidar:
        env.add_module(LidarSites(n_agents=n_agents,
                                  n_lidar_per_agent=n_lidar_per_agent))
    if np.max(n_boxes) > 0 and grab_box:
        env.add_module(AgentManipulation())
    if box_floor_friction is not None:
        env.add_module(FloorAttributes(friction=box_floor_friction))
    env.add_module(WorldConstants(gravity=gravity))
    env.reset()
    keys_self = ['agent_qpos_qvel', 'current_step']
    keys_mask_self = ['mask_aa_obs']
    keys_external = ['agent_qpos_qvel', 'construction_site_obs']
    keys_copy = ['you_lock', 'team_lock', 'ramp_you_lock', 'ramp_team_lock']
    keys_mask_external = []

    env = AddConstantObservationsWrapper(env, new_obs=additional_obs)
    keys_external += list(additional_obs)
    keys_mask_external += [ob for ob in additional_obs if 'mask' in ob]

    env = SplitMultiAgentActions(env)
    if team_size_obs:
        keys_self += ['team_size']
    env = TeamMembership(env, np.zeros((n_agents,)))
    env = AgentAgentObsMask2D(env)
    env = DiscretizeActionWrapper(env, 'action_movement')
    if np.max(n_boxes) > 0:
        env = AgentGeomObsMask2D(env, pos_obs_key='box_pos', mask_obs_key='mask_ab_obs',
                                 geom_idxs_obs_key='box_geom_idxs')
        keys_external += ['mask_ab_obs', 'box_obs']
        keys_mask_external.append('mask_ab_obs')
    if lock_box and np.max(n_boxes) > 0:
        agent_allowed_to_lock_keys = None if lock_out_of_vision else [
            "mask_ab_obs"]
        env = LockObjWrapper(env, body_names=[f'moveable_box{i}' for i in range(n_boxes)],
                             agent_idx_allowed_to_lock=np.arange(n_agents),
                             lock_type=lock_type,
                             radius_multiplier=lock_radius_multiplier,
                             obj_in_game_metadata_keys=["curr_n_boxes"],
                             agent_allowed_to_lock_keys=agent_allowed_to_lock_keys)
    if grab_box and np.max(n_boxes) > 0:
        env = GrabObjWrapper(env, [f'moveable_box{i}' for i in range(n_boxes)],
                             radius_multiplier=grab_radius_multiplier,
                             grab_exclusive=grab_exclusive,
                             obj_in_game_metadata_keys=['curr_n_boxes'])

    if n_lidar_per_agent > 0:
        env = Lidar(env, n_lidar_per_agent=n_lidar_per_agent, visualize_lidar=visualize_lidar,
                    compress_lidar_scale=compress_lidar_scale)
        keys_copy += ['lidar']
        keys_external += ['lidar']

    env = ConstructionDistancesWrapper(env)
    env = NumpyArrayRewardWrapper(env)

    reward_wrappers = {
        'construction_dense': ConstructionDenseRewardWrapper,
        'construction_completed': ConstructionCompletedRewardWrapper,
    }

    for rew_info in reward_infos:
        rew_type = rew_info['type']
        del rew_info['type']
        env = reward_wrappers[rew_type](env, **rew_info)
    env = TimeWrapper(env, horizon)
    env = SplitObservations(
        env, keys_self + keys_mask_self, keys_copy=keys_copy)
    if n_agents == 1:
        env = SpoofEntityWrapper(env, 2, ['agent_qpos_qvel'], ['mask_aa_obs'])
    env = SpoofEntityWrapper(env, n_boxes,
                             ['box_obs', 'you_lock', 'team_lock', 'obj_lock'],
                             ['mask_ab_obs'])
    env = SpoofEntityWrapper(
        env, n_sites[1], ['construction_site_obs'], ['mask_acs_obs'])
    keys_mask_external += ['mask_ab_obs_spoof', 'mask_acs_obs_spoof']
    env = LockAllWrapper(env, remove_object_specific_lock=True)
    if not grab_out_of_vision and grab_box:
        # Can only pull if in vision
        env = MaskActionWrapper(env, 'action_pull', ['mask_ab_obs'])
    if not grab_selective and grab_box:
        env = GrabClosestWrapper(env)
    env = DiscardMujocoExceptionEpisodes(env, n_agents)
    env = ConcatenateObsWrapper(env, {'agent_qpos_qvel': ['agent_qpos_qvel'],
                                      'box_obs': ['box_obs', 'you_lock', 'team_lock', 'obj_lock']})
    env = SelectKeysWrapper(env, keys_self=keys_self,
                            keys_external=keys_external,
                            keys_mask=keys_mask_self + keys_mask_external,
                            flatten=False)
    return env
