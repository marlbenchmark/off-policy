"""
Environments for training vehicles to reduce capacity drops in a bottleneck.
This environment was used in:
TODO(ak): add paper after it has been published.
"""

from collections import defaultdict
from copy import deepcopy, copy

from gym.spaces.box import Box
from gym.spaces.dict_space import Dict
from gym.spaces.discrete import Discrete
from gym.spaces.tuple_space import Tuple
import numpy as np

from flow.controllers.velocity_controllers import FakeDecentralizedALINEAController, IDMController
from flow.controllers.rlcontroller import RLController
from flow.controllers.routing_controllers import ContinuousRouter
from flow.controllers.lane_change_controllers import SimLaneChangeController
from flow.multiagent_envs.multiagent_env import MultiEnv
from flow.envs.bottleneck_env import DesiredVelocityEnv
from flow.core.params import InFlows, NetParams, VehicleParams, \
    SumoCarFollowingParams, SumoLaneChangeParams

MAX_LANES = 4  # base number of largest number of lanes in the network
EDGE_LIST = ["1", "2", "3", "4", "5"]  # Edge 1 is before the toll booth

# Keys for RL experiments
ADDITIONAL_RL_ENV_PARAMS = {
    # velocity to use in reward functions
    "target_velocity": 30,
    # if an RL vehicle exits, place it back at the front
    "add_rl_if_exit": True,
    # whether communication between vehicles is on
    "communicate": False,
    # whether the observation space is aggregate counts or local observations
    "centralized_obs": False,
    # whether to add aggregate info (speed, number of congested vehicles) about some of the edges
    "aggregate_info": False,
    # whether to add an additional penalty for allowing too many vehicles into the bottleneck
    "congest_penalty": False,
    "av_frac": 0.1,
    # Above this number, the congestion penalty starts to kick in
    "congest_penalty_start": 30,
    # What lane changing mode the human drivers should have
    "lc_mode": 0,
    # how many seconds the outflow reward should sample over
    "num_sample_seconds": 20,
    # whether the reward function should be over speed
    "speed_reward": False
}


class MultiBottleneckEnv(MultiEnv, DesiredVelocityEnv):
    """Environment used to train decentralized vehicles to effectively pass
       through a bottleneck by specifying the velocity that RL vehicles
       should attempt to travel in certain regions of space
       States
           An observation is the speed and velocity of leading and
           following vehicles
       Actions
           The action space consist of a dict of accelerations for each
           autonomous vehicle
       Rewards
           The reward is a dict consisting of the normalized
           outflow of the bottleneck
    """
    def __init__(self, env_params, sim_params, scenario, simulator='traci'):
        super().__init__(env_params, sim_params, scenario, simulator)
        self.simple_env = env_params.additional_params.get("simple_env")
        self.super_simple_env = env_params.additional_params.get("super_simple_env")

        self.rew_n_crit = env_params.additional_params.get("rew_n_crit")

        self.curr_iter = 0
        self.rew_history = 0
        self.exit_counter = 0
        self.last_exit_counter = 0
        self.total_exit_counter = 0
        self.total_reward = defaultdict(float)
        self.total_reward_sum = 0
        self.rl_ids_reroute = []
        self.num_curr_iters = env_params.additional_params["num_curr_iters"]
        self.curriculum = env_params.additional_params["curriculum"]
        self.min_horizon = env_params.additional_params["min_horizon"]
        self.max_horizon = env_params.additional_params["horizon"]
        self.reroute_on_exit = env_params.additional_params["reroute_on_exit"]
        if self.curriculum:
            self.env_params.horizon = self.min_horizon
        self.warmup_done = False


    def increase_curr_iter(self):
        self.curr_iter += 1
        curriculum_scaling = min(self.curr_iter / self.num_curr_iters, 1.0)
        self.env_params.horizon = self.min_horizon + curriculum_scaling * (self.max_horizon - self.min_horizon)

        print('YO THE HORIZON IS', self.env_params.horizon)

    @property
    def observation_space(self):
        """See class definition."""

        # normalized speed and velocity of leading and following vehicles
        # additionally, for each lane leader we add if it is
        # an RL vehicle or not
        # the position edge id, and lane of the vehicle
        # additionally, we add the time-step (for the baseline)
        # the outflow over the last 10 seconds
        # the number of vehicles in the congested section
        # the average velocity on each edge 3,4,5
        add_params = self.env_params.additional_params
        num_obs = 0
        if add_params['centralized_obs']:
            # density and velocity for rl and non-rl vehicles per segment
            # Last element is the outflow and inflow and the vehicles speed and headway, edge id, lane, edge pos
            for segment in self.obs_segments:
                num_obs += 4 * segment[1] * \
                           self.k.scenario.num_lanes(segment[0])
            num_obs += 7
        elif self.simple_env:
            # abs_position duration, time since stopped, number of vehicles in the bottleneck, speed, lead speed, headway
            num_obs = 7
        elif self.super_simple_env:
            num_obs = 3
        else:
            if self.env_params.additional_params['communicate']:
                # eight possible signals if above
                if self.env_params.additional_params.get('aggregate_info'):
                    num_obs = 6 * MAX_LANES * self.scaling + 22
                else:
                    num_obs = 6 * MAX_LANES * self.scaling + 16
            else:
                if self.env_params.additional_params.get('aggregate_info'):
                    num_obs = 6 * MAX_LANES * self.scaling + 14
                else:
                    num_obs = 6 * MAX_LANES * self.scaling + 8

        # TODO(@evinitsky) eventually remove the get once backwards compatibility is no longer needed
        if self.env_params.additional_params.get('keep_past_actions', False):
            self.num_past_actions = 100
            num_obs += self.num_past_actions
        return Box(low=-10.0, high=10.0,
                   shape=(num_obs,),
                   dtype=np.float32)

    @property
    def action_space(self):
        """See class definition."""
        if self.env_params.additional_params['communicate']:
            accel = Box(
                low=-4.0 / 8.0, high=2.6 / 8.0, shape=(1,), dtype=np.float32)
            communicate = Discrete(2)
            return Tuple((accel, communicate))
        else:
            return Box(
                low=-4.0 / 8.0, high=2.6 / 8.0, shape=(1,), dtype=np.float32)

    def init_decentral_controller(self, rl_id):
        return FakeDecentralizedALINEAController(rl_id, stop_edge="2", stop_pos=310,
                                                       additional_env_params=self.env_params.additional_params,
                                                       car_following_params=SumoCarFollowingParams())

    def update_curr_rl_vehicles(self):
        self.curr_rl_vehicles.update({rl_id: {'controller': self.init_decentral_controller(rl_id),
                                              'time_since_stopped': 0.0,
                                              'is_stopped': False,}
                                              for rl_id in self.k.vehicle.get_rl_ids()
                                      if rl_id not in self.curr_rl_vehicles.keys()})

    def get_state(self, rl_actions=None):
        """See class definition."""
        # action space is speed and velocity of leading and following
        # vehicles for all of the avs
        self.update_curr_rl_vehicles()
        add_params = self.env_params.additional_params
        if self.reroute_on_exit:
            rl_ids = self.rl_ids_reroute
        else:
            rl_ids = [veh_id for veh_id in self.k.vehicle.get_rl_ids() if self.k.vehicle.get_edge(veh_id) in ['1', '2', '3', '4', '5']]

        if add_params['centralized_obs']:
            state = self.get_centralized_state()
            veh_info = {rl_id: np.concatenate((self.veh_statistics(rl_id), state)) for rl_id in rl_ids}
        elif self.simple_env:
            self.update_curr_rl_vehicles()
            veh_info = {}
            if self.reroute_on_exit:
                rl_ids = self.rl_ids_reroute
            else:
                rl_ids = [veh_id for veh_id in self.k.vehicle.get_rl_ids() if self.k.vehicle.get_edge(veh_id) in ['1', '2', '3', '4', '5']]
            congest_number = len(self.k.vehicle.get_ids_by_edge('4')) / 50
            for rl_id in rl_ids:
                # if rl_id out of network
                if rl_id not in self.k.vehicle.get_rl_ids():
                    veh_info[rl_id] = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
                else:
                    controller = self.curr_rl_vehicles[rl_id]['controller']
                    if self.k.vehicle.get_speed(rl_id) <= 0.2:
                        self.curr_rl_vehicles[rl_id]['time_since_stopped'] += 1.0
                    else:
                        self.curr_rl_vehicles[rl_id]['time_since_stopped'] = 0.0

                    duration = controller.duration
                    abs_position = self.k.vehicle.get_position(rl_id)
                    # if rl_actions and rl_id in rl_actions.keys():
                    #     print('RL ', rl_actions[rl_id])
                    #     print('Expert ', accel)
                    speed = self.k.vehicle.get_speed(rl_id)
                    lead_id = self.k.vehicle.get_leader(rl_id)
                    lead_speed = self.k.vehicle.get_speed(lead_id)
                    if lead_speed == -1001:
                        lead_speed = -10
                    headway = self.k.vehicle.get_headway(rl_id)
                    veh_info[rl_id] = np.array([abs_position / 1000.0,
                                                            self.curr_rl_vehicles[rl_id][
                                                                'time_since_stopped'] / self.env_params.horizon,
                                                            duration / 100.0,
                                                            congest_number,
                                                            speed / 50.0,
                                                            lead_speed / 50.0,
                                                            headway / 1000.0])
                    veh_info[rl_id] = np.clip(veh_info[rl_id], -10, 10)

        elif self.super_simple_env:
            self.update_curr_rl_vehicles()
            veh_info = {}
            if self.reroute_on_exit:
                rl_ids = self.rl_ids_reroute
            else:
                rl_ids = [veh_id for veh_id in self.k.vehicle.get_rl_ids() if self.k.vehicle.get_edge(veh_id) in ['1', '2', '3', '4', '5']]
            congest_number = len(self.k.vehicle.get_ids_by_edge('4')) / 50
            for rl_id in rl_ids:
                abs_position = self.k.vehicle.get_position(rl_id)

                if self.k.vehicle.get_speed(rl_id) <= 0.2:
                    self.curr_rl_vehicles[rl_id]['time_since_stopped'] += 1.0
                else:
                    self.curr_rl_vehicles[rl_id]['time_since_stopped'] = 0.0
                veh_info[rl_id] = np.array([abs_position / 1000.0,
                                            self.curr_rl_vehicles[rl_id][
                                                'time_since_stopped'] / self.env_params.horizon,
                                            congest_number,
                                            ])
        else:
            veh_info = {}
            for rl_id in rl_ids:
                if rl_id not in self.k.vehicle.get_rl_ids():
                    if self.env_params.additional_params['communicate']:
                        # eight possible signals if above
                        if self.env_params.additional_params.get('aggregate_info'):
                            num_obs = 6 * MAX_LANES * self.scaling + 22
                        else:
                            num_obs = 6 * MAX_LANES * self.scaling + 16
                    else:
                        if self.env_params.additional_params.get('aggregate_info'):
                            num_obs = 6 * MAX_LANES * self.scaling + 14
                        else:
                            num_obs = 6 * MAX_LANES * self.scaling + 8
                    veh_info[rl_id] = np.array([-1.0] * num_obs)
                else:
                    if self.env_params.additional_params.get('communicate', False):
                        veh_info[rl_id] = np.concatenate((self.veh_statistics(rl_id),
                                                        self.state_util(rl_id),
                                                        self.get_signal(rl_id,
                                                                        rl_actions)
                                                        )
                                                    )
                    else:
                        veh_info[rl_id] = np.concatenate((self.veh_statistics(rl_id),
                                                            self.state_util(rl_id),
                                                        ))

                    if self.env_params.additional_params.get('aggregate_info'):
                        agg_statistics = self.aggregate_statistics()
                        veh_info[rl_id] = np.concatenate((veh_info[rl_id], agg_statistics))


        if self.env_params.additional_params.get('keep_past_actions', False):
            # update the actions history with the most recent actions
            for rl_id in self.k.vehicle.get_rl_ids():
                agent_past_dict, num_steps = self.past_actions_dict[rl_id]
                if rl_actions and rl_id in rl_actions.keys():
                    agent_past_dict[num_steps] = rl_actions[rl_id] / self.action_space.high
                num_steps += 1
                num_steps %= self.num_past_actions
                self.past_actions_dict[rl_id] = [agent_past_dict, num_steps]
            actions_history = {rl_id: self.past_actions_dict[rl_id][0] for rl_id in self.k.vehicle.get_rl_ids()}
            veh_info = {rl_id: np.concatenate((veh_info[rl_id], actions_history[rl_id])) for
                        rl_id in rl_ids}

        # Go through the human drivers and add zeros if the vehicles have left as a final observation
        # if int(self.time_counter / self.env_params.sims_per_step) == self.env_params.horizon:
        #     if isinstance(self.observation_space, Box):
        #         left_vehicles_dict = {veh_id: np.zeros(self.observation_space.shape[0]) for veh_id
        #                               in self.left_av_time_dict.keys()}
        #     elif isinstance(self.observation_space, Dict):
        #         num_obs = 0
        #         for space in self.observation_space.spaces.values():
        #             num_obs += space.shape[0]
        #         left_vehicles_dict = {veh_id: np.zeros(num_obs) for veh_id
        #                               in self.left_av_time_dict.keys()}
        #     veh_info.update(left_vehicles_dict)

        # if isinstance(self.observation_space, Box):
        #     veh_info = {key: np.clip(value, a_min=self.observation_space.low[0:value.shape[0]],
        #                              a_max=self.observation_space.high[0:value.shape[0]]) for
        #                 key, value in veh_info.items()}
        # elif isinstance(self.observation_space, Dict):
        #     # TODO(@evinitsky) this is bad subclassing and will break if the obs space isn't uniform
        #     veh_info = {key: np.clip(value, a_min=[self.observation_space.spaces['a_obs'].low[0]] * value.shape[0],
        #                              a_max=[self.observation_space.spaces['a_obs'].high[0]] * value.shape[0]) for
        #                 key, value in veh_info.items()}

        return veh_info

    def get_centralized_state(self):
        """See class definition."""
        # action space is number of vehicles in each segment in each lane,
        # number of rl vehicles in each segment in each lane
        # mean speed in each segment, and mean rl speed in each
        # segment in each lane
        num_vehicles_list = []
        num_rl_vehicles_list = []
        vehicle_speeds_list = []
        rl_speeds_list = []
        NUM_VEHICLE_NORM = 20
        for i, edge in enumerate(EDGE_LIST):
            num_lanes = self.k.scenario.num_lanes(edge)
            num_vehicles = np.zeros((self.num_obs_segments[i], num_lanes))
            num_rl_vehicles = np.zeros((self.num_obs_segments[i], num_lanes))
            vehicle_speeds = np.zeros((self.num_obs_segments[i], num_lanes))
            rl_vehicle_speeds = np.zeros((self.num_obs_segments[i], num_lanes))
            ids = self.k.vehicle.get_ids_by_edge(edge)
            lane_list = self.k.vehicle.get_lane(ids)
            pos_list = self.k.vehicle.get_position(ids)
            for i, id in enumerate(ids):
                segment = np.searchsorted(self.obs_slices[edge],
                                          pos_list[i]) - 1
                if id in self.k.vehicle.get_rl_ids():
                    rl_vehicle_speeds[segment, lane_list[i]] \
                        += self.k.vehicle.get_speed(id)
                    num_rl_vehicles[segment, lane_list[i]] += 1
                else:
                    vehicle_speeds[segment, lane_list[i]] \
                        += self.k.vehicle.get_speed(id)
                    num_vehicles[segment, lane_list[i]] += 1

            # normalize

            num_vehicles /= NUM_VEHICLE_NORM
            num_rl_vehicles /= NUM_VEHICLE_NORM
            num_vehicles_list += num_vehicles.flatten().tolist()
            num_rl_vehicles_list += num_rl_vehicles.flatten().tolist()
            vehicle_speeds_list += vehicle_speeds.flatten().tolist()
            rl_speeds_list += rl_vehicle_speeds.flatten().tolist()

        unnorm_veh_list = np.asarray(num_vehicles_list) * \
                          NUM_VEHICLE_NORM
        unnorm_rl_list = np.asarray(num_rl_vehicles_list) * \
                         NUM_VEHICLE_NORM
        # compute the mean speed if the speed isn't zero
        num_rl = len(num_rl_vehicles_list)
        num_veh = len(num_vehicles_list)
        mean_speed = np.nan_to_num([
            vehicle_speeds_list[i] / unnorm_veh_list[i]
            if int(unnorm_veh_list[i]) else 0 for i in range(num_veh)
        ])
        mean_speed_norm = mean_speed / 50
        mean_rl_speed = np.nan_to_num([
            rl_speeds_list[i] / unnorm_rl_list[i]
            if int(unnorm_rl_list[i]) else 0 for i in range(num_rl)
        ]) / 50
        outflow = np.asarray(
            self.k.vehicle.get_outflow_rate(20 * self.sim_step) / 2000.0)
        temp = np.concatenate((num_vehicles_list, num_rl_vehicles_list,
                               mean_speed_norm, mean_rl_speed, [outflow],
                               [self.inflow]))
        if np.any(temp < 0):
            import ipdb; ipdb.set_trace()
        return np.concatenate((num_vehicles_list, num_rl_vehicles_list,
                               mean_speed_norm, mean_rl_speed, [outflow],
                               [self.inflow]))

    def _apply_rl_actions(self, rl_actions):
        """
        Per-vehicle accelerations
        """
        if rl_actions:
            accel_list = []
            rl_ids = []
            for rl_id, action in rl_actions.items():
                if self.env_params.additional_params.get('communicate', False):
                    accel = np.concatenate([action[0] for action in action])
                else:
                    accel = [val * 8.0 for val in action]
                if self.k.vehicle.get_edge(rl_id) in ['3']:
                    accel_list.extend(accel)
                    rl_ids.append(rl_id)
            self.k.vehicle.apply_acceleration(rl_ids, accel_list)

    def compute_reward(self, rl_actions, **kwargs):
        """Outflow rate over last ten seconds normalized to max of 1."""
        if self.env_params.evaluate:
            if int(self.time_counter/self.env_params.sims_per_step) == self.env_params.horizon:
                reward = self.k.vehicle.get_outflow_rate(500)
                return reward
            else:
                return 0

        if self.reroute_on_exit:
            rl_ids = self.rl_ids_reroute
        else:
            rl_ids = [veh_id for veh_id in self.k.vehicle.get_rl_ids() if self.k.vehicle.get_edge(veh_id) in ['1', '2', '3', '4', '5']]

        if self.rew_n_crit > 0:
            num_vehs = len(self.k.vehicle.get_ids_by_edge('4'))
            reward = (self.rew_n_crit - np.abs(self.rew_n_crit - num_vehs)) / 100
        else:
            # only if reroute_on_exit is on
            reward = self.last_exit_counter / 50.0
            self.last_exit_counter = 0

            for rl_id in rl_ids:
                self.total_reward[rl_id] += reward
            self.total_reward_sum += reward
            # print('total_reward=', self.total_reward)
            # print('total_reward_sum=', self.total_reward_sum)
            # reward = len(self.k.vehicle.get_ids_by_edge('5')) / 5.0

        reward_dict = {rl_id: reward for rl_id in rl_ids}
        self.rew_history += reward

        return reward_dict

    def reset(self, new_inflow_rate=None):
        self.curr_rl_vehicles = {}
        self.exit_counter = 0
        # print('THE TOTAL REWARD FOR THIS ROUND WAS ', self.rew_history)
        try:
            # print('THE TOTAL OUTFLOW FOR THIS ROUND WAS ', self.k.vehicle.get_outflow_rate(10000))
            pass
        except:
            pass
        self.rew_history = 0
        self.update_curr_rl_vehicles()

        # dict tracking past actions
        if self.env_params.additional_params.get('keep_past_actions', False):
            self.past_actions_dict = defaultdict(lambda: [np.zeros(self.num_past_actions), 0])

        add_params = self.env_params.additional_params
        if True:#add_params.get("reset_inflow") and self.sim_params.restart_instance:
            inflow_range = add_params.get("inflow_range")
            if new_inflow_rate:
                flow_rate = new_inflow_rate
            else:
                flow_rate = np.random.uniform(
                    min(inflow_range), max(inflow_range)) * self.scaling
            self.inflow = flow_rate
            #print('THE FLOW RATE IS: ', flow_rate)
            for _ in range(100):
                try:
                    vehicles = VehicleParams()
                    if not np.isclose(add_params.get("av_frac"), 1):
                        vehicles.add(
                            veh_id="human",
                            lane_change_controller=(SimLaneChangeController, {}),
                            routing_controller=(ContinuousRouter, {}),
                            car_following_params=SumoCarFollowingParams(
                                speed_mode=31,
                            ),
                            lane_change_params=SumoLaneChangeParams(
                                lane_change_mode=add_params.get("lc_mode"),
                            ),
                            num_vehicles=1)
                        vehicles.add(
                            veh_id="av",
                            acceleration_controller=(RLController, {}),
                            lane_change_controller=(SimLaneChangeController, {}),
                            routing_controller=(ContinuousRouter, {}),
                            car_following_params=SumoCarFollowingParams(
                                speed_mode=31,
                            ),
                            lane_change_params=SumoLaneChangeParams(
                                lane_change_mode=0,
                            ),
                            num_vehicles=1)
                    else:
                        vehicles.add(
                            veh_id="av",
                            acceleration_controller=(RLController, {}),
                            lane_change_controller=(SimLaneChangeController, {}),
                            routing_controller=(ContinuousRouter, {}),
                            car_following_params=SumoCarFollowingParams(
                                speed_mode=31,
                            ),
                            lane_change_params=SumoLaneChangeParams(
                                lane_change_mode=add_params.get("lc_mode"),
                            ),
                            num_vehicles=1)

                    inflow = InFlows()
                    if not np.isclose(add_params.get("av_frac"), 1.0):
                        inflow.add(
                            veh_type="av",
                            edge="1",
                            vehs_per_hour=flow_rate * add_params.get("av_frac"),
                            departLane="random",
                            departSpeed=23.0)
                        inflow.add(
                            veh_type="human",
                            edge="1",
                            vehs_per_hour=flow_rate * (1 - add_params.get("av_frac")),
                            departLane="random",
                            departSpeed=23.0)
                        # print('INFLOWS')
                        # print(f'av/h = {flow_rate * add_params.get("av_frac")}')
                        # print(f'veh/h = {flow_rate * (1 - add_params.get("av_frac"))}')
                    else:
                        inflow.add(
                            veh_type="av",
                            edge="1",
                            vehs_per_hour=flow_rate,
                            departLane="random",
                            departSpeed=23.0)

                    additional_net_params = {
                        "scaling": self.scaling,
                        "speed_limit": self.scenario.net_params.
                            additional_params['speed_limit']
                    }
                    net_params = NetParams(
                        inflows=inflow,
                        no_internal_links=False,
                        additional_params=additional_net_params)

                    self.scenario = self.scenario.__class__(
                        self.scenario.orig_name, vehicles,
                        net_params, self.scenario.initial_config)
                    self.k.vehicle = deepcopy(self.initial_vehicles)
                    self.k.vehicle.kernel_api = self.k.kernel_api
                    self.k.vehicle.master_kernel = self.k

                    # restart the sumo instance
                    self.restart_simulation(
                        sim_params=self.sim_params,
                        render=self.sim_params.render)

                    observation = super().reset()

                    # reset the timer to zero
                    self.time_counter = 0

                    return observation

                except Exception as e:
                    print('error on reset ', e)

        # perform the generic reset function
        observation = super().reset()

        # reset the timer to zero
        self.time_counter = 0

        return observation

    def veh_statistics(self, rl_id):
        '''Returns speed and edge information about the vehicle itself'''
        speed = self.k.vehicle.get_speed(rl_id) / 100.0
        edge = self.k.vehicle.get_edge(rl_id)
        lane = (self.k.vehicle.get_lane(rl_id) + 1) / 10.0
        headway = self.k.vehicle.get_headway(rl_id) / 2000.0
        position = self.k.vehicle.get_position(rl_id) / 1000.0
        if edge:
            if edge[0] != ':':
                edge_id = int(self.k.vehicle.get_edge(rl_id)) / 10.0
            else:
                edge_id = - 1 / 10.0
        else:
            edge_id = - 1 / 10.0
        # an absolute position used to make it easier to sort the vehicles for the centralized value function
        absolute_pos = self.k.vehicle.get_x_by_id(rl_id) / 1000.0
        return np.array([absolute_pos, speed, edge_id, lane, headway, position,
                         self.curr_rl_vehicles[rl_id]['time_since_stopped'] / self.env_params.horizon,
                         self.time_counter / (self.env_params.sims_per_step * self.env_params.horizon)]).clip(min=-1.0)

    def state_util(self, rl_id):
        ''' Returns an array of headway, tailway, leader speed, follower speed
            a 1 if leader is rl 0 otherwise, a 1 if follower is rl 0
            otherwise
            If there are fewer than self.scaling*MAX_LANES the extra
            entries are filled with -1 to disambiguate from zeros
        '''
        veh = self.k.vehicle
        lane_headways = veh.get_lane_headways(rl_id).copy()
        lane_tailways = veh.get_lane_tailways(rl_id).copy()
        lane_leader_speed = veh.get_lane_leaders_speed(rl_id).copy()
        lane_follower_speed = veh.get_lane_followers_speed(rl_id).copy()
        leader_ids = veh.get_lane_leaders(rl_id).copy()
        follower_ids = veh.get_lane_followers(rl_id).copy()
        rl_ids = self.k.vehicle.get_rl_ids()
        is_leader_rl = [1 if l_id in rl_ids else 0 for l_id in leader_ids]
        is_follow_rl = [1 if f_id in rl_ids else 0 for f_id in follower_ids]
        diff = self.scaling * MAX_LANES - len(is_leader_rl)
        if diff > 0:
            # the minus 1 disambiguates missing cars from missing lanes
            lane_headways += diff * [-1]
            lane_tailways += diff * [-1]
            lane_leader_speed += diff * [-1]
            lane_follower_speed += diff * [-1]
            is_leader_rl += diff * [-1]
            is_follow_rl += diff * [-1]
        lane_headways = np.asarray(lane_headways) / 1000
        lane_tailways = np.asarray(lane_tailways) / 1000
        lane_leader_speed = np.asarray(lane_leader_speed) / 100
        lane_follower_speed = np.asarray(lane_follower_speed) / 100
        return np.concatenate((lane_headways, lane_tailways, lane_leader_speed,
                               lane_follower_speed, is_leader_rl,
                               is_follow_rl)).clip(min=-1.0)

    def aggregate_statistics(self):
        ''' Returns the time-step, outflow over the last 10 seconds,
            number of vehicles in the congested area
            and average velocity of segments 3,4,5,6
        '''
        time_step = self.time_counter / (self.env_params.horizon * self.env_params.sims_per_step)
        outflow = self.k.vehicle.get_outflow_rate(10) / 3600
        valid_edges = ['3', '4', '5']
        congest_number = len(self.k.vehicle.get_ids_by_edge('4')) / 50
        avg_speeds = np.zeros(len(valid_edges))
        for i, edge in enumerate(valid_edges):
            edge_veh = self.k.vehicle.get_ids_by_edge(edge)
            if len(edge_veh) > 0:
                veh = self.k.vehicle
                avg_speeds[i] = np.mean(veh.get_speed(edge_veh)) / 100.0
        return np.concatenate(([time_step], [outflow],
                               [congest_number], avg_speeds))

    def get_signal(self, rl_id, rl_actions):
        ''' Returns the communication signals that should be
            pass to the autonomous vehicles
        '''
        lead_ids = self.k.vehicle.get_lane_leaders(rl_id)
        follow_ids = self.k.vehicle.get_lane_followers(rl_id)
        comm_ids = lead_ids + follow_ids
        if rl_actions:
            signals = [rl_actions[av_id][1] / 4.0 if av_id in
                                                     rl_actions.keys() else -1 / 4.0 for av_id in comm_ids]
            if len(signals) < 8:
                # the -2 disambiguates missing cars from missing lanes
                signals += (8 - len(signals)) * [-2 / 4.0]
            return signals
        else:
            return [-1 / 4.0 for _ in range(8)]

    def additional_command(self):
        super().additional_command()

        # print(f'step={self.step_counter}, warmup_step={self.env_params.warmup_steps}')
        if self.reroute_on_exit and self.warmup_done and not self.env_params.evaluate:
            if len(self.rl_ids_reroute) == 0:
                self.rl_ids_reroute = list(self.k.vehicle.get_rl_ids())
            
            # cc = 0
            # for vid in self.rl_ids_reroute:
            #     if vid in self.k.vehicle.get_rl_ids():
            #         cc += 1
            # print(f'{cc}/{len(self.rl_ids_reroute)}')



            veh_ids = list(self.k.vehicle.get_ids())
            edges = self.k.vehicle.get_edge(veh_ids)
            for veh_id, edge in zip(veh_ids, edges):
                if edge == "":
                    continue
                if edge[0] == ":":  # center edge
                    continue

                if edge == '5':
                    # only count exited vehicles during the last 500s
                    total_time_step = self.env_params.horizon * self.env_params.sims_per_step
                    if self.time_counter > total_time_step - 500 / self.sim_step:
                        self.exit_counter += 1
                        # print('exit_counter/500*3600=', self.exit_counter*3.6)
                    self.last_exit_counter += 1
                    self.total_exit_counter += 1
                    # print(self.total_exit_counter)
                    type_id = self.k.vehicle.get_type(veh_id)
                    # remove the vehicle
                    self.k.vehicle.remove(veh_id)
                    lane = np.random.randint(low=0, high=MAX_LANES * self.scaling)
                    # reintroduce it at the start of the network
                    self.k.vehicle.add(
                        veh_id=veh_id,
                        edge='2',
                        type_id=str(type_id),
                        lane=str(lane),
                        pos="0",
                        speed="23.0")

            departed_ids = list(self.k.vehicle.get_departed_ids())
            # if len(departed_ids) > 0:
            #     for veh_id in departed_ids:
            #         if veh_id not in self.observed_cars:
            #             self.k.vehicle.remove(veh_id)
            for veh_id in departed_ids:
                if self.k.vehicle.get_edge(veh_id) == '1':
                    self.k.vehicle.remove(veh_id)


class MultiBottleneckImitationEnv(MultiBottleneckEnv):
    """MultiBottleneckEnv but we return as our obs dict that also contains the actions of a queried expert"""

    def __init__(self, env_params, sim_params, scenario, simulator='traci'):
        super().__init__(env_params, sim_params, scenario, simulator)
        self.iter_num = 0
        self.num_imitation_iters = env_params.additional_params.get("num_imitation_iters")
        self.simple_env = env_params.additional_params.get("simple_env")

    def set_iteration_num(self, iter_num):
        self.iter_num = iter_num

    @property
    def observation_space(self):
        if self.simple_env:
            # abs_position duration, time since stopped, number of vehicles in the bottleneck, speed, lead speed, headway
            new_obs = Box(low=-10.0, high=10.0, shape=(7,), dtype=np.float32)
        else:
            obs = super().observation_space
            # Extra keys "time since stop", duration
            new_obs = Box(low=-10.0, high=10.0, shape=(obs.shape[0] + 2,), dtype=np.float32)
            # new_obs = Box(low=-3.0, high=3.0, shape=(obs.shape[0],), dtype=np.float32)
        return Dict({"a_obs": new_obs, "expert_action": self.action_space})

    def reset(self, new_inflow_rate=None):

        self.curr_rl_vehicles = {}
        self.update_curr_rl_vehicles()

        state_dict = super().reset(new_inflow_rate)
        return state_dict

    def get_state(self, rl_actions=None):
        # iterate through the RL vehicles and find what the other agent would have done
        self.update_curr_rl_vehicles()
        if self.simple_env:
            state_dict = {}
            rl_ids = [veh_id for veh_id in self.k.vehicle.get_rl_ids() if self.k.vehicle.get_edge(veh_id) in ['1', '2', '3', '4', '5']]
            congest_number = len(self.k.vehicle.get_ids_by_edge('4')) / 50
            for rl_id in rl_ids:
                controller = self.curr_rl_vehicles[rl_id]['controller']
                if self.k.vehicle.get_speed(rl_id) <= 0.2:
                    self.curr_rl_vehicles[rl_id]['time_since_stopped'] += 1.0
                else:
                    self.curr_rl_vehicles[rl_id]['time_since_stopped'] = 0.0

                accel = controller.get_accel(self)

                if accel is None:
                    accel = -np.abs(self.action_space.low[0])
                duration = controller.duration
                abs_position = self.k.vehicle.get_position(rl_id)
                # if rl_actions and rl_id in rl_actions.keys():
                #     print('RL ', rl_actions[rl_id])
                #     print('Expert ', accel)
                speed = self.k.vehicle.get_speed(rl_id)
                lead_id = self.k.vehicle.get_leader(rl_id)
                lead_speed = self.k.vehicle.get_speed(lead_id)
                if lead_speed == -1001:
                    lead_speed = -10
                headway = self.k.vehicle.get_headway(rl_id)
                state_dict[rl_id] = {"a_obs": np.array([abs_position / 1000.0,
                                                      self.curr_rl_vehicles[rl_id]['time_since_stopped'] / self.env_params.horizon,
                                                      duration / 100.0,
                                                      congest_number,
                                                      speed / 50.0,
                                                      lead_speed / 50.0,
                                                       headway / 1000.0]),
                                       "expert_action": np.array([np.clip(accel, a_min=self.action_space.low[0],
                                                                          a_max=self.action_space.high[0])])}

        else:
            state_dict = super().get_state(rl_actions)

            for key, value in state_dict.items():
                # this could be the fake final state for vehicles that have left the system
                if key in self.k.vehicle.get_ids():
                    controller = self.curr_rl_vehicles[key]['controller']
                    if self.k.vehicle.get_speed(key) <= 0.2:
                        self.curr_rl_vehicles[key]['time_since_stopped'] += 1.0
                    else:
                        self.curr_rl_vehicles[key]['time_since_stopped'] = 0.0

                    accel = controller.get_accel(self)
                    if accel is None:
                        accel = -np.abs(self.action_space.low[0])

                    duration = controller.duration

                    state_dict[key] = {"a_obs": np.concatenate((value, [self.curr_rl_vehicles[key]['time_since_stopped'] / self.env_params.horizon,
                                                                      duration / 100.0])),
                                       "expert_action": np.array([np.clip(accel, a_min=self.action_space.low[0],
                                                                          a_max=self.action_space.high[0])])}
                else:
                    state_dict[key] = {"a_obs": value[:-1], "expert_action": np.array([0.0])}
        return state_dict

    def _apply_rl_actions(self, rl_actions):

        # iterate through the RL vehicles and find what the other agent would have done
        self.update_curr_rl_vehicles()
        if rl_actions:
            if self.iter_num < self.num_imitation_iters:
                id_list = []
                action_list = []
                for key, value in rl_actions.items():

                    # a vehicle may have left since we got the state
                    if key not in self.k.vehicle.get_arrived_ids() and key in self.k.vehicle.get_rl_ids():
                        controller = self.curr_rl_vehicles[key]['controller']
                        accel = controller.get_accel(self)
                        id_list.append(key)
                        if not accel:
                            accel = -np.abs(self.action_space.low[0])
                        action_list.append(accel)
                self.k.vehicle.apply_acceleration(id_list, action_list)
            else:
                super()._apply_rl_actions(rl_actions)


class MultiBottleneckDFQDEnv(MultiBottleneckEnv):
    """For the first X iterations it takes the expert action instead of the agent action"""

    def __init__(self, env_params, sim_params, scenario, simulator='traci'):
        """Initialize DesiredVelocityEnv."""
        super().__init__(env_params, sim_params, scenario, simulator)

        self.num_actions = self.env_params.additional_params['action_discretization']
        # how many steps we let the expert control the environment for
        self.num_expert_steps = self.env_params.additional_params['num_expert_steps']
        # whether to include the iteration number in the sample
        # TODO(@evinitsky) add exploration to fingerprinting
        self.fingerprinting = self.env_params.additional_params['fingerprinting']
        self.num_steps_sampled = 0
        self.iteration = 0
        self.exp_vals = 1.0

        self.action_values = np.linspace(-3, 3, self.num_actions)

    def init_decentral_controller(self, rl_id):
        return FakeDecentralizedALINEAController(rl_id, stop_edge="2", stop_pos=310,
                                                       additional_env_params=self.env_params.additional_params,
                                                       car_following_params=SumoCarFollowingParams())

    def update_curr_rl_vehicles(self):
        self.curr_rl_vehicles.update({rl_id: {'controller': self.init_decentral_controller(rl_id),
                                              'time_since_stopped': 0.0,
                                              'is_stopped': False,}
                                              for rl_id in self.k.vehicle.get_rl_ids()
                                      if rl_id not in self.curr_rl_vehicles.keys()})

    def _apply_rl_actions(self, rl_actions):

        # iterate through the RL vehicles and find what the other agent would have done
        self.update_curr_rl_vehicles()
        if rl_actions:
            if self.num_steps_sampled < self.num_expert_steps:
                id_list = []
                action_list = []
                for key, value in rl_actions.items():

                    # a vehicle may have left since we got the state
                    if key not in self.k.vehicle.get_arrived_ids() and key in self.k.vehicle.get_rl_ids():
                        controller = self.curr_rl_vehicles[key]['controller']
                        accel = controller.get_accel(self)
                        id_list.append(key)
                        if not accel:
                            accel = -np.abs(self.action_space.low[0])
                        action_list.append(accel)
                self.k.vehicle.apply_acceleration(id_list, action_list)
            else:
                action_list = [(id, self.action_values[action]) for id, action in rl_actions.items()]
                id_list = [item[0] for item in action_list]
                accels = [item[1] for item in action_list]
                self.k.vehicle.apply_acceleration(id_list, accels)

    @property
    def observation_space(self):
        # Extra keys "time since stop", duration, whether you are first in the queue, acceleration
        num_obs = super().observation_space.low.shape[0] + 3
        if self.fingerprinting:
            num_obs += 2

        if self.env_params.additional_params.get('keep_past_actions', False):
            self.num_past_actions = 100
            num_obs += self.num_past_actions

        new_obs = Box(low=-10.0, high=10.0, shape=(num_obs,), dtype=np.float32)
        # new_obs = Box(low=-3.0, high=3.0, shape=(obs.shape[0],), dtype=np.float32)
        return new_obs

    def get_state(self, rl_actions=None):
        state_dict = super().get_state(rl_actions)

        # iterate through the RL vehicles and find what the other agent would have done
        self.update_curr_rl_vehicles()
        for key, value in state_dict.items():
            if key in self.k.vehicle.get_ids():
                controller = self.curr_rl_vehicles[key]['controller']
                if self.k.vehicle.get_speed(key) <= 0.2:
                    self.curr_rl_vehicles[key]['time_since_stopped'] += 1.0
                else:
                    self.curr_rl_vehicles[key]['time_since_stopped'] = 0.0

                duration = controller.duration
                concat_list = np.concatenate((value, [self.curr_rl_vehicles[key]['time_since_stopped'] / self.env_params.horizon,
                                        duration / 100.0]))

                if self.fingerprinting:
                    # Since we only either are totally exploring or not exploring at all, lets just put a one in here for now
                    if self.num_steps_sampled < self.num_expert_steps:
                        concat_list = np.concatenate((concat_list, [self.num_steps_sampled / 1e5, 1.0]))
                    else:
                        concat_list = np.concatenate((concat_list, [self.num_steps_sampled / 1e5, 0.0]))
                accel = controller.get_accel(self)
                if not accel:
                    accel = -np.abs(self.action_values[0])
                expert_action = int(self.find_nearest_idx(self.action_values, accel))
                concat_list = np.concatenate((concat_list, [expert_action]))

                # value = np.clip(concat_list, a_min=self.observation_space.low, a_max=self.observation_space.high)
                state_dict[key] = concat_list

        return state_dict

    def reset(self, new_inflow_rate=None):

        self.curr_rl_vehicles = {}
        self.update_curr_rl_vehicles()

        state_dict = super().reset(new_inflow_rate)
        return state_dict

    def update_num_steps(self, num_steps_sampled, iteration, exp_vals):
        self.num_steps_sampled = num_steps_sampled
        print('number of sampled steps is ', self.num_steps_sampled)
        self.iteration = iteration
        self.exp_vals = exp_vals

    @property
    def action_space(self):
        return Discrete(self.num_actions)

    def find_nearest_idx(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
