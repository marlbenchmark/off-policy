"""Contains a list of custom velocity controllers."""

from flow.controllers import IDMController
from flow.controllers.base_controller import BaseController
from flow.controllers.car_following_models import SimCarFollowingController
import numpy as np


class FollowerStopper(BaseController):
    """Inspired by Dan Work's... work.

    Dissipation of stop-and-go waves via control of autonomous vehicles:
    Field experiments https://arxiv.org/abs/1705.01693

    Parameters
    ----------
    veh_id : str
        unique vehicle identifier
    v_des : float, optional
        desired speed of the vehicles (m/s)
    """

    def __init__(self,
                 veh_id,
                 car_following_params,
                 v_des=15,
                 danger_edges=None):
        """Instantiate FollowerStopper."""
        BaseController.__init__(
            self, veh_id, car_following_params, delay=1.0,
            fail_safe='safe_velocity')

        # desired speed of the vehicle
        self.v_des = v_des

        # maximum achievable acceleration by the vehicle
        self.max_accel = car_following_params.controller_params['accel']

        # other parameters
        self.dx_1_0 = 4.5
        self.dx_2_0 = 5.25
        self.dx_3_0 = 6.0
        self.d_1 = 1.5
        self.d_2 = 1.0
        self.d_3 = 0.5
        self.danger_edges = danger_edges if danger_edges else {}

    def find_intersection_dist(self, env):
        """Find distance to intersection.

        Parameters
        ----------
        env : flow.envs.Env
            see flow/envs/base_env.py

        Returns
        -------
        float
            distance from the vehicle's current position to the position of the
            node it is heading toward.
        """
        edge_id = env.k.vehicle.get_edge(self.veh_id)
        # FIXME this might not be the best way of handling this
        if edge_id == "":
            return -10
        if 'center' in edge_id:
            return 0
        edge_len = env.k.scenario.edge_length(edge_id)
        relative_pos = env.k.vehicle.get_position(self.veh_id)
        dist = edge_len - relative_pos
        return dist

    def get_accel(self, env):
        """See parent class."""
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        this_vel = env.k.vehicle.get_speed(self.veh_id)
        lead_vel = env.k.vehicle.get_speed(lead_id)

        if self.v_des is None:
            return None

        if lead_id is None:
            v_cmd = self.v_des
        else:
            dx = env.k.vehicle.get_headway(self.veh_id)
            dv_minus = min(lead_vel - this_vel, 0)

            dx_1 = self.dx_1_0 + 1 / (2 * self.d_1) * dv_minus**2
            dx_2 = self.dx_2_0 + 1 / (2 * self.d_2) * dv_minus**2
            dx_3 = self.dx_3_0 + 1 / (2 * self.d_3) * dv_minus**2
            v = min(max(lead_vel, 0), self.v_des)
            # compute the desired velocity
            if dx <= dx_1:
                v_cmd = 0
            elif dx <= dx_2:
                v_cmd = v * (dx - dx_1) / (dx_2 - dx_1)
            elif dx <= dx_3:
                v_cmd = v + (self.v_des - this_vel) * (dx - dx_2) \
                        / (dx_3 - dx_2)
            else:
                v_cmd = self.v_des

        edge = env.k.vehicle.get_edge(self.veh_id)

        if edge == "":
            return None

        if self.find_intersection_dist(env) <= 10 and \
                env.k.vehicle.get_edge(self.veh_id) in self.danger_edges or \
                env.k.vehicle.get_edge(self.veh_id)[0] == ":":
            return None
        else:
            # compute the acceleration from the desired velocity
            return (v_cmd - this_vel) / env.sim_step


class PISaturation(BaseController):
    """Inspired by Dan Work's... work.

    Dissipation of stop-and-go waves via control of autonomous vehicles:
    Field experiments https://arxiv.org/abs/1705.01693

    Parameters
    ----------
    veh_id : str
        unique vehicle identifier
    car_following_params : flow.core.params.SumoCarFollowingParams
        object defining sumo-specific car-following parameters
    """

    def __init__(self, veh_id, car_following_params):
        """Instantiate PISaturation."""
        BaseController.__init__(self, veh_id, car_following_params, delay=1.0)

        # maximum achievable acceleration by the vehicle
        self.max_accel = car_following_params.controller_params['accel']

        # history used to determine AV desired velocity
        self.v_history = []

        # other parameters
        self.gamma = 2
        self.g_l = 7
        self.g_u = 30
        self.v_catch = 1

        # values that are updated by using their old information
        self.alpha = 0
        self.beta = 1 - 0.5 * self.alpha
        self.U = 0
        self.v_target = 0
        self.v_cmd = 0

    def get_accel(self, env):
        """See parent class."""
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        lead_vel = env.k.vehicle.get_speed(lead_id)
        this_vel = env.k.vehicle.get_speed(self.veh_id)

        dx = env.k.vehicle.get_headway(self.veh_id)
        dv = lead_vel - this_vel
        dx_s = max(2 * dv, 4)

        # update the AV's velocity history
        self.v_history.append(this_vel)

        if len(self.v_history) == int(38 / env.sim_step):
            del self.v_history[0]

        # update desired velocity values
        v_des = np.mean(self.v_history)
        v_target = v_des + self.v_catch \
            * min(max((dx - self.g_l) / (self.g_u - self.g_l), 0), 1)

        # update the alpha and beta values
        alpha = min(max((dx - dx_s) / self.gamma, 0), 1)
        beta = 1 - 0.5 * alpha

        # compute desired velocity
        self.v_cmd = beta * (alpha * v_target + (1 - alpha) * lead_vel) \
            + (1 - beta) * self.v_cmd

        # compute the acceleration
        accel = (self.v_cmd - this_vel) / env.sim_step

        return min(accel, self.max_accel)


class HandTunedVelocityController(IDMController):
    def __init__(self,
                 veh_id,
                 v_regions,
                 car_following_params):
        super().__init__(
            veh_id, car_following_params=car_following_params)
        self.v_regions = v_regions

    def get_accel(self, env):
        env.k.vehicle.set_color(self.veh_id, (255, 0, 0))
        edge = env.k.vehicle.get_edge(self.veh_id)
        if edge:
            if edge[0] != ':' and edge in env.controlled_edges:
                pos = env.k.vehicle.get_position(self.veh_id)
                # find what segment we fall into
                bucket = np.searchsorted(env.slices[edge], pos) - 1
                
                action = self.v_regions[bucket +
                                        env.action_index[edge]]
                if self.veh_id == "flow_0.0":
                    print("edge, bucket, action", edge, bucket, action)
                # set the desired velocity of the controller to the action
                controller = env.k.vehicle.set_max_speed(self.veh_id, action)
        
        action = super().get_accel(env)
        return action


class TimeDelayVelocityController(SimCarFollowingController):
    def __init__(self,
                 veh_id,
                 stop_edge,
                 stop_pos,
                 car_following_params):
        """[summary]
        
        Arguments:
            veh_id {[int]} 
            stop_distance {[type]} -- distance from bottleneck to stop at

            car_following_params {[type]} -- 
        """
        super().__init__(
            veh_id, car_following_params=car_following_params)
        self.stop_pos = stop_pos
        self.stop_edge = stop_edge
        self.stop_set = False

    def get_duration(self, env):
        return 10.0
    
    def set_stop(self, env):
        lane_id = env.k.vehicle.get_lane(self.veh_id)
        duration = self.get_duration(env)

        if duration < 1.0:
            if self.stop_set:
                env.k.vehicle.cancel_stop(self.veh_id, edgeid=self.stop_edge, pos=self.stop_pos, lane=lane_id)
            self.stop_set = False
        else:
            env.k.vehicle.set_stop_with_duration(self.veh_id, edgeid=self.stop_edge, pos=self.stop_pos, lane=lane_id, duration=duration)
            self.duration = duration
            self.stop_set = True

    def get_accel(self, env):
        if (env.k.vehicle.is_stopped(self.veh_id)):
            env.k.vehicle.set_color(self.veh_id, (255, 255, 0))
        else:
            env.k.vehicle.set_color(self.veh_id, (0, 255, 0))
        cur_pos = env.k.vehicle.get_position(self.veh_id)
        cur_speed = env.k.vehicle.get_speed(self.veh_id)
        if int(env.k.vehicle.get_edge(self.veh_id)) > int(self.stop_edge):
            return None
        elif int(env.k.vehicle.get_edge(self.veh_id)) == int(self.stop_edge) and self.stop_pos - cur_pos - 4 < (cur_speed**2) / self.car_following_params.controller_params['decel']:
            return None
        else:
            self.set_stop(env)

        return None


class DecentralizedALINEAController(TimeDelayVelocityController):
    def __init__(self, veh_id, stop_edge, stop_pos, additional_env_params, car_following_params):
        super().__init__(veh_id, stop_edge, stop_pos, car_following_params)
        # values for the ALINEA ramp meter algorithm
        self.n_crit = additional_env_params.get("n_crit")
        self.q_max = 14401
        self.q_min = 200
        self.feedback_coeff = additional_env_params.get('feedback_coeff')
        self.q = additional_env_params.get('q_init')  # 600 # ramp meter feedback controller
        self.feedback_update_time = 0
        self.feedback_timer = 0.0
        self.duration = 0.0

    def get_duration(self, env):
        self.feedback_timer += env.sim_step
        if self.feedback_timer > self.feedback_update_time:
            self.feedback_timer = 0
            # now implement the integral controller update
            # find all the vehicles in an edge
            q_update = self.feedback_coeff * (
                self.n_crit - np.average(env.smoothed_num))
            self.q = min(max(self.q + q_update, self.q_min), self.q_max)
            # convert q to cycle time, we keep track of the previous cycle time to let the cycle coplete
            duration = 3600 * env.scaling * 4 / self.q
        return duration

class StaggeringDecentralizedALINEAController(DecentralizedALINEAController):
    def __init__(self, veh_id, stop_edge, stop_pos, additional_env_params, car_following_params):
        super().__init__(veh_id, stop_edge, stop_pos, additional_env_params, car_following_params)
        self.is_waiting_to_go = False
        self.lane_leader = False
        self.check_next = False

    def get_accel(self, env):
        env.k.vehicle.set_color(self.veh_id, (0, 255, 0))
        cur_pos = env.k.vehicle.get_position(self.veh_id)
        cur_speed = env.k.vehicle.get_speed(self.veh_id)
        cur_lane = env.k.vehicle.get_lane(self.veh_id)

        if not self.lane_leader:
            cars_in_lane = []
            if self.stop_edge in env.edge_dict:
                cars_in_lane = env.edge_dict[self.stop_edge][cur_lane]
            if len(cars_in_lane) and max(cars_in_lane, key=lambda x: x[1])[0] == self.veh_id and self.stop_set:
                self.lane_leader = True
                env.waiting_queue.append(self.veh_id)

        if int(env.k.vehicle.get_edge(self.veh_id)) > int(self.stop_edge):
            if self.veh_id in env.waiting_queue:
                env.waiting_queue.remove(self.veh_id)
            return None
        elif self.is_waiting_to_go:
            if not env.k.vehicle.is_stopped(self.veh_id):
                if (env.waiting_queue[0] == self.veh_id):
                    if len(env.waiting_queue) == 4:
                        # car can depart
                        env.waiting_queue.pop(0)
                        self.is_waiting_to_go = False
                        return None
            return 0.0
        elif env.k.vehicle.is_stopped(self.veh_id):
            self.is_waiting_to_go = True
            return 0.0
        elif not self.stop_set and int(env.k.vehicle.get_edge(self.veh_id)) == int(self.stop_edge) and \
                0.5 * (cur_speed ** 2) / self.car_following_params.controller_params[
            'decel'] + cur_pos > self.stop_pos - 4:
            return None
        else:
            self.set_stop(env)

        return None

class FakeDecentralizedALINEAController(DecentralizedALINEAController):
    """Same as the controller above but never actually calls set_stop"""

    def __init__(self, veh_id, stop_edge, stop_pos, additional_env_params, car_following_params):
        super().__init__(veh_id, stop_edge, stop_pos, additional_env_params, car_following_params)
        self.idm_controller = IDMController(veh_id, car_following_params=car_following_params)       
        self.stop_time = 0.0 
        self.is_waiting_to_go = False

    def get_accel(self, env):
        env.k.vehicle.set_color(self.veh_id, (255, 0, 0))
        cur_pos = env.k.vehicle.get_position(self.veh_id)
        cur_speed = env.k.vehicle.get_speed(self.veh_id)
        cur_lane = env.k.vehicle.get_lane(self.veh_id)

        if len(env.k.vehicle.get_edge(self.veh_id)) and env.k.vehicle.get_edge(self.veh_id)[0] != ':':
            if int(env.k.vehicle.get_edge(self.veh_id)) > int(self.stop_edge):
                if self.veh_id in env.waiting_queue:
                    env.waiting_queue.remove(self.veh_id)
                self.is_waiting_to_go = False
                return self.idm_controller.get_accel(env)
            elif self.is_waiting_to_go:
                # stop until conditions are met
                if env.sim_step * (env.time_counter + 1) > self.stop_time + self.duration:
                    self.stop_set = False
                    return self.idm_controller.get_accel(env)
                else:
                    return -np.abs(self.max_deaccel)
            elif self.stop_set:
                self.set_stop(env)
                b = self.max_deaccel
                dt = env.sim_step
                h = self.stop_pos - cur_pos - 4
                if (b ** 2) * (dt ** 2) + 2 * b * h > 0:
                    safe_velocity = - b * dt + np.sqrt((b ** 2) * (dt ** 2) + 2 * b * h)
                else:
                    safe_velocity = 0.0
                idm_accel = self.idm_controller.get_accel(env)

                if self.stop_pos - cur_pos < 4:
                    self.stop_time = env.sim_step * env.time_counter
                    self.is_waiting_to_go = True
                    return -np.abs(self.max_deaccel)
                else:
                    if cur_speed + idm_accel * env.sim_step > safe_velocity:
                        if safe_velocity > 0:
                            return (safe_velocity - cur_speed) / env.sim_step
                        else:
                            return -np.abs(self.max_deaccel) # return max deaccel
                    else:
                        return idm_accel
            else:
                self.set_stop(env)
                return self.idm_controller.get_accel(env)
        else:
            return self.idm_controller.get_accel(env)

    def set_stop(self, env):
        duration = self.get_duration(env)
        if duration < 1.0:        
            self.duration = 0.0
            self.stop_set = False
        else:
            self.duration = duration
            self.stop_set = True



class FakeStaggeringDecentralizedALINEAController(StaggeringDecentralizedALINEAController):
    """Same as the controller above but never actually calls set_stop"""

    def __init__(self, veh_id, stop_edge, stop_pos, additional_env_params, car_following_params):
        super().__init__(veh_id, stop_edge, stop_pos, additional_env_params, car_following_params)
        self.idm_controller = IDMController(veh_id, car_following_params=car_following_params)       
        self.stop_time = 0.0 

    def get_accel(self, env):
        cur_pos = env.k.vehicle.get_position(self.veh_id)
        cur_speed = env.k.vehicle.get_speed(self.veh_id)
        cur_lane = env.k.vehicle.get_lane(self.veh_id)

        if len(env.k.vehicle.get_edge(self.veh_id)) and env.k.vehicle.get_edge(self.veh_id)[0] != ':':
            if not self.lane_leader:
                cars_in_lane = []
                if self.stop_edge in env.edge_dict:
                    cars_in_lane = env.edge_dict[self.stop_edge][cur_lane]
                if len(cars_in_lane) and max(cars_in_lane, key=lambda x: x[1])[0] == self.veh_id and self.stop_set:
                    self.lane_leader = True
                    env.waiting_queue.append(self.veh_id)

            # if self.lane_leader and len(env.waiting_queue) == 4:
            #     all_stopped = False
            #     for waiting_veh in env.waiting_queue:
            #         if self.env.kernel.vehicle.get_speed()

            if int(env.k.vehicle.get_edge(self.veh_id)) > int(self.stop_edge):
                if self.veh_id in env.waiting_queue:
                    env.waiting_queue.remove(self.veh_id)
                self.is_waiting_to_go = False
                return self.idm_controller.get_accel(env)
            elif self.is_waiting_to_go:
                # stop until conditions are met
                if env.sim_step * (env.time_counter + 1) > self.stop_time + self.duration:
                    if (env.waiting_queue[0] == self.veh_id):
                        if len(env.waiting_queue) == 4:
                            self.stop_set = False
                            return self.idm_controller.get_accel(env)
                return -1 * self.max_deaccel
            elif self.stop_set:
                self.set_stop(env)
                b = self.max_deaccel
                dt = env.sim_step
                h = self.stop_pos - cur_pos - 4
                if (b ** 2) * (dt ** 2) + 2 * b * h > 0:
                    safe_velocity = - b * dt + np.sqrt((b ** 2) * (dt ** 2) + 2 * b * h)
                else:
                    safe_velocity = 0.0
                idm_accel = self.idm_controller.get_accel(env)

                if self.stop_pos - cur_pos < 4:
                    self.stop_time = env.sim_step * env.time_counter
                    self.is_waiting_to_go = True
                    return None
                else:
                    if cur_speed + idm_accel * env.sim_step > safe_velocity:
                        if safe_velocity > 0:
                            return (safe_velocity - cur_speed) / env.sim_step
                        else:
                            return None # return max deaccel
                    else:
                        return idm_accel
            else:
                self.set_stop(env)
                return self.idm_controller.get_accel(env)
        else:
            return self.idm_controller.get_accel(env)

    def set_stop(self, env):
        duration = self.get_duration(env)
        if duration < 1.0:                
            self.stop_set = False
        else:
            self.duration = duration
            self.stop_set = True


class FakeDecentralizedALINEAController(TimeDelayVelocityController):
    def __init__(self, veh_id, stop_edge, stop_pos, additional_env_params, car_following_params):
        super().__init__(veh_id, stop_edge, stop_pos, car_following_params)
        # values for the ALINEA ramp meter algorithm
        self.n_crit = additional_env_params.get("n_crit")
        self.q_max = 14401
        self.q_min = 200
        self.feedback_coeff = additional_env_params.get('feedback_coeff')
        self.q = additional_env_params.get('q_init')  # 600 # ramp meter feedback controller
        self.feedback_update_time = 0
        self.feedback_timer = 0.0
        self.duration = 0.0
        self.idm_controller = IDMController(veh_id, car_following_params=car_following_params)
        self.stop_time = 0.0
        self.is_waiting_to_go = False
        self.stop_set = False


    def get_duration(self, env):
        self.feedback_timer += env.sim_step
        if self.feedback_timer > self.feedback_update_time:
            self.feedback_timer = 0
            # now implement the integral controller update
            # find all the vehicles in an edge
            q_update = self.feedback_coeff * (
                self.n_crit - np.average(env.smoothed_num))
            self.q = min(max(self.q + q_update, self.q_min), self.q_max)
            # convert q to cycle time, we keep track of the previous cycle time to let the cycle coplete
            duration = 3600 * env.scaling * 4 / self.q
        return duration

    def set_stop(self, env):
        duration = self.get_duration(env)

        if duration < 1.0:
            self.stop_set = False
        else:
            self.duration = duration
            self.stop_set = True

    def get_accel(self, env):
        if (env.k.vehicle.is_stopped(self.veh_id)):
            env.k.vehicle.set_color(self.veh_id, (255, 255, 0))
        else:
            env.k.vehicle.set_color(self.veh_id, (0, 255, 0))
        cur_pos = env.k.vehicle.get_position(self.veh_id)
        cur_speed = env.k.vehicle.get_speed(self.veh_id)

        if self.stop_pos - cur_pos < 4 and not self.is_waiting_to_go:
            self.stop_time = env.sim_step * env.time_counter
            self.is_waiting_to_go = True

        self.set_stop(env)

        if len(env.k.vehicle.get_edge(self.veh_id)) and env.k.vehicle.get_edge(self.veh_id)[0] != ':':

            if int(env.k.vehicle.get_edge(self.veh_id)) > int(self.stop_edge):
                return self.idm_controller.get_accel(env)
            elif self.is_waiting_to_go:
                # stop until conditions are met
                if env.sim_step * (env.time_counter + 1) > self.stop_time + self.duration:
                    return self.idm_controller.get_accel(env)
                return -1 * self.max_deaccel
            elif self.stop_set:
                self.set_stop(env)
                b = self.max_deaccel
                dt = env.sim_step
                h = self.stop_pos - cur_pos - 4
                if (b ** 2) * (dt ** 2) + 2 * b * h > 0:
                    safe_velocity = - b * dt + np.sqrt((b ** 2) * (dt ** 2) + 2 * b * h)
                else:
                    safe_velocity = 0.0
                idm_accel = self.idm_controller.get_accel(env)

                if cur_speed + idm_accel * env.sim_step > safe_velocity:
                    if safe_velocity > 0:
                        return (safe_velocity - cur_speed) / env.sim_step
                    else:
                        return None  # return max deaccel
                else:
                    return idm_accel
            else:
                return self.idm_controller.get_accel(env)
        else:
            return self.idm_controller.get_accel(env)
