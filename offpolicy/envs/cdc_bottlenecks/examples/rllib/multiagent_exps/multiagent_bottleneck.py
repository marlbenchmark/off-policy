"""Multi-agent Bottleneck example.
In this example, each agent is given a single acceleration per timestep.

The agents all share a single model.
"""
from copy import deepcopy
from datetime import datetime
import errno
import json
from math import ceil
import os
import subprocess
import sys

import numpy as np
import pytz
import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.agents.ddpg.td3 import TD3_DEFAULT_CONFIG, TD3Trainer
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune import run
from ray.tune.registry import register_env

from flow.agents.custom_ppo import CustomPPOTrainer, CustomPPOTFPolicy
from flow.agents.centralized_PPO import CentralizedCriticModel, CentralizedCriticModelRNN
from flow.agents.centralized_PPO import CCTrainer
from flow.agents.centralized_imitation_PPO import ImitationCentralizedTrainer
from flow.agents.DQfD import DQFDTrainer
import flow.agents.DQfD as DQfD

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoLaneChangeParams, SumoCarFollowingParams
from flow.core.params import TrafficLightParams
from flow.core.params import VehicleParams
from flow.controllers import RLController, ContinuousRouter, \
    SimLaneChangeController
from flow.models.GRU import GRU
from flow.visualize.bottleneck_results import run_bottleneck_results
from flow.utils.parsers import get_multiagent_bottleneck_parser
from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder


# TODO(@evinitsky) clean this up
EXAMPLE_USAGE = """
example usage:
    python ./visualizer_rllib.py /ray_results/experiment_dir/result_dir 1

Here the arguments are:
1 - the path to the simulation results
2 - the number of the checkpoint
"""


def setup_rllib_params(args):
    # time horizon of a single rollout
    horizon = args.horizon
    # number of parallel workers
    n_cpus = args.n_cpus
    # number of rollouts per training iteration scaled by how many sets of rollouts per iter we want
    n_rollouts = int(args.n_cpus * args.rollout_scale_factor)
    return {'horizon': horizon, 'n_cpus': n_cpus, 'n_rollouts': n_rollouts}


def setup_flow_params(args):
    DISABLE_TB = True
    DISABLE_RAMP_METER = True
    av_frac = args.av_frac
    if args.lc_on:
        lc_mode = 1621
    else:
        lc_mode = 0

    vehicles = VehicleParams()
    if not np.isclose(av_frac, 1):
        vehicles.add(
            veh_id="human",
            lane_change_controller=(SimLaneChangeController, {}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode=31,
            ),
            lane_change_params=SumoLaneChangeParams(
                lane_change_mode=lc_mode,
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
                lane_change_mode=0,
            ),
            num_vehicles=1)

    # flow rate
    flow_rate = 1900 * args.scaling

    controlled_segments = [('1', 1, False), ('2', 2, True), ('3', 2, True),
                           ('4', 2, True), ('5', 1, False)]
    num_observed_segments = [('1', 1), ('2', 3), ('3', 3), ('4', 3), ('5', 1)]
    if np.isclose(args.av_frac, 0.4):
        q_init = 1000
    else:
        q_init = 600
    additional_env_params = {
        'target_velocity': 40,
        'disable_tb': True,
        'disable_ramp_metering': True,
        'controlled_segments': controlled_segments,
        'symmetric': False,
        'observed_segments': num_observed_segments,
        'reset_inflow': True,
        'lane_change_duration': 5,
        'max_accel': 3,
        'max_decel': 3,
        'inflow_range': [args.low_inflow, args.high_inflow],
        'start_inflow': flow_rate,
        'congest_penalty': args.congest_penalty,
        'communicate': args.communicate,
        "centralized_obs": args.central_obs,
        "aggregate_info": args.aggregate_info,
        "av_frac": args.av_frac,
        "congest_penalty_start": args.congest_penalty_start,
        "lc_mode": lc_mode,
        "life_penalty": args.life_penalty,
        'keep_past_actions': args.keep_past_actions,
        "num_sample_seconds": args.num_sample_seconds,
        "speed_reward": args.speed_reward,
        'fair_reward': False,  # This doesn't do anything, remove
        'exit_history_seconds': 0,  # This doesn't do anything, remove
        'reroute_on_exit': args.reroute_on_exit,

        # parameters for the staggering controller that we imitate
        "n_crit": 8,
        "q_max": 15000,
        "q_min": 200,
        "q_init": q_init, #
        "feedback_coeff": 1, #
        'num_imitation_iters': args.num_imitation_iters,

        # parameters from imitation
        "simple_env": args.simple_env,
        "super_simple_env": args.super_simple_env,

        # curriculum stuff
        "curriculum": args.curriculum,
        "num_curr_iters": args.num_curr_iters,
        "min_horizon": args.min_horizon,
        "horizon": args.horizon,
        "rew_n_crit": args.rew_n_crit,
        "warmup_fixed_agents": args.warmup_fixed_agents,
        "num_agents": args.num_agents
    }

    if args.dqfd:
        additional_env_params.update({
            "num_expert_steps": args.num_expert_steps,
            "action_discretization": 5,
            "fingerprinting": args.fingerprinting
        })

    # percentage of flow coming out of each lane
    inflow = InFlows()
    if not np.isclose(args.av_frac, 1.0):
        inflow.add(
            veh_type='human',
            edge='1',
            vehs_per_hour=flow_rate * (1 - args.av_frac),
            departLane='random',
            departSpeed=23.0)
        inflow.add(
            veh_type='av',
            edge='1',
            vehs_per_hour=flow_rate * args.av_frac,
            departLane='random',
            departSpeed=23.0)
    else:
        inflow.add(
            veh_type='av',
            edge='1',
            vehs_per_hour=flow_rate,
            departLane='random',
            departSpeed=23.0)

    traffic_lights = TrafficLightParams()
    if not DISABLE_TB:
        traffic_lights.add(node_id='2')
    if not DISABLE_RAMP_METER:
        traffic_lights.add(node_id='3')

    additional_net_params = {'scaling': args.scaling, "speed_limit": 23.0}

    if args.imitate:
        env_name = 'MultiBottleneckImitationEnv'
    elif args.dqfd:
        env_name = 'MultiBottleneckDFQDEnv'
    else:
        env_name = 'MultiBottleneckEnv'

    # if args.super_simple_env:
    #     scenario = 'SimpleBottleneckScenario'
    # else:
    scenario='BottleneckScenario'
    warmup_steps = 0
    if args.reroute_on_exit:
        warmup_steps = int(300 / args.sims_per_step)

    flow_params = dict(
        # name of the experiment
        exp_tag=args.exp_title,

        # name of the flow environment the experiment is running on
        env_name=env_name,

        # name of the scenario class the experiment is running on
        scenario=scenario,

        # simulator that is used by the experiment
        simulator='traci',

        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=SumoParams(
            sim_step=args.sim_step,
            render=args.render,
            print_warnings=False,
            restart_instance=True
        ),

        # environment related parameters (see flow.core.params.EnvParams)
        env=EnvParams(
            warmup_steps=int(warmup_steps / args.sim_step),
            sims_per_step=args.sims_per_step,
            horizon=args.horizon,
            clip_actions=False,
            additional_params=additional_env_params,
        ),

        # network-related parameters (see flow.core.params.NetParams and the
        # scenario's documentation or ADDITIONAL_NET_PARAMS component)
        net=NetParams(
            inflows=inflow,
            no_internal_links=False,
            additional_params=additional_net_params,
        ),

        # vehicles to be placed in the network at the start of a rollout (see
        # flow.core.vehicles.Vehicles)
        veh=vehicles,

        # parameters specifying the positioning of vehicles upon initialization/
        # reset (see flow.core.params.InitialConfig)
        initial=InitialConfig(
            spacing='uniform',
            min_gap=5,
            lanes_distribution=float('inf'),
            edges_distribution=['2', '3', '4', '5'],
        ),

        # traffic lights to be introduced to specific nodes (see
        # flow.core.traffic_lights.TrafficLights)
        tls=traffic_lights,
    )
    return flow_params


def on_episode_start(info):
    episode = info["episode"]
    episode.user_data["outflow"] = []
    episode.user_data["n_crit"] = []
    episode.user_data["speed_edge_4"] = []
    episode.user_data["n_veh_edge4_l0"] = []
    episode.user_data["n_veh_edge4_l1"] = []
    episode.user_data["mean_speed_edge2_l0"] = []
    episode.user_data["mean_speed_edge2_l1"] = []
    episode.user_data["mean_speed_edge2_l2"] = []
    episode.user_data["mean_speed_edge2_l3"] = []


def on_episode_end(info):
    env = info['env'].get_unwrapped()[0]
    total_time_step = env.env_params.horizon * env.sim_step * env.env_params.sims_per_step
    time_step = 250
    episode = info["episode"]
    episode.custom_metrics["num_congested"] = np.mean(episode.user_data["n_crit"])
    episode.custom_metrics["speed_edge_4"] = np.mean(episode.user_data["speed_edge_4"])
    episode.custom_metrics["n_veh_edge4_l0"] = np.mean(episode.user_data["n_veh_edge4_l0"])
    episode.custom_metrics["n_veh_edge4_l1"] = np.mean(episode.user_data["n_veh_edge4_l1"])
    episode.custom_metrics["mean_speed_edge2_l0"] = np.mean(episode.user_data["mean_speed_edge2_l0"])
    episode.custom_metrics["mean_speed_edge2_l1"] = np.mean(episode.user_data["mean_speed_edge2_l1"])
    episode.custom_metrics["mean_speed_edge2_l2"] = np.mean(episode.user_data["mean_speed_edge2_l2"])
    episode.custom_metrics["mean_speed_edge2_l3"] = np.mean(episode.user_data["mean_speed_edge2_l3"])
    episode.custom_metrics["exit_counter"] = env.exit_counter * (3600 / 500)

    step_offset = env.env_params.warmup_steps * env.sim_step * env.env_params.sims_per_step
    for i in range(int(ceil(total_time_step / time_step))):
        total_outflow = env.k.vehicle.get_outflow_rate_between_times(step_offset + i * time_step, step_offset + (i+1) * time_step)
        inflow = env.inflow
        # round it to 100
        inflow = int(inflow / 100) * 100
        episode.custom_metrics["net_outflow_{}_time0_{}_time1_{}".format(inflow, time_step * i, time_step * (i+1))] = total_outflow


def on_episode_step(info):
    episode = info["episode"]
    env = info['env'].get_unwrapped()[0]
    outflow = env.k.vehicle.get_outflow_rate(int(env.sim_step * env.env_params.sims_per_step))
    episode.user_data["outflow"].append(outflow)
    edge_4_veh = env.k.vehicle.get_ids_by_edge('4')
    episode.user_data["speed_edge_4"].append(np.nan_to_num(np.mean(env.k.vehicle.get_speed(edge_4_veh))))
    episode.user_data["n_crit"].append(len(edge_4_veh))
    l0 = np.sum(["0" == env.k.vehicle.get_lane(veh_id) for veh_id in edge_4_veh])
    episode.user_data["n_veh_edge4_l0"].append(l0)
    l1 = np.sum(["1" == env.k.vehicle.get_lane(veh_id) for veh_id in edge_4_veh])
    episode.user_data["n_veh_edge4_l1"].append(l1)

    edge_2_veh = env.k.vehicle.get_ids_by_edge('2')
    l0_edge2 = [veh_id for veh_id in edge_2_veh if env.k.vehicle.get_lane(veh_id) == "0"]
    l1_edge2 = [veh_id for veh_id in edge_2_veh if env.k.vehicle.get_lane(veh_id) == "1"]
    l2_edge2 = [veh_id for veh_id in edge_2_veh if env.k.vehicle.get_lane(veh_id) == "2"]
    l3_edge2 = [veh_id for veh_id in edge_2_veh if env.k.vehicle.get_lane(veh_id) == "3"]
    episode.user_data["mean_speed_edge2_l0"].append(np.nan_to_num(np.mean(env.k.vehicle.get_speed(l0_edge2))))
    episode.user_data["mean_speed_edge2_l1"].append(np.nan_to_num(np.mean(env.k.vehicle.get_speed(l1_edge2))))
    episode.user_data["mean_speed_edge2_l2"].append(np.nan_to_num(np.mean(env.k.vehicle.get_speed(l2_edge2))))
    episode.user_data["mean_speed_edge2_l3"].append(np.nan_to_num(np.mean(env.k.vehicle.get_speed(l3_edge2))))



def on_train_result(info):
    """Store the mean score of the episode, and increment or decrement how many adversaries are on"""
    result = info["result"]
    trainer = info["trainer"]
    trainer.workers.foreach_worker(
        lambda ev: ev.foreach_env(
            lambda env: env.set_iteration_num(result['training_iteration'])))


def on_train_result_dqfd(info):
    trainer = info["trainer"]
    iteration = trainer._iteration
    num_steps_sampled = trainer.optimizer.num_steps_sampled
    exp_vals = [trainer.exploration0.value(num_steps_sampled)]
    trainer.workers.foreach_worker(
        lambda ev: ev.foreach_env(
            lambda env: env.update_num_steps(num_steps_sampled, iteration, exp_vals)))

def on_train_result_curriculum(info):
    trainer = info["trainer"]

    trainer.workers.foreach_worker(
        lambda ev: ev.foreach_env(
            lambda env: env.increase_curr_iter()))


def setup_exps(args):
    rllib_params = setup_rllib_params(args)
    flow_params = setup_flow_params(args)
    if args.dqfd:
        alg_run = 'DQFD'
        config = DQfD.DEFAULT_CONFIG.copy()
        config['num_expert_steps'] = args.num_expert_steps
        config['compress_observations'] = False
        # Grid search things
        if args.grid_search:
            config['lr'] = tune.grid_search([5e-6, 5e-5, 5e-4])
            config['n_step'] = tune.grid_search([1, 5, 10])
            config['train_batch_size'] = tune.grid_search([32])
            config['reserved_frac'] = tune.grid_search([0.1, 0.3])
    elif args.td3:
        alg_run = 'TD3'
        config = deepcopy(TD3_DEFAULT_CONFIG)
        config["buffer_size"] = 100000
        config["sample_batch_size"] = 5
        if args.local_mode:
            config["learning_starts"] = 1000
            config["pure_exploration_steps"] = 1000
        else:
            config["learning_starts"] = 10000
            config["pure_exploration_steps"] = 10000
        if args.grid_search:
            config["prioritized_replay"] = args.td3_prioritized_replay # tune.grid_search(['True', 'False'])
            config["actor_lr"] = args.td3_actor_lr # tune.grid_search([1e-3, 1e-4])
            config["critic_lr"] = args.td3_critic_lr # tune.grid_search([1e-3, 1e-4])
            config["n_step"] = args.td3_n_step # tune.grid_search([1, 5])
            config["seed"] = tune.grid_search([None] + list(range(34)))
    else:
        alg_run = 'PPO'
        config = ppo.DEFAULT_CONFIG.copy()
        config['num_workers'] = rllib_params['n_cpus']
        config['train_batch_size'] = args.horizon * rllib_params['n_rollouts']
        config["entropy_coeff"] = args.entropy_coeff

        # if we have a centralized vf we can't use big batch sizes or we eat up all the system memory
        config['sgd_minibatch_size'] = 128
        if args.use_lstm:
            config['vf_loss_coeff'] = args.vf_loss_coeff
            # Grid search things
        if args.grid_search:
            config['lr'] = tune.grid_search([5e-5, 5e-4, 5e-3])
        else:
            config['num_sgd_iter'] = 10

            # LSTM Things
        if args.use_lstm and args.use_gru:
            sys.exit("You should not specify both an LSTM and a GRU")
        if args.use_lstm:
            if args.grid_search:
                config['model']["max_seq_len"] = tune.grid_search([10, 20])
            else:
                config['model']["max_seq_len"] = 20
            config['model'].update({'fcnet_hiddens': []})
            config['model']["lstm_cell_size"] = 64
            config['model']['lstm_use_prev_action_reward'] = True
        elif args.use_gru:
            if args.grid_search:
                config['model']["max_seq_len"] = tune.grid_search([20, 40])
            else:
                config['model']["max_seq_len"] = 20
                config['model'].update({'fcnet_hiddens': []})
            model_name = "GRU"
            ModelCatalog.register_custom_model(model_name, GRU)
            config['model']['custom_model'] = model_name
            config['model']['custom_options'].update({"cell_size": 64, 'use_prev_action': True})
        else:
            config['model'].update({'fcnet_hiddens': [100, 50, 25]})
            # model_name = "FeedForward"
            # ModelCatalog.register_custom_model(model_name, FeedForward)
            # config['model']['custom_model'] = model_name
            # config['model']['custom_options'].update({'use_prev_action': True})

            # model setup for the centralized case
            # Set up model
        if args.centralized_vf:
            if args.use_lstm:
                ModelCatalog.register_custom_model("cc_model", CentralizedCriticModelRNN)
            else:
                ModelCatalog.register_custom_model("cc_model", CentralizedCriticModel)
            config['model']['custom_model'] = "cc_model"
            config['model']['custom_options']['central_vf_size'] = args.central_vf_size
            config['model']['custom_options']['max_num_agents'] = args.max_num_agents

        if args.imitate:
            config[
                'kl_coeff'] = 20  # start with it high so we take smaller steps to start and don't just forget the imitation
            config['model']['custom_options'].update({"imitation_weight": 1e0})
            config['model']['custom_options'].update({"num_imitation_iters": args.num_imitation_iters})
            config['model']['custom_options']['hard_negative_mining'] = args.hard_negative_mining
            config['model']['custom_options']['mining_frac'] = args.mining_frac
            config["model"]["custom_options"]["final_imitation_weight"] = args.final_imitation_weight

    config['gamma'] = 0.99  # discount rate
    config['horizon'] = args.horizon
    config['no_done_at_end'] = True
    # config["batch_mode"] = "truncate_episodes"
    # config["sample_batch_size"] = args.horizon
    # config["observation_filter"] = "MeanStdFilter"
    config['model']['custom_options']['terminal_reward'] = args.terminal_reward
    config['model']['custom_options']['post_exit_rew_len'] = args.post_exit_rew_len
    config['model']['custom_options']['horizon'] = args.horizon

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json

    create_env, env_name = make_create_env(params=flow_params, version=0)

    # Register as rllib env
    register_env(env_name, create_env)

    test_env = create_env()
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    # Setup PG with an ensemble of `num_policies` different policy graphs
    if alg_run == 'PPO':
        policy_graphs = {'av': (CustomPPOTFPolicy, obs_space, act_space, {})}

    else:
        policy_graphs = {'av': (None, obs_space, act_space, {})}

    def policy_mapping_fn(_):
        return 'av'

    config.update({
        'multiagent': {
            'policies': policy_graphs,
            'policy_mapping_fn': tune.function(policy_mapping_fn),
            "policies_to_train": ["av"]
        }
    })
    return alg_run, env_name, config


if __name__ == '__main__':
    parser = get_multiagent_bottleneck_parser()
    args = parser.parse_args()

    alg_run, env_name, config = setup_exps(args)
    if args.multi_node:
        ray.init(redis_address='localhost:6379')
    elif args.local_mode:
        ray.init(local_mode=True)
    else:
        ray.init()
    eastern = pytz.timezone('US/Eastern')
    date = datetime.now(tz=pytz.utc)
    date = date.astimezone(pytz.timezone('US/Pacific')).strftime("%m-%d-%Y")
    s3_string = "s3://nathan.experiments/trb_bottleneck_paper/" \
                + date + '/' + args.exp_title
    config['env'] = env_name


    # create a custom string that makes looking at the experiment names easier
    def trial_str_creator(trial):
        return "{}_{}".format(trial.trainable_name, trial.experiment_tag)

    # store custom metrics
    if args.imitate:
        config["callbacks"] = {"on_episode_end": tune.function(on_episode_end),
                               "on_episode_start": tune.function(on_episode_start),
                               "on_episode_step": tune.function(on_episode_step),
                               "on_train_result": tune.function(on_train_result)}
    elif args.dqfd:
        config["callbacks"] = {"on_episode_end": tune.function(on_episode_end),
                               "on_episode_start": tune.function(on_episode_start),
                               "on_episode_step": tune.function(on_episode_step),
                               "on_train_result": tune.function(on_train_result_dqfd)}
    else:
        config["callbacks"] = {"on_episode_end": tune.function(on_episode_end),
                               "on_episode_start": tune.function(on_episode_start),
                               "on_episode_step": tune.function(on_episode_step),}
    if args.curriculum:
        config["callbacks"].update({"on_train_result": tune.function(on_train_result_curriculum)})

    if args.imitate and not args.centralized_vf:
        from flow.agents.ImitationPPO import ImitationTrainer
        alg_run = ImitationTrainer
        run_name = "imitation_trainer"
    elif args.imitate and args.centralized_vf:
        alg_run = ImitationCentralizedTrainer
        run_name = "imitation_central_trainer"
    elif not args.imitate and args.centralized_vf:
        alg_run = CCTrainer
        run_name = "central_trainer"
    elif args.dqfd:
        alg_run = DQFDTrainer
        run_name = "dqfd"
    elif args.td3:
        alg_run = TD3Trainer
        run_name = "TD3"
    else:
        alg_run = CustomPPOTrainer
        run_name = "ppo_custom"

    config['env_config']['run'] = run_name

    exp_dict = {
            'name': args.exp_title,
            'run_or_experiment': alg_run,
            'checkpoint_freq': args.checkpoint_freq,
            'stop': {
                'training_iteration': args.num_iters
            },
            'trial_name_creator': trial_str_creator,
            'config': config,
            'num_samples': args.num_samples,
        }
    if args.use_s3:
        exp_dict['upload_dir'] = s3_string

    run(**exp_dict, queue_trials=False, raise_on_failed_trial=False)

    # Now we add code to loop through the results and create scores of the results
    if args.create_inflow_graph:
        output_path = os.path.join(os.path.join(os.path.expanduser('~/bottleneck_results'), date), args.exp_title)
        if not os.path.exists(output_path):
            try:
                os.makedirs(output_path)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
        for (dirpath, dirnames, filenames) in os.walk(os.path.expanduser("~/ray_results")):
            if "checkpoint_{}".format(args.checkpoint_freq) in dirpath and dirpath.split('/')[-3] == args.exp_title:
                # grab the experiment name
                folder = os.path.dirname(dirpath)
                tune_name = folder.split("/")[-1]
                checkpoint_path = os.path.dirname(dirpath)

                ray.shutdown()
                if args.local_mode:
                    ray.init(local_mode=True)
                else:
                    ray.init()

                run_bottleneck_results(400, 3600, 100, args.num_test_trials, output_path, args.exp_title, checkpoint_path,
                                       gen_emission=False, render_mode='no_render', checkpoint_num=str(700), #dirpath.split('_')[-1], TMP HARDCODED
                                       horizon=max(args.horizon, int(1000 / (args.sims_per_step * args.sim_step))), end_len=500)

                if args.use_s3:
                    for i in range(4):
                        try:
                            p1 = subprocess.Popen("aws s3 sync {} {}".format(output_path,
                                                                             "s3://nathan.experiments/trb_bottleneck_paper/graphs/{}/{}/{}".format(date,
                                                                                                                              args.exp_title,
                                                                                                                              tune_name)).split(
                                ' '))
                            p1.wait(5000)
                        except Exception as e:
                            print('This is the error ', e)
