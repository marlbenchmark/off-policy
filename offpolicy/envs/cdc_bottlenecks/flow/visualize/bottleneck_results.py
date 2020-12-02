"""Visualizer for rllib experiments specifically for the bottleneck.

Attributes
----------
EXAMPLE_USAGE : str
    Example call to the function, which is
    ::

        python ./visualizer_rllib.py /tmp/ray/checkpoint_dir 1

parser : ArgumentParser
    Command-line argument parser
"""

import argparse
import collections
from datetime import datetime
from functools import reduce
import os

import gym
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import ray
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation.episode import _flatten_action
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from flow.models.FeedForward import FeedForward
from flow.models.GRU import GRU
from flow.agents.ImitationPPO import ImitationTrainer
from flow.agents.centralized_PPO import CCTrainer
from flow.agents.centralized_imitation_PPO import ImitationCentralizedTrainer
from flow.agents.centralized_PPO import CentralizedCriticModel, CentralizedCriticModelRNN

from flow.utils.registry import make_create_env
from flow.utils.rllib import get_flow_params
from flow.utils.rllib import get_rllib_pkl


class DefaultMapping(collections.defaultdict):
    """default_factory now takes as an argument the missing key."""

    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value

def default_policy_agent_mapping(unused_agent_id):
    return DEFAULT_POLICY_ID

@ray.remote(memory=1500 * 1024 * 1024)
def run_bottleneck(checkpoint_dir, inflow_rate, num_trials, gen_emission, render_mode,
                   checkpoint_num,
                   end_len,
                   horizon=None):
    checkpoint_dir = checkpoint_dir if checkpoint_dir[-1] != '/' \
        else checkpoint_dir[:-1]

    config = get_rllib_pkl(checkpoint_dir)

    # Run on only one cpu for rendering purposes
    config['num_workers'] = 0

    flow_params = get_flow_params(config)

    # Determine agent and checkpoint
    config_run = config['env_config']['run']
    if config_run == "imitation_trainer":
        from flow.agents.ImitationPPO import ImitationTrainer
        agent_cls = ImitationTrainer
    elif config_run == "imitation_central_trainer":
        agent_cls = ImitationCentralizedTrainer
    elif config_run == "central_trainer":
        from flow.agents.centralized_PPO import CCTrainer
        agent_cls = CCTrainer
    elif config_run == "dqfd":
        from flow.agents.DQfD import DQFDTrainer
        agent_cls = DQFDTrainer
    elif config_run == "ppo_custom":
        from flow.agents.custom_ppo import CustomPPOTrainer
        agent_cls = CustomPPOTrainer
    elif config_run == 'TD3':
        from ray.rllib.agents.ddpg.td3 import TD3Trainer
        agent_cls = TD3Trainer
    else:
        agent_cls = get_agent_class(config_run)

    # if using a custom model
    if config['model']['custom_model'] == "cc_model":
        if config['model']['use_lstm']:
            ModelCatalog.register_custom_model("cc_model", CentralizedCriticModelRNN)
        else:
            ModelCatalog.register_custom_model("cc_model", CentralizedCriticModel)
        from flow.agents.centralized_PPO import CCTrainer
        agent_cls = CCTrainer
    elif config['model']['custom_model'] == "GRU":
        ModelCatalog.register_custom_model("GRU", GRU)
    elif config['model']['custom_model'] == "FeedForward":
        ModelCatalog.register_custom_model("FeedForward", FeedForward)

    sim_params = flow_params['sim']
    sim_params.restart_instance = True
    dir_path = os.path.dirname(os.path.realpath(__file__))
    emission_path = '{0}/test_time_rollout/'.format(dir_path)
    sim_params.emission_path = emission_path if gen_emission else None

    # pick your rendering mode
    if render_mode == 'sumo_web3d':
        sim_params.num_clients = 2
        sim_params.render = False
    elif render_mode == 'drgb':
        sim_params.render = 'drgb'
        sim_params.pxpm = 4
    elif render_mode == 'sumo_gui':
        sim_params.render = True
    elif render_mode == 'no_render':
        sim_params.render = False

    # Start the environment with the gui turned on and a path for the
    # emission file
    env_params = flow_params['env']
    # # TODO(@evinitsky) remove this this is a backwards compatibility hack
    # if 'life_penalty' not in env_params.additional_params.keys():
    #     env_params.additional_params['life_penalty'] = - 3
    # if args.evaluate:
    #     env_params.evaluate = True

    # lower the horizon if testing
    if horizon:
        config['horizon'] = horizon
        env_params.horizon = horizon

    # Create and register a gym+rllib env
    create_env, env_name = make_create_env(params=flow_params, version=0)
    register_env(env_name, create_env)

    # create the agent that will be used to compute the actions
    agent = agent_cls(env=env_name, config=config)
    checkpoint = checkpoint_dir + '/checkpoint_' + checkpoint_num
    checkpoint = checkpoint + '/checkpoint-' + checkpoint_num
    agent.restore(checkpoint)

    policy_agent_mapping = default_policy_agent_mapping
    if hasattr(agent, "workers"):
        env = agent.workers.local_worker().env
        multiagent = isinstance(env, MultiAgentEnv)
        if agent.workers.local_worker().multiagent:
            policy_agent_mapping = agent.config["multiagent"][
                "policy_mapping_fn"]

        policy_map = agent.workers.local_worker().policy_map
        state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
        use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
        action_init = {
            p: m.action_space.sample()
            for p, m in policy_map.items()
        }
    else:
        env = gym.make(env_name)
        multiagent = False
        use_lstm = {DEFAULT_POLICY_ID: False}

    if render_mode == 'sumo_gui':
        env.sim_params.render = True  # set to True after initializing agent and env

    # Simulate and collect metrics
    outflow_arr = []
    final_outflows = []
    final_inflows = []
    mean_speed = []
    std_speed = []
    mean_rewards = []
    per_agent_rew = collections.defaultdict(lambda: 0.0)

    # keep track of the last 500 seconds of velocity data for lane 0
    # and 1 in edge 4
    velocity_arr = []
    vel = []
    mapping_cache = {}  # in case policy_agent_mapping is stochastic

    sim_step = sim_params.sim_step
    env.env_params.horizon = 2000
    env_params.sims_per_step = 1
    env.env_params.warmup_steps = 0
    if hasattr(env, 'reroute_on_exit'):
        env.reroute_on_exit = False

    for j in range(num_trials):
        agent_states = DefaultMapping(
            lambda agent_id: state_init[mapping_cache[agent_id]])
        prev_actions = DefaultMapping(
            lambda agent_id: action_init[mapping_cache[agent_id]])
        prev_rewards = collections.defaultdict(lambda: 0.0)
        done = False
        reward_total = 0.0
        obs = env.reset(inflow_rate)
        k = 0
        while k < env_params.horizon and not done:
            vehicles = env.unwrapped.k.vehicle
            vel.append(np.mean(vehicles.get_speed(vehicles.get_ids())))
            # don't start recording till we have hit the warmup time
            if k >= env_params.horizon - (end_len / sim_step):
                vehs_on_four = vehicles.get_ids_by_edge('4')
                lanes = vehicles.get_lane(vehs_on_four)
                lane_dict = {veh_id: lane for veh_id, lane in
                             zip(vehs_on_four, lanes)}
                sort_by_lane = sorted(vehs_on_four,
                                      key=lambda x: lane_dict[x])
                num_zeros = lanes.count(0)
                if num_zeros > 0:
                    speed_on_zero = np.mean(vehicles.get_speed(
                        sort_by_lane[0:num_zeros]))
                else:
                    speed_on_zero = 0.0
                if num_zeros < len(vehs_on_four):
                    speed_on_one = np.mean(vehicles.get_speed(
                        sort_by_lane[num_zeros:]))
                else:
                    speed_on_one = 0.0
                velocity_arr.append(
                    [inflow_rate,
                     speed_on_zero,
                     speed_on_one])
            multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
            action_dict = {}
            for agent_id, a_obs in multi_obs.items():
                if a_obs is not None:
                    policy_id = mapping_cache.setdefault(
                        agent_id, policy_agent_mapping(agent_id))
                    p_use_lstm = use_lstm[policy_id]
                    if p_use_lstm:
                        a_action, p_state, _ = agent.compute_action(
                            a_obs,
                            state=agent_states[agent_id],
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                        agent_states[agent_id] = p_state
                    else:
                        a_action = agent.compute_action(
                            a_obs,
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                    a_action = _flatten_action(a_action)  # tuple actions
                    action_dict[agent_id] = a_action
                    prev_actions[agent_id] = a_action
            action = action_dict

            action = action if multiagent else action[_DUMMY_AGENT_ID]
            # print(action)
            next_obs, reward, done, _ = env.step(action)

            if multiagent:
                for agent_id, r in reward.items():
                    prev_rewards[agent_id] = r
                    per_agent_rew[agent_id] += r
            else:
                prev_rewards[_DUMMY_AGENT_ID] = reward

            if multiagent:
                done = done["__all__"]
                reward_total += sum(reward.values())
            else:
                reward_total += reward
            k += 1
            obs = next_obs

        vehicles = env.unwrapped.k.vehicle
        outflow = vehicles.get_outflow_rate(500)
        print('Outflow was ', outflow)
        final_outflows.append(outflow)
        inflow = vehicles.get_inflow_rate(500)
        final_inflows.append(inflow)
        outflow_arr.append([inflow_rate, outflow, outflow/inflow_rate])
        mean_speed.append(np.mean(vel))
        std_speed.append(np.std(vel))
        mean_rewards.append([inflow, np.mean(list(per_agent_rew.values()))])
    return [outflow_arr, velocity_arr, mean_rewards]


def create_parser():
    """Create the parser to capture CLI arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Evaluates a bottleneck agent over a grid of inflows')

    # required input parameters
    parser.add_argument(
        'checkpoint_dir', type=str, help='Directory containing results')
    parser.add_argument('checkpoint_num', type=str, help='Checkpoint number.')
    parser.add_argument('filename', type=str, help='Specifies the filename to output the results into.')

    # optional input parameters
    parser.add_argument(
        '--num_rollouts',
        type=int,
        default=1,
        help='The number of rollouts to visualize.')
    parser.add_argument(
        '--gen_emission',
        action='store_true',
        help='Specifies whether to generate an emission file from the '
             'simulation')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Specifies whether to use the \'evaluate\' reward '
             'for the environment.')
    parser.add_argument(
        '--render_mode',
        type=str,
        default='no_render',
        help='Pick the render mode. Options include sumo_web3d, '
             'rgbd, no_render, and sumo_gui')
    parser.add_argument(
        '--save_render',
        action='store_true',
        help='Saves a rendered video to a file. NOTE: Overrides render_mode '
             'with pyglet rendering.')
    parser.add_argument(
        '--horizon',
        type=int,
        help='Specifies the horizon.')
    parser.add_argument('--local_mode', action='store_true', default=False)
    parser.add_argument('--num_cpus', type=int, default=1, help='how many cpus to run ray with')
    parser.add_argument('--outflow_min', type=int, default=400, help='Lowest inflow to evaluate over')
    parser.add_argument('--outflow_max', type=int, default=2500, help='Lowest inflow to evaluate over')
    parser.add_argument('--step_size', type=int, default=100, help='The size of increments to sweep over inflow in')
    parser.add_argument('--num_trials', type=int, default=20, help='How many samples of each inflow to take')
    parser.add_argument('--end_len', type=int, default=500, help='How many last seconds of the run to use for '
                                                                 'calculating the outflow statistics')
    parser.add_argument('--cluster_mode', action='store_true', help='Specifies if we run it on a cluster')
    parser.add_argument('--output_path', type=str, default=os.getcwd())

    return parser


def run_bottleneck_results(outflow_min, outflow_max, step_size, num_trials, output_path, filename,
                           checkpoint_dir, gen_emission, render_mode, checkpoint_num,
                           horizon=None,
                           end_len=500):
    inflow_grid = list(range(outflow_min, outflow_max + step_size,
                             step_size))
    temp_output = [run_bottleneck.remote(checkpoint_dir, inflow, num_trials, gen_emission,
                                         render_mode, checkpoint_num, end_len, horizon) for inflow in inflow_grid]
    final_output = ray.get(temp_output)

    outflow_arr = np.asarray([elem[0] for elem in final_output])
    outflow_arr = np.reshape(outflow_arr, (-1, outflow_arr.shape[-1]))
    velocity_arr = reduce(lambda x, y: x + y, [elem[1] for elem in final_output])
    final_rewards = np.asarray(reduce(lambda x, y: x + y, [elem[2] for elem in final_output]))

    # save the file
    output_path = os.path.abspath(output_path)
    output_path = os.path.join(output_path, datetime.now().strftime("%m-%d-%Y"))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        os.makedirs(os.path.join(output_path, 'figures'))
    outflow_name = 'bottleneck_outflow_{}.txt'.format(filename)
    speed_name = 'speed_outflow_{}.txt'.format(filename)
    rewards_name = 'rewards_{}.txt'.format(filename)
    with open(os.path.join(output_path, outflow_name), 'ab') as file:
        np.savetxt(file, outflow_arr, delimiter=', ')
    with open(os.path.join(output_path, speed_name), 'ab') as file:
        np.savetxt(file, velocity_arr, delimiter=', ')
    with open(os.path.join(output_path, rewards_name), 'ab') as file:
        np.savetxt(file, final_rewards, delimiter=', ')

    # Plot the inflow results
    # open the file and pull from there
    unique_inflows = sorted(list(set(outflow_arr[:, 0])))
    inflows = outflow_arr[:, 0]
    outflows = outflow_arr[:, 1]
    throughputs = outflow_arr[:, 2]
    sorted_outflows = {inflow: [] for inflow in unique_inflows}
    sorted_throughputs = {inflow: [] for inflow in unique_inflows}
    sorted_rewards = {inflow: [] for inflow in unique_inflows}

    for inflow, outflow, throughput, final_reward in zip(inflows, outflows, throughputs, final_rewards[:, 1]):
        sorted_outflows[inflow].append(outflow)
        sorted_throughputs[inflow].append(throughput)
        sorted_rewards[inflow].append(final_reward)
    mean_outflows = np.asarray([np.mean(sorted_outflows[inflow])
                                for inflow in unique_inflows])
    std_outflows = np.asarray([np.std(sorted_outflows[inflow])
                               for inflow in unique_inflows])
    mean_throughputs = np.asarray([np.mean(sorted_throughputs[inflow])
                                   for inflow in unique_inflows])
    std_throughputs = np.asarray([np.std(sorted_throughputs[inflow])
                                  for inflow in unique_inflows])

    mean_rewards = np.asarray([np.mean(sorted_rewards[inflow])
                               for inflow in unique_inflows])
    std_rewards = np.asarray([np.std(sorted_rewards[inflow])
                              for inflow in unique_inflows])

    plt.figure(figsize=(27, 9))
    plt.plot(unique_inflows, mean_outflows, linewidth=2, color='orange')
    plt.fill_between(unique_inflows, mean_outflows - std_outflows,
                     mean_outflows + std_outflows, alpha=0.25, color='orange')
    plt.xlabel('Inflow' + r'$ \ \frac{vehs}{hour}$')
    plt.ylabel('Outflow' + r'$ \ \frac{vehs}{hour}$')
    plt.tick_params(labelsize=20)
    plt.rcParams['xtick.minor.size'] = 20
    plt.minorticks_on()
    plt.savefig(os.path.join(output_path, 'figures/outflow_{}'.format(filename)) + '.png')

    # plot the velocity results
    velocity_arr = np.asarray(velocity_arr)
    unique_inflows = sorted(list(set(velocity_arr[:, 0])))
    inflows = velocity_arr[:, 0]
    lane_0 = velocity_arr[:, 1]
    lane_1 = velocity_arr[:, 2]
    sorted_vels = {inflow: [] for inflow in unique_inflows}

    for inflow, vel_0, vel_1 in zip(inflows, lane_0, lane_1):
        sorted_vels[inflow] += [vel_0, vel_1]
    mean_vels = np.asarray([np.mean(sorted_vels[inflow])
                            for inflow in unique_inflows])
    std_vels = np.asarray([np.std(sorted_vels[inflow])
                           for inflow in unique_inflows])

    plt.figure(figsize=(27, 9))

    plt.plot(unique_inflows, mean_vels, linewidth=2, color='orange')
    plt.fill_between(unique_inflows, mean_vels - std_vels,
                     mean_vels + std_vels, alpha=0.25, color='orange')
    plt.xlabel('Inflow' + r'$ \ \frac{vehs}{hour}$')
    plt.ylabel('Velocity' + r'$ \ \frac{m}{s}$')
    plt.tick_params(labelsize=20)
    plt.rcParams['xtick.minor.size'] = 20
    plt.minorticks_on()
    plt.savefig(os.path.join(output_path, 'figures/speed_{}'.format(filename)) + '.png')

    # Plot the throughput results
    plt.figure(figsize=(27, 9))
    plt.plot(unique_inflows, mean_throughputs, linewidth=2, color='orange')
    plt.fill_between(unique_inflows, mean_throughputs - std_throughputs,
                     mean_throughputs + std_throughputs, alpha=0.25, color='orange')
    plt.xlabel('Inflow' + r'$ \ \frac{vehs}{hour}$')
    plt.ylabel('Throughput' + r'$ \ \frac{vehs}{hour}$')
    plt.tick_params(labelsize=20)
    plt.rcParams['xtick.minor.size'] = 20
    plt.minorticks_on()
    plt.savefig(os.path.join(output_path, 'figures/throughput_{}'.format(filename)) + '.png')

    # Plot the throughput results
    plt.figure(figsize=(27, 9))
    plt.plot(unique_inflows, mean_rewards, linewidth=2, color='orange')
    plt.fill_between(unique_inflows, mean_rewards - std_rewards,
                     mean_rewards + std_rewards, alpha=0.25, color='orange')
    plt.xlabel('Inflow' + r'$ \ \frac{vehs}{hour}$')
    plt.ylabel('Mean per-AV Rewards' + r'$ \ \frac{vehs}{hour}$')
    plt.tick_params(labelsize=20)
    plt.rcParams['xtick.minor.size'] = 20
    plt.minorticks_on()
    plt.savefig(os.path.join(output_path, 'figures/rewards_{}'.format(filename)) + '.png')


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    if args.local_mode:
        ray.init(num_cpus=args.num_cpus, local_mode=True)
    else:
        ray.init(num_cpus=args.num_cpus)
    horizon = None
    if args.horizon:
        horizon = args.horizon
    run_bottleneck_results(args.outflow_min, args.outflow_max, args.step_size, args.num_trials,
                           args.output_path, args.filename,
                           args.checkpoint_dir, args.gen_emission, args.render_mode, args.checkpoint_num,
                           end_len=args.end_len,
                           horizon=horizon,
                           )