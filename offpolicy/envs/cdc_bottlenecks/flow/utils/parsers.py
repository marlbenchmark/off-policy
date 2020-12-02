import argparse

def get_multiagent_bottleneck_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Parses command line args for multi-agent bottleneck exps')
    
    # required input parameters for tune
    parser.add_argument('exp_title', type=str, help='Informative experiment title to help distinguish results')
    parser.add_argument('--use_s3', action='store_true', help='If true, upload results to s3')
    parser.add_argument('--n_cpus', type=int, default=1, help='Number of cpus to run experiment with')
    parser.add_argument('--multi_node', action='store_true', help='Set to true if this will '
                                                                  'be run in cluster mode')
    parser.add_argument('--local_mode', action='store_true', default=False,
                        help='If true only 1 CPU will be used')
    parser.add_argument("--num_iters", type=int, default=350)
    parser.add_argument("--checkpoint_freq", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--grid_search", action='store_true')
    parser.add_argument('--rollout_scale_factor', type=float, default=1.0, help='the total number of rollouts is'
                                                                                'args.n_cpus * rollout_scale_factor')
    parser.add_argument("--vf_loss_coeff", type=float, default=.0001, help='coeff of the vf loss')
    parser.add_argument("--entropy_coeff", type=float, default=0.0)
    
    # arguments for flow
    parser.add_argument('--sims_per_step', type=int, default=1, help='How many steps to take per action')
    parser.add_argument('--render', action='store_true', help='Show sumo-gui of results')
    parser.add_argument('--horizon', type=int, default=1000, help='Horizon of the environment')
    parser.add_argument('--sim_step', type=float, default=0.5, help='dt of a timestep')
    parser.add_argument('--low_inflow', type=int, default=800, help='the lowest inflow to sample from')
    parser.add_argument('--high_inflow', type=int, default=2200, help='the highest inflow to sample from')
    parser.add_argument('--av_frac', type=float, default=0.1, help='What fraction of the vehicles should be autonomous')
    parser.add_argument('--scaling', type=int, default=1, help='How many lane should we start with. Value of 1 -> 4, '
                                                               '2 -> 8, etc.')
    parser.add_argument('--lc_on', action='store_true', help='If true, lane changing is enabled.')
    parser.add_argument('--congest_penalty', action='store_true', help='If true, an additional penalty is added '
                                                                       'for vehicles queueing in the bottleneck')
    parser.add_argument('--communicate', action='store_true', help='If true, the agents have an additional action '
                                                                   'which consists of sending a discrete signal '
                                                                   'to all nearby vehicles')
    parser.add_argument('--central_obs', action='store_true', help='If true, all agents receive the same '
                                                                   'aggregate statistics')
    parser.add_argument('--aggregate_info', action='store_true', help='If true, agents receive some '
                                                                      'centralized info')
    parser.add_argument('--congest_penalty_start', type=int, default=30, help='If congest_penalty is true, this '
                                                                              'sets the number of vehicles in edge 4'
                                                                              'at which the penalty sets in')
    parser.add_argument('--life_penalty', type=float, default=0, help='How much to subtract in the reward at each '
                                                                     'time-step for remaining in the system.')
    parser.add_argument('--keep_past_actions', action='store_true', help='If true we append the agents past actions '
                                                                         'to its observations')
    parser.add_argument('--num_sample_seconds', type=float, default=0.5,
                        help='How many seconds back in time the outflow reward should sample over. It defaults to '
                             'only looking at the current step')
    parser.add_argument('--speed_reward', action='store_true', default=False,
                        help='If true the reward is the mean AV speed. If not set the reward is outflow')
    parser.add_argument('--imitate', action='store_true', default=False,
                        help='If true, the first 30 iterations are supervised learning on imitation of an IDM vehicle')
    parser.add_argument('--final_imitation_weight', type=float, default=0.0,
                        help='This is the value we decrease the imitation weight to after imitation is done.'
                             'Make it non-zero to prevent the policy from totally straying from the imitation weight.')
    parser.add_argument('--num_imitation_iters', type=int, default=50,
                        help='Number of iterations to imitate controller. must enable --imitate')
    parser.add_argument('--hard_negative_mining', action='store_true', default=False,
                        help='Use only the top 10 percent actions to imitate')
    parser.add_argument('--mining_frac', type=float, default=0.1,
                        help='The percentage of top scores to imitate on. .10 will imitate on the top 10%')
    parser.add_argument('--simple_env', action='store_true', default=False,
                        help='If true, the imitation env mimics the observations available to the imitated controllers')
    parser.add_argument('--super_simple_env', action='store_true', default=False,
                        help='If true, the imitation env mimics the observations available to the imitated controllers')
    parser.add_argument('--centralized_vf', action='store_true', default=False,
                        help='If true, use a centralized value function')
    parser.add_argument('--central_vf_size', type=int, default=64, help='The number of hidden units in '
                                                                        'the value function')
    parser.add_argument('--max_num_agents', type=int, default=120, help='The maximum number of agents we could ever have')
    parser.add_argument('--terminal_reward', action='store_true', default=False)
    parser.add_argument('--post_exit_rew_len', type=int, default=200, help='This is how much of the future outflow '
                                                                           'is given to the agent as a terminal reward')
    parser.add_argument('--rew_n_crit', type=int, default=0, help='If set to a value above zero, we get rewarded if fewer than '
                                                              'n_crit AVs in the bottleneck, and penalized if above')
    parser.add_argument('--reroute_on_exit', action='store_true', default=False,
                        help='Put back RL vehicles that have left')
    parser.add_argument('--warmup_fixed_agents', action='store_true', default=False, help="Whether to warmup until a specified # agents are in the system")
    parser.add_argument('--num_agents', type=int, default=-1, help="If warmup_fixed_agents True, how many agents should be in system")

    # dqfd arguments
    parser.add_argument('--dqfd', action='store_true', default=False,
                        help='Whether to use dqfd')
    parser.add_argument('--num_expert_steps', type=int, default=5e4, help='How many steps to let the expert take'
                                                                          'before switching back to the actor')
    parser.add_argument('--fingerprinting', action='store_true', default=False,
                        help='Whether to add the iteration number to the inputs')

    # TD3 arguments
    parser.add_argument('--td3', action='store_true', default=False,
                        help='Whether to use td3')

    # QMIX arguments
    parser.add_argument('--qmix', action='store_true', default=False,
                        help='Whether to use qmix')
    parser.add_argument('--order_agents', action='store_true', default=False,
                        help='If true, the agents are sorted by absolute position before being passed'
                             'to the mixer')

    # Curriculum stuff
    parser.add_argument('--curriculum', action='store_true', help='If true, anneal the av_frac and inflow over '
                                                                  'num_curr_iter steps')
    parser.add_argument("--num_curr_iters", type=int, default=100, help='How many steps to run curriculum over')
    parser.add_argument("--min_horizon", type=int, default=200, help='How many steps to run curriculum over')

    # arguments for ray
    parser.add_argument('--use_lstm', action='store_true')
    parser.add_argument('--use_gru', action='store_true')

    # arguments about output
    parser.add_argument('--create_inflow_graph', action='store_true', default=False)
    parser.add_argument('--num_test_trials', type=int, default=20)

    # TD3 hyperparams (for doing faster seed search)
    parser.add_argument('--td3_prioritized_replay', action='store_true', default=False)
    parser.add_argument('--td3_actor_lr', type=float, default=0.001)
    parser.add_argument('--td3_critic_lr', type=float, default=0.001)
    parser.add_argument('--td3_n_step', type=int, default=5)

    return parser