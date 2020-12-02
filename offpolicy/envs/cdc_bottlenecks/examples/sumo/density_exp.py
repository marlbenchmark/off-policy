"""Bottleneck runner script for generating flow-density plots.

Run density experiment to generate capacity diagram for the
bottleneck experiment
"""

import argparse
import multiprocessing
import numpy as np
import os
import ray

from examples.sumo.bottlenecks import bottleneck_example


@ray.remote
def run_bottleneck(flow_rate, penetration_rate, num_trials, num_steps, render=None, disable_ramp_meter=True, n_crit=8,
                   feedback_coef=20, lc_on=False, q_init=400):
    """Run a rollout of the bottleneck environment.

    Parameters
    ----------
    flow_rate : float
        bottleneck inflow rate
    num_trials : int
        number of rollouts to perform
    num_steps : int
        number of simulation steps per rollout
    render : bool
        whether to render the environment

    Returns
    -------
    float
        average outflow rate across rollouts
    float
        average speed across rollouts
    float
        average rollout density outflow
    list of float
        per rollout outflows
    float
        inflow rate
    """
    print('Running experiment for inflow rate: ', flow_rate,
          render, q_init, penetration_rate, feedback_coef)
    exp = bottleneck_example(flow_rate, num_steps, render=render, restart_instance=True,
                             disable_ramp_meter=disable_ramp_meter,
                             feedback_coef=feedback_coef, n_crit=n_crit, lc_on=lc_on, q_init=q_init, penetration_rate=penetration_rate)
    info_dict = exp.run(num_trials, num_steps)

    return info_dict['average_outflow'], \
        info_dict['velocities'], \
        np.mean(info_dict['average_rollout_density_outflow']), \
        info_dict['per_rollout_outflows'], \
        flow_rate, info_dict['lane_4_vels'],


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Runs the bottleneck exps and stores the results for processing')
    parser.add_argument('--render', action='store_true',
                        help='Display the scenarios')

    parser.add_argument('--ramp_meter', action='store_true',
                        help='If set, ALINEA is active in this scenario')
    parser.add_argument('--alinea_sweep', action='store_true', help='If set, perform a hyperparam sweep over ALINEA '
                                                                    'hyperparams')
    parser.add_argument('--decentralized_alinea_sweep', action='store_true', help='If set, perform a hyperparam sweep'
                        ' over hyperparams for decentralized ALINEA controller')
    parser.add_argument('--penetration_rate', type=float,
                        help='percentage of AVs in the system', default=0.0)
    parser.add_argument('--inflow_min', type=int, default=400)
    parser.add_argument('--inflow_max', type=int, default=2500)
    parser.add_argument('--ncrit_min', type=int, default=6)
    parser.add_argument('--ncrit_max', type=int, default=12)
    parser.add_argument('--ncrit_step_size', type=int, default=1)
    parser.add_argument('--step_size', type=int, default=100)
    parser.add_argument('--num_trials', type=int, default=20)
    parser.add_argument('--horizon', type=int, default=2000)
    parser.add_argument('--opt_feedback_coef', type=int, help="feedback coef for a single sweep")
    parser.add_argument('--opt_n_crit', type=int, help="n_crit for a single sweep")
    parser.add_argument('--opt_q_init', type=int, help="q_init for a single sweep")
    parser.add_argument('--lc_on', action='store_true')
    parser.add_argument('--multi_node', action='store_true')
    parser.add_argument('--clear_data', action='store_true', help='If true, clean the folder where the files are '
                                                                  'stored before running anything')
    parser.add_argument('--test_run', action='store_true',
                        help='If true, sweep over a tiny grid')
    args = parser.parse_args()

    if args.alinea_sweep:
        assert args.ramp_meter, \
            "If alinea sweep is on, the ramp meter must be on as well, or a fully human sweep"
    elif args.ramp_meter:
        assert args.penetration_rate == 0.0, \
            "Shouldn't have AVs and ramp meter!"

    path = os.path.dirname(os.path.abspath(__file__))
    if args.alinea_sweep:
        if args.test_run:
            outer_path = '../../flow/visualize/trb_data/alinea_test'
        else:
            outer_path = '../../flow/visualize/trb_data/alinea_data'
    if args.decentralized_alinea_sweep:
        if args.test_run:
            outer_path = '../../flow/visualize/trb_data/decentraliezd_alinea_test'
        else:
            outer_path = '../../flow/visualize/trb_data/decentraliezd_alinea_data'
    else:
        outer_path = '../../flow/visualize/trb_data/human_driving'

    if args.clear_data:
        for the_file in os.listdir(os.path.join(path, outer_path)):
            file_path = os.path.join(os.path.join(path, outer_path), the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                # elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)

    # list(range(args.ncrit_min, args.ncrit_max + args.ncrit_step_size, args.ncrit_step_size))
    # TODO(kp): make this work for both alinea and decentralized alinea
    n_crit_range = [6, 8, 10]
    feedback_coef_range = [1, 5, 10, 20, 50]
    q_init_range = [200, 600, 1000, 5000, 10000]
    densities = list(range(args.inflow_min, args.inflow_max +
                           args.step_size, args.step_size))
    if args.test_run:
        args.num_trials = 2
        densities = [3000]
        n_crit_range = [7, 8]
        feedback_coef_range = [5]
        q_init_range = [500, 100]

    num_cpus = multiprocessing.cpu_count()
    if args.multi_node:
        ray.init(redis_address="localhost:6379")
    else:
        ray.init()
    if args.alinea_sweep or (args.decentralized_alinea_sweep and args.penetration_rate != 0):
        bottleneck_outputs = []
        hyperparams = []
        for n_crit in n_crit_range:
            for q_init in q_init_range:
                for feedback_coef in feedback_coef_range:
                    bottleneck_outputs.extend([run_bottleneck.remote(flow_rate=d, penetration_rate=args.penetration_rate,
                                                                     num_trials=args.num_trials, num_steps=args.horizon, 
                                                                     render=args.render,
                                                                     disable_ramp_meter=not args.ramp_meter,
                                                                     lc_on=args.lc_on,
                                                                     feedback_coef=feedback_coef, n_crit=n_crit, q_init=q_init,)
                                               for d in densities])
                    hyperparams.append((n_crit, q_init, feedback_coef))
        print(bottleneck_outputs)
        all_outputs = ray.get(bottleneck_outputs)
        output_sets = [all_outputs[i * len(densities): (i+1) * len(densities)]
                       for i in range(int(len(all_outputs) / len(densities)))]
        print(len(output_sets))
        print(hyperparams)
        assert len(output_sets) == len(hyperparams)
        for params, output_set in zip(hyperparams, output_sets):
            outflows = []
            velocities = []
            lane_4_vels = []
            bottleneckdensities = []

            rollout_inflows = []
            rollout_outflows = []
            percent_congested = []
            for output in output_set:
                outflow, per_step_velocities, bottleneckdensity, \
                    per_rollout_outflows, flow_rate, lane_4_vel = output
                for i, _ in enumerate(per_rollout_outflows):
                    rollout_outflows.append(per_rollout_outflows[i])
                    rollout_inflows.append(flow_rate)
                outflows.append(outflow)
                velocities.append(np.mean(per_step_velocities))
                congested = 0
                for rollout_per_step_vels in per_step_velocities:
                    # cars in bottleneck, velocity less than 5
                    congested += np.sum(np.logical_and(
                        rollout_per_step_vels != -1, rollout_per_step_vels < 5))
                percent_congested.append(congested / args.horizon)
                lane_4_vels += lane_4_vel
                bottleneckdensities.append(bottleneckdensity)
            n_crit, q_init, feedback_coef = params
            # save the returns
            if args.lc_on:
                ret_string = 'rets_LC_n{}_fcoeff{}_qinit{}_alinea.csv'.format(
                    n_crit, feedback_coef, q_init)
                inflow_outflow_str = 'inflows_outflows_LC_n{}_fcoeff{}_qinit{}_alinea.csv'.format(
                    n_crit, feedback_coef, q_init)
                inflow_velocity_str = 'inflows_velocity_LC_n{}_fcoeff{}_qinit{}_alinea.csv'.format(
                    n_crit, feedback_coef, q_init)

            else:
                ret_string = 'rets_n{}_fcoeff{}_qinit{}_alinea.csv'.format(
                    n_crit, feedback_coef, q_init)
                inflow_outflow_str = 'inflows_outflows_n{}_fcoeff{}_qinit{}_alinea.csv'.format(
                    n_crit, feedback_coef, q_init)
                inflow_velocity_str = 'inflows_velocity_n{}_fcoeff{}_qinit{}_alinea.csv'.format(
                    n_crit, feedback_coef, q_init)

            ret_path = os.path.join(path, os.path.join(outer_path, ret_string))
            outflow_path = os.path.join(
                path, os.path.join(outer_path, inflow_outflow_str))
            vel_path = os.path.join(path, os.path.join(
                outer_path, inflow_velocity_str))

            with open(ret_path, 'wb') as file:
                np.savetxt(file, np.matrix(
                    [densities, outflows, velocities, percent_congested, bottleneckdensities]).T, delimiter=',')
            with open(outflow_path, 'wb') as file:
                np.savetxt(file,  np.matrix(
                    [rollout_inflows, rollout_outflows]).T, delimiter=',')

    else:
        outflows = []
        velocities = []
        lane_4_vels = []
        bottleneckdensities = []

        rollout_inflows = []
        rollout_outflows = []

        if args.penetration_rate == 0.0:
            print("Penetration rate is 0.0, running human curve exp.")
        # TODO(kp): add q_init here too
        if args.ramp_meter or args.penetration_rate:
            assert args.opt_feedback_coef, "feedback coef must be defined to run sweep"
            assert args.opt_n_crit, "n_crit must be defined to run sweep"
            if args.penetration_rate:
                assert args.opt_q_init, "q_init must be defined to run sweep w/ AVs"
            else:
                args.opt_q_init = 0 # set some default q_init even though there aren't AVs
        
        bottleneck_outputs = [run_bottleneck.remote(flow_rate=d, penetration_rate=args.penetration_rate,
                                                    num_trials=args.num_trials, num_steps=args.horizon,
                                                    render=args.render,
                                                    disable_ramp_meter=not args.ramp_meter,
                                                    lc_on=args.lc_on,
                                                    feedback_coef=args.opt_feedback_coef,
                                                    n_crit=args.opt_n_crit,
                                                    q_init=args.opt_q_init)
                              for d in densities]
        for output in ray.get(bottleneck_outputs):
            outflow, velocity, bottleneckdensity, \
                per_rollout_outflows, flow_rate, lane_4_vel = output
            for i, _ in enumerate(per_rollout_outflows):
                rollout_outflows.append(per_rollout_outflows[i])
                rollout_inflows.append(flow_rate)
            outflows.append(outflow)
            velocities.append(velocity)
            lane_4_vels += lane_4_vel
            bottleneckdensities.append(bottleneckdensity)

        path = os.path.dirname(os.path.abspath(__file__))

        if args.lc_on:
            np.savetxt(path + '/../../flow/visualize/trb_data/human_driving/rets_LC.csv',
                       np.matrix([densities,
                                  outflows,
                                  [np.mean([np.sum(np.logical_and(velocity != -1, velocity < 5))
                                            for velocity in velocities_at_d]) / args.horizon for velocities_at_d in velocities],
                                  bottleneckdensities]).T,
                       delimiter=',')
            np.savetxt(path + '/../../flow/visualize/trb_data/human_driving/inflows_outflows_LC.csv',
                       np.matrix([rollout_inflows,
                                  rollout_outflows]).T,
                       delimiter=',')
            np.savetxt(path + '/../../flow/visualize/trb_data/human_driving/inflows_velocity_LC.csv',
                       np.matrix(lane_4_vels),
                       delimiter=',')
        else:
            np.savetxt(path + '/../../flow/visualize/trb_data/human_driving/rets.csv',
                       np.matrix([densities,
                                  outflows,
                                  [np.mean([np.sum(np.logical_and(velocity != -1, velocity < 5))
                                            for velocity in velocities_at_d]) / args.horizon for velocities_at_d in velocities],
                                  bottleneckdensities]).T,
                       delimiter=',')
            np.savetxt(path + '/../../flow/visualize/trb_data/human_driving/inflows_outflows.csv',
                       np.matrix([rollout_inflows,
                                  rollout_outflows]).T,
                       delimiter=',')
            np.savetxt(path + '/../../flow/visualize/trb_data/human_driving/inflows_velocity.csv',
                       np.matrix(lane_4_vels),
                       delimiter=',')
