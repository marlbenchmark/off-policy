from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
"""An example of customizing PPO to leverage a centralized critic.
Here the model and policy are hard-coded to implement a centralized critic
for TwoStepGame, but you can adapt this for your own use cases.
Compared to simply running `twostep_game.py --run=PPO`, this centralized
critic version reaches vf_explained_variance=1.0 more stably since it takes
into account the opponent actions as well as the policy's. Note that this is
also using two independent policies instead of weight-sharing with one.
See also: centralized_critic_2.py for a simpler approach that instead
modifies the environment.
"""

from datetime import datetime

import numpy as np
import pytz
import ray
from ray import tune
from ray.rllib.models import ModelCatalog

from examples.rllib.multiagent_exps.multiagent_bottleneck import setup_exps, on_episode_end
from flow.agents.centralized_PPO import CentralizedCriticModel, CentralizedCriticModelRNN
from flow.agents.centralized_PPO import CCTrainer
from flow.utils.parsers import get_multiagent_bottleneck_parser

if __name__ == "__main__":

    parser = get_multiagent_bottleneck_parser()
    args = parser.parse_args()

    alg_run, env_name, config = setup_exps(args)

    # set up LSTM
    config['model']['use_lstm'] = args.use_lstm
    if args.use_lstm:
        config['model']["max_seq_len"] = tune.grid_search([5, 10])
        config['model']["lstm_cell_size"] = 32

    # Set up model
    if args.use_lstm:
        ModelCatalog.register_custom_model("cc_model", CentralizedCriticModelRNN)
    else:
        ModelCatalog.register_custom_model("cc_model", CentralizedCriticModel)
    config['model']['custom_model'] = "cc_model"
    config['model']['custom_options']['central_vf_size'] = args.central_vf_size

    # store custom metrics
    config["callbacks"] = {"on_episode_end": tune.function(on_episode_end)}

    if args.multi_node:
        ray.init(redis_address='localhost:6379')
    else:
        ray.init()
    eastern = pytz.timezone('US/Eastern')
    date = datetime.now(tz=pytz.utc)
    date = date.astimezone(pytz.timezone('US/Pacific')).strftime("%m-%d-%Y")
    s3_string = "s3://eugene.experiments/trb_bottleneck_paper/" \
                + date + '/' + args.exp_title
    config['env'] = env_name
    exp_dict = {
            'name': args.exp_title,
            'run_or_experiment': CCTrainer,
            'checkpoint_freq': args.checkpoint_freq,
            'stop': {
                'training_iteration': args.num_iters
            },
            'config': config,
            'num_samples': args.num_samples,
        }
    if args.use_s3:
        exp_dict['upload_dir'] = s3_string

    tune.run(**exp_dict, queue_trials=False)