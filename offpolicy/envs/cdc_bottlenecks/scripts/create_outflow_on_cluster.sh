#!/usr/bin/env bash

# This runs on a policy on a cluster and sync the output graphs to s3

startdate=$1

ray up ray_autoscale.yaml --cluster-name br_plot -y

# arguments are result directory, checkpoint number, file name
python flow/flow/visualize/bottleneck_results.py $2 $3 $4 --num_trials 20 --outflow_min 400 --outflow_max 3000  --cluster_mode

# move this into the script
if [ ! -d ~/flow/flow/visualize/trb_data/av_results/$startdate ]; then
  mkdir -p ~/flow/flow/visualize/trb_data/av_results/$startdate
fi

mv ~/flow/flow/visualize/trb_data/av_results/tmp ~/flow/flow/visualize/trb_data/av_results/$startdate/$startdate

aws s3 sync ~/flow/flow/visualize/trb_data/av_results/$startdate/ s3://eugene.experiments/trb_bottleneck_paper/policy_graph
