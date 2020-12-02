#!/usr/bin/env bash

# 9/05/19 experiments
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py MA_NLC_NCM_NLSTM_NAG_NCN --num_iters 350 --checkpoint_freq 50 \
##    --num_samples 2 --grid_search --n_cpus 30 --use_s3 --rollout_scale_factor 0.5 --horizon 2000" \
##    --start --stop --cluster-name exp1 --tmux
##
##ray exec ray_autoscale.yaml \
##"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py MA_NLC_NCM_LSTM_NAG_NCN --num_iters 350 --checkpoint_freq 50 \
##    --num_samples 2 --grid_search --n_cpus 30 --use_lstm --use_s3 --rollout_scale_factor 0.5 --horizon 2000" \
##    --start --stop --cluster-name exp2 --tmux
##
##ray exec ray_autoscale.yaml \
##"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py MA_NLC_NCM_NLSTM_AG_NCN --num_iters 350 --checkpoint_freq 50 \
##    --num_samples 2 --grid_search --n_cpus 30 --use_s3 --aggregate_info --rollout_scale_factor 0.5 --horizon 2000" \
#    --start --stop --cluster-name exp3 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py MA_NLC_NCM_LSTM_AG_NCN --num_iters 350 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 30 --use_lstm --use_s3 --aggregate_info --rollout_scale_factor 0.5 --horizon 2000" \
#    --start --stop --cluster-name exp4 --tmux
#
#----------------------------------- Add communication ------------------------------------------------------------------------------------------
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py MA_NLC_CM_NLSTM_NAG_NCN --num_iters 350 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 30 --use_s3 --communicate --rollout_scale_factor 0.5 --horizon 2000" \
#    --start --stop --cluster-name exp5 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py MA_NLC_CM_LSTM_NAG_NCN --num_iters 350 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 30 --use_lstm --use_s3 --communicate --rollout_scale_factor 0.5 --horizon 2000" \
#    --start --stop --cluster-name exp6 --tmux

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py MA_NLC_CM_NLSTM_AG_NCN --num_iters 350 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 30 --use_s3 --aggregate_info --communicate --rollout_scale_factor 0.5 --horizon 2000" \
#    --start --stop --cluster-name exp7 --tmux

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py MA_NLC_CM_LSTM_AG_NCN --num_iters 350 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 30 --use_lstm --use_s3 --aggregate_info --communicate --rollout_scale_factor 0.5 --horizon 2000" \
#    --start --stop --cluster-name exp8 --tmux
####################################################################################################################################################
####################################################################################################################################################

# 9/10/19 experiments
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py MA_NLC_NCM_NLSTM_NAG_CN_PEN --num_iters 350 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 8 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --congest_penalty" \
#    --start --stop --cluster-name exp1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py MA_NLC_NCM_NLSTM_AG_NCN_PEN --num_iters 350 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 8 --use_s3 --aggregate_info --rollout_scale_factor 1.0 --horizon 2000 --congest_penalty" \
#    --start --stop --cluster-name exp3 --tmux

####################################################################################################################################################
####################################################################################################################################################
# 9/15/19 experiments with centralized vf
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multi_bottleneck_centralized.py MA_NLC_NCM_NLSTM_NAG_CN_CVF --num_iters 350 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 8 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --central_vf_size 64" \
#    --start --stop --cluster-name exp1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multi_bottleneck_centralized.py MA_NLC_NCM_NLSTM_AG_NCN_CVF --num_iters 350 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 8 --use_s3 --aggregate_info --rollout_scale_factor 1.0 --horizon 2000 --central_vf_size 64" \
#    --start --stop --cluster-name exp3 --tmux

####################################################################################################################################################
####################################################################################################################################################
# 9/19/19 experiments with centralized observations, 1 experiment with a longer timestep, and 1 experiment with a higher range of inflows
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py MA_NLC_NCM_NLSTM_NAG_CN --num_iters 350 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 8 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --central_obs --high_inflow 2000" \
#    --start --stop --cluster-name exp1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py large_sim_step_NCN --num_iters 350 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 8 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --sim_step 1.0 --high_inflow 2000" \
#    --start --stop --cluster-name exp2 --tmux

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py large_inflows_NCN --num_iters 350 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 8 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 1400 --high_inflow 2200" \
#    --start --stop --cluster-name exp3 --tmux

####################################################################################################################################################
####################################################################################################################################################
# 9/20/19 experiments with a state space that contains edge position
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py MA_NLC_NCM_NLSTM_NAG_CN --num_iters 350 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 8 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --central_obs --high_inflow 2000" \
#    --start --stop --cluster-name exp1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py MA_NLC_NCM_NLSTM_NAG_NCN --num_iters 350 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 8 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --high_inflow 2000" \
#    --start --stop --cluster-name exp2 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multi_bottleneck_centralized.py MA_NLC_NCM_NLSTM_NAG_NCN_CVF --num_iters 350 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 8 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --central_vf_size 64 --high_inflow 2000" \
#    --start --stop --cluster-name exp3 --tmux

####################################################################################################################################################
####################################################################################################################################################
# 9/23/19 experiments with high inflows
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py MA_NLC_NCM_NLSTM_NAG_NCN_high_in --num_iters 350 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 8 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 1600 --high_inflow 2400" \
#    --start --stop --cluster-name exp2 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py MA_NLC_NCM_NLSTM_NAG_NCN_high_in_PEN --num_iters 350 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 8 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 1600 --high_inflow 2400 --congest_penalty" \
#    --start --stop --cluster-name exp3 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py fixed_inflow --num_iters 350 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 8 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --congest_penalty" \
#    --start --stop --cluster-name exp4 --tmux

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py fixed_inflow_lstm --num_iters 350 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 8 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --congest_penalty --use_lstm" \
#    --start --stop --cluster-name exp5 --tmux

####################################################################################################################################################
####################################################################################################################################################
# 9/25/19 experiments with fixed inflows and CVF and centralized obs and aggregate info

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py high_in_CN_NAGG_NLSTM --num_iters 200 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 8 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 \
#    --central_obs" \
#    --start --stop --cluster-name exp1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py high_in_CN_NAGG_LSTM --num_iters 200 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 4 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --use_lstm \
#    --central_obs" \
#    --start --stop --cluster-name exp2 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multi_bottleneck_centralized.py high_in_CN_NAGG_NLSTM_CVF --num_iters 200 --checkpoint_freq 50 --central_vf_size 64 \
#--num_samples 2 --grid_search --n_cpus 1 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --central_obs" --start --stop --cluster-name exp3 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multi_bottleneck_centralized.py high_in_CN_NAGG_LSTM_CVF --num_iters 200 --checkpoint_freq 50 --central_vf_size 64 \
#--num_samples 2 --grid_search --n_cpus 4 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --use_lstm --central_obs" --start --stop --cluster-name exp4 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py high_in_CN_AGG_NLSTM --num_iters 200 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 8 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --aggregate_info \
#    --central_obs" \
#    --start --stop --cluster-name exp5 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py high_in_CN_AGG_LSTM --num_iters 200 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 4 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --use_lstm --aggregate_info \
#    --central_obs" \
#    --start --stop --cluster-name exp6 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multi_bottleneck_centralized.py high_in_CN_AGG_NLSTM_CVF --num_iters 200 --checkpoint_freq 50 --central_vf_size 64 \
#--num_samples 2 --grid_search --n_cpus 1 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --central_obs" --start --stop --cluster-name exp7 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multi_bottleneck_centralized.py high_in_CN_AGG_LSTM_CVF --num_iters 200 --checkpoint_freq 50 --central_vf_size 64 \
#--num_samples 2 --grid_search --n_cpus 4 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --use_lstm --aggregate_info --central_obs" --start --stop --cluster-name exp8 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py high_in_NCN_AGG_NLSTM --num_iters 200 --checkpoint_freq 50 \
#--num_samples 2 --grid_search --n_cpus 8 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --aggregate_info" \
#--start --stop --cluster-name exp1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multi_bottleneck_centralized.py high_in_NCN_AGG_NLSTM_CVF --num_iters 200 --checkpoint_freq 50 \
#--num_samples 2 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --central_vf_size 64" \
#--start --stop --cluster-name exp9 --tmux

####################################################################################################################################################
####################################################################################################################################################
# 9/26/19 experiments with an action history and varied amounts of staying in the bottleneck penalty

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py high_in_NCN_AGG_NLSTM_0LPen --num_iters 200 --checkpoint_freq 50 \
#--num_samples 2 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --life_penalty 0.0" \
#--start --stop --cluster-name exp1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py high_in_NCN_AGG_NLSTM_Past --num_iters 200 --checkpoint_freq 50 \
#--num_samples 2 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --keep_past_actions" \
#--start --stop --cluster-name exp2 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multi_bottleneck_centralized.py high_in_NCN_AGG_NLSTM_0LPen_CVF --num_iters 200 --checkpoint_freq 50 \
#--num_samples 2 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --life_penalty 0.0 --central_vf_size 64" \
#--start --stop --cluster-name exp3 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multi_bottleneck_centralized.py high_in_NCN_AGG_NLSTM_Past_CVF --num_iters 200 --checkpoint_freq 50 \
#--num_samples 2 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --keep_past_actions --central_vf_size 64" \
#--start --stop --cluster-name exp4 --tmux

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py high_in_NCN_AGG_NLSTM_Past_0pen --num_iters 200 --checkpoint_freq 50 \
#--num_samples 2 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --keep_past_actions --life_penalty 0.0" \
#--start --stop --cluster-name exp5 --tmux

####################################################################################################################################################
####################################################################################################################################################
# 9/28/19 experiments with an action history and varied amounts of staying in the bottleneck penalty. Increased NN size.

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multi_bottleneck_centralized.py high_in_NCN_AGG_NLSTM_Past_1pen_CVF --num_iters 1000 --checkpoint_freq 50 \
#--num_samples 2 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --keep_past_actions --life_penalty 1.0 --central_vf_size 64" \
#--start --stop --cluster-name exp1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multi_bottleneck_centralized.py high_in_NCN_AGG_NLSTM_Past_2pen_CVF --num_iters 1000 --checkpoint_freq 50 \
#--num_samples 2 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --keep_past_actions --life_penalty 2.0 --central_vf_size 64" \
#--start --stop --cluster-name exp2 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multi_bottleneck_centralized.py high_in_NCN_AGG_NLSTM_Past_3pen_CVF --num_iters 1000 --checkpoint_freq 50 \
#--num_samples 2 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --keep_past_actions --life_penalty 3.0 --central_vf_size 64" \
#--start --stop --cluster-name exp3 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multi_bottleneck_centralized.py high_in_NCN_AGG_NLSTM_1pen_CVF --num_iters 1000 --checkpoint_freq 50 \
#--num_samples 2 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --life_penalty 1.0 --central_vf_size 64" \
#--start --stop --cluster-name exp4 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multi_bottleneck_centralized.py high_in_NCN_AGG_NLSTM_2pen_CVF --num_iters 1000 --checkpoint_freq 50 \
#--num_samples 2 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --life_penalty 2.0 --central_vf_size 64" \
#--start --stop --cluster-name exp5 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multi_bottleneck_centralized.py high_in_NCN_AGG_NLSTM_3pen_CVF --num_iters 1000 --checkpoint_freq 50 \
#--num_samples 2 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --life_penalty 3.0 --central_vf_size 64" \
#--start --stop --cluster-name exp6 --tmux

####################################################################################################################################################
####################################################################################################################################################
# 9/29/19 experiments with 40% penetration rate

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py high_in_NCN_AGG_NLSTM_Past_1pen_04frac --num_iters 1000 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 2 --grid_search --n_cpus 1 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --keep_past_actions --life_penalty 1.0" \
#--start --stop --cluster-name exp7 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py high_in_NCN_AGG_NLSTM_Past_2pen_04frac --num_iters 1000 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 2 --grid_search --n_cpus 1 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --keep_past_actions --life_penalty 2.0" \
#--start --stop --cluster-name exp8 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py high_in_NCN_AGG_NLSTM_Past_3pen_04frac --num_iters 1000 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 2 --grid_search --n_cpus 1 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --keep_past_actions --life_penalty 3.0" \
#--start --stop --cluster-name exp9 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py high_in_NCN_AGG_NLSTM_Past_1pen_99frac --num_iters 1000 --checkpoint_freq 50 --av_frac 0.99 \
#--num_samples 2 --grid_search --n_cpus 1 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --keep_past_actions --life_penalty 1.0" \
#--start --stop --cluster-name exp10 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py high_in_NCN_AGG_NLSTM_Past_2pen_99frac --num_iters 1000 --checkpoint_freq 50 --av_frac 0.99 \
#--num_samples 2 --grid_search --n_cpus 1 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --keep_past_actions --life_penalty 2.0" \
#--start --stop --cluster-name exp11 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py high_in_NCN_AGG_NLSTM_Past_3pen_9frac --num_iters 1000 --checkpoint_freq 50 --av_frac 0.99 \
#--num_samples 2 --grid_search --n_cpus 1 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --keep_past_actions --life_penalty 3.0" \
#--start --stop --cluster-name exp12 --tmux

####################################################################################################################################################
####################################################################################################################################################
# 9/30/19 experiments with centralized controller

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_LSTM --num_iters 400 --av_frac 0.1 \
#--num_samples 2 --n_cpus 36 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2300 --high_inflow 2301 --life_penalty 0.0 --use_lstm" \
#--start --stop --cluster-name exp1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_LSTM_PEN --num_iters 400 --av_frac 0.1 --congest_penalty \
#--num_samples 2 --n_cpus 36 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2300 --high_inflow 2301 --life_penalty 0.0 --use_lstm" \
#--start --stop --cluster-name exp2 --tmux

####################################################################################################################################################
####################################################################################################################################################
# 1/01/19 experiments with centralized controller. Added a grid search.

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_LSTM --num_iters 400 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 100 \
#--num_samples 1 --grid_search --n_cpus 36 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2300 --high_inflow 2301 --life_penalty 0.0 --use_lstm" \
#--start --stop --cluster-name exp1 --tmux

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_LSTM_A3C --num_iters 3000 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 100 \
#--num_samples 1 --grid_search --n_cpus 36 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2300 --high_inflow 2301 --life_penalty 0.0 --use_lstm --algorithm A3C" \
#--start --stop --cluster-name exp2 --tmux

####################################################################################################################################################
####################################################################################################################################################
# 10/12/19 experiments with centralized controller over PPO, SAC, A3C w/ a GRU.

## 0.1 PEN
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_A3C --num_iters 3000 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 100 \
#--num_samples 1 --grid_search --n_cpus 36 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 1200 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm A3C" \
#--start --stop --cluster-name exp1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_PPO --num_iters 300 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 100 \
#--num_samples 1 --grid_search --n_cpus 36 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 1200 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO" \
#--start --stop --cluster-name exp2 --tmux
#
##ray exec ray_autoscale.yaml \
##"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_SAC --num_iters 5000 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 100 \
##--num_samples 1 --grid_search --n_cpus 36 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 1200 --high_inflow 3000 --life_penalty 0.0 --algorithm SAC" \
##--start --stop --cluster-name exp3 --tmux
#
### 0.4 PEN
##
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p4_GRU_A3C --num_iters 3000 --av_frac 0.4 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 100 \
#--num_samples 1 --grid_search --n_cpus 36 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 1200 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm A3C" \
#--start --stop --cluster-name exp4 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p4_GRU_PPO --num_iters 300 --av_frac 0.4 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 100 \
#--num_samples 1 --grid_search --n_cpus 36 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 1200 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO" \
#--start --stop --cluster-name exp5 --tmux
##
##ray exec ray_autoscale.yaml \
##"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p4_SAC --num_iters 5000 --av_frac 0.4 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 100 \
##--num_samples 1 --grid_search --n_cpus 36 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 1200 --high_inflow 3000 --life_penalty 0.0 --algorithm SAC" \
##--start --stop --cluster-name exp6 --tmux
##
### 1.0 PEN
##
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_1p0_GRU_A3C --num_iters 3000 --av_frac 1.0 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 100 \
#--num_samples 1 --grid_search --n_cpus 36 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 1200 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm A3C" \
#--start --stop --cluster-name exp7 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_1p0_GRU_PPO --num_iters 300 --av_frac 1.0 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 100 \
#--num_samples 1 --grid_search --n_cpus 36 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 1200 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO" \
#--start --stop --cluster-name exp8 --tmux

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_1p0_SAC --num_iters 5000 --av_frac 1.0 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 100 \
#--num_samples 1 --grid_search --n_cpus 36 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 1200 --high_inflow 3000 --life_penalty 0.0 --algorithm SAC" \
#--start --stop --cluster-name exp9 --tmux

####################################################################################################################################################
####################################################################################################################################################
# 10/16/19 experiments with centralized controller over PPO, SAC, A3C w/ a GRU.

## 0.1 PEN
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_A3C --num_iters 3000 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 100 \
#--num_samples 1 --grid_search --n_cpus 36 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 1200 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm A3C --checkpoint_freq 500" \
#--start --stop --cluster-name ev_exp1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_PPO --num_iters 300 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 100 \
#--num_samples 1 --grid_search --n_cpus 36 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 1200 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO" \
#--start --stop --cluster-name ev_exp2 --tmux
#
### 0.4 PEN
##
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p4_GRU_A3C --num_iters 3000 --av_frac 0.4 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 100 \
#--num_samples 1 --grid_search --n_cpus 36 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 1200 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm A3C --checkpoint_freq 500" \
#--start --stop --cluster-name ev_exp4 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p4_GRU_PPO --num_iters 300 --av_frac 0.4 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 100 \
#--num_samples 1 --grid_search --n_cpus 36 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 1200 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO" \
#--start --stop --cluster-name ev_exp5 --tmux
#
### 1.0 PEN
##
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_1p0_GRU_A3C --num_iters 3000 --av_frac 1.0 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 100 \
#--num_samples 1 --grid_search --n_cpus 36 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 1200 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm A3C --checkpoint_freq 500" \
#--start --stop --cluster-name ev_exp7 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_1p0_GRU_PPO --num_iters 300 --av_frac 1.0 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 100 \
#--num_samples 1 --grid_search --n_cpus 36 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 1200 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO" \
#--start --stop --cluster-name ev_exp8 --tmux

## 0.1 PEN
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_A3C_ns20 --num_iters 3000 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 20 \
#--num_samples 1 --grid_search --n_cpus 15 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 1200 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm A3C --checkpoint_freq 500 \
#--vf_loss_coeff .0001" \
#--start --stop --cluster-name ev_exp1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_PPO_ns20 --num_iters 300 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 20 \
#--num_samples 1 --grid_search --n_cpus 15 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 1200 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001" \
#--start --stop --cluster-name ev_exp2 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_A3C_ns0p5 --num_iters 3000 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 15 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 1200 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm A3C --checkpoint_freq 500 \
#--vf_loss_coeff .0001" \
#--start --stop --cluster-name ev_exp3 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_PPO_ns0p5 --num_iters 300 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 15 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 1200 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001" \
#--start --stop --cluster-name ev_exp4 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_A3C_sr --num_iters 3000 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --speed_reward \
#--num_samples 1 --grid_search --n_cpus 15 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 1200 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm A3C --checkpoint_freq 500 \
#--vf_loss_coeff .0001" \
#--start --stop --cluster-name ev_exp5 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_PPO_sr --num_iters 300 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --speed_reward \
#--num_samples 1 --grid_search --n_cpus 15 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 1200 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001" \
#--start --stop --cluster-name ev_exp6 --tmux

####################################################################################################################################################
####################################################################################################################################################
# 10/20/19 experiments with centralized controller over PPO w/ a GRU and a congestion penalty
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_PPO_ns0p5_PEN --num_iters 300 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--congest_penalty_start 30 --state_space_scaling 1 --congest_penalty" \
#--start --stop --cluster-name ev_exp1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_PPO_ns0p5_PEN --num_iters 300 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--congest_penalty_start 20 --state_space_scaling 1 --congest_penalty" \
#--start --stop --cluster-name ev_exp2 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p4_GRU_PPO_ns0p5_PEN --num_iters 300 --av_frac 0.4 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--congest_penalty_start 30 --state_space_scaling 2 --congest_penalty" \
#--start --stop --cluster-name ev_exp3 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p4_GRU_PPO_ns0p5_PEN --num_iters 300 --av_frac 0.4 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--congest_penalty_start 20 --state_space_scaling 2 --congest_penalty" \
#--start --stop --cluster-name ev_exp4 --tmux

####################################################################################################################################################
####################################################################################################################################################
# 10/22/19 experiments with the fair reward

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_PPO_ns0p5_FAIR --num_iters 300 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 1 --fair_reward --exit_history_seconds 60" \
#--start --stop --cluster-name ev_exp1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p4_GRU_PPO_ns0p5_FAIR --num_iters 400 --av_frac 0.4 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 2 --fair_reward --exit_history_seconds 60" \
#--start --stop --cluster-name ev_exp2 --tmux

####################################################################################################################################################
####################################################################################################################################################
# 10/27/19 experiments with the fair reward

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_PPO_ns0p5_FAIR_bp25 --num_iters 300 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 1 --fair_reward --exit_history_seconds 60 --base_fair_reward 0.25" \
#--start --stop --cluster-name evexp1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p4_GRU_PPO_ns0p5_FAIR_bp25 --num_iters 400 --av_frac 0.4 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 2 --fair_reward --exit_history_seconds 60 --base_fair_reward 0.25" \
#--start --stop --cluster-name evexp2 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_PPO_ns0p5_FAIR_bp1 --num_iters 300 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 1 --fair_reward --exit_history_seconds 60 --base_fair_reward 0.1" \
#--start --stop --cluster-name evexp3 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p4_GRU_PPO_ns0p5_FAIR_bp1 --num_iters 400 --av_frac 0.4 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 2 --fair_reward --exit_history_seconds 60 --base_fair_reward 0.1" \
#--start --stop --cluster-name evexp4 --tmux

####################################################################################################################################################
####################################################################################################################################################
# 10/29/19 experiments with the fair reward

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_PPO_ns0p5_FAIR_bp1_PEN20 --num_iters 300 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 1 --fair_reward --exit_history_seconds 60 --base_fair_reward 0.1 --congest_penalty --congest_penalty_start 20" \
#--start --stop --cluster-name evexp1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p4_GRU_PPO_ns0p5_FAIR_bp1_PEN30 --num_iters 400 --av_frac 0.4 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 2 --fair_reward --exit_history_seconds 60 --base_fair_reward 0.1 --congest_penalty --congest_penalty_start 30" \
#--start --stop --cluster-name evexp2 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_PPO_ns0p5_FAIR_bp1_PEN20 --num_iters 300 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 1 --fair_reward --exit_history_seconds 60 --base_fair_reward 0.1 --congest_penalty --congest_penalty_start 20" \
#--start --stop --cluster-name evexp3 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p4_GRU_PPO_ns0p5_FAIR_bp1_PEN30 --num_iters 400 --av_frac 0.4 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 2 --fair_reward --exit_history_seconds 60 --base_fair_reward 0.1 --congest_penalty  --congest_penalty_start 20" \
#--start --stop --cluster-name evexp4 --tmux

####################################################################################################################################################
####################################################################################################################################################
# 10/30/19 experiments with the fair reward

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_PPO_ns0p5_FAIR_PEN20 --num_iters 300 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 1 --fair_reward --exit_history_seconds 60 --congest_penalty --congest_penalty_start 20" \
#--start --stop --cluster-name evexp1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p4_GRU_PPO_ns0p5_FAIR_PEN20 --num_iters 400 --av_frac 0.4 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 2 --fair_reward --exit_history_seconds 60 --congest_penalty --congest_penalty_start 20" \
#--start --stop --cluster-name evexp2 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_PPO_ns0p5_FAIR_PEN30 --num_iters 300 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 1 --fair_reward --exit_history_seconds 60 --congest_penalty --congest_penalty_start 30" \
#--start --stop --cluster-name evexp3 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p4_GRU_PPO_ns0p5_FAIR_PEN30 --num_iters 400 --av_frac 0.4 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 2 --fair_reward --exit_history_seconds 60 --congest_penalty  --congest_penalty_start 30" \
#--start --stop --cluster-name evexp4 --tmux

####################################################################################################################################################
####################################################################################################################################################
# 11/11/19 experiments with the fair reward. We have removed control outside of section 3 so the congest penalty is unnecessary.
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_PPO_ns0p5_FAIR --num_iters 300 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 1 --fair_reward --exit_history_seconds 60" \
#--start --stop --cluster-name evexp1 --tmux

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_SAC_ns0p5_FAIR --num_iters 300 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --algorithm SAC --vf_loss_coeff .0001 \
#--state_space_scaling 1 --fair_reward --exit_history_seconds 60" \
#--start --stop --cluster-name evexp2 --tmux

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p4_GRU_PPO_ns0p5_FAIR --num_iters 400 --av_frac 0.4 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 2 --fair_reward --exit_history_seconds 60" \
#--start --stop --cluster-name evexp3 --tmux

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p4_SAC_ns0p5_FAIR --num_iters 400 --av_frac 0.4 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --algorithm SAC --vf_loss_coeff .0001 \
#--state_space_scaling 2 --fair_reward --exit_history_seconds 60" \
#--start --stop --cluster-name evexp4 --tmux

####################################################################################################################################################
####################################################################################################################################################
# 11/12/19 experiments with the fair reward. We have removed control outside of section 3 so the congest penalty is unnecessary.
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_PPO_ns0p5_FAIR_PEN30 --num_iters 300 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 2 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 1 --fair_reward --exit_history_seconds 60 --congest_penalty --congest_penalty_start 30" \
#--start --stop --cluster-name evexp1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p4_GRU_PPO_ns0p5_FAIR_PEN30 --num_iters 400 --av_frac 0.4 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 2 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 2 --fair_reward --exit_history_seconds 60 --congest_penalty --congest_penalty_start 30" \
#--start --stop --cluster-name evexp2 --tmux

####################################################################################################################################################
####################################################################################################################################################
# 11/13/19 experiments with the paired fair reward. We have removed control outside of section 3 so the congest penalty is unnecessary.
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_PPO_ns0p5_FAIR_PEN30 --num_iters 300 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 2 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 1 --fair_reward --exit_history_seconds 60 --congest_penalty --congest_penalty_start 30" \
#--start --stop --cluster-name evexp1 --tmux

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p4_GRU_PPO_ns0p5_FAIR_PEN30 --num_iters 400 --av_frac 0.4 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 2 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 2 --fair_reward --exit_history_seconds 60 --congest_penalty --congest_penalty_start 30" \
#--start --stop --cluster-name evexp2 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p6_GRU_PPO_ns0p5_FAIR_PEN30 --num_iters 400 --av_frac 0.6 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 2 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 2 --fair_reward --exit_history_seconds 60 --congest_penalty --congest_penalty_start 30" \
#--start --stop --cluster-name evexp3 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_PPO_ns0p5_FAIR_PEN20 --num_iters 300 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 2 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 1 --fair_reward --exit_history_seconds 60 --congest_penalty --congest_penalty_start 20" \
#--start --stop --cluster-name evexp4 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p4_GRU_PPO_ns0p5_FAIR_PEN20 --num_iters 400 --av_frac 0.4 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 2 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 2 --fair_reward --exit_history_seconds 60 --congest_penalty --congest_penalty_start 20" \
#--start --stop --cluster-name evexp5 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p6_GRU_PPO_ns0p5_FAIR_PEN20 --num_iters 400 --av_frac 0.6 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 2 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 2 --fair_reward --exit_history_seconds 60 --congest_penalty --congest_penalty_start 20" \
#--start --stop --cluster-name evexp6 --tmux

####################################################################################################################################################
####################################################################################################################################################
# 11/14/19 experiments with the paired fair reward. We have removed control outside of section 3 so the congest penalty is unnecessary.
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_PPO_ns0p5_FAIR_PEN30_es30 --num_iters 300 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 1 --fair_reward --exit_history_seconds 30 --congest_penalty --congest_penalty_start 30" \
#--start --stop --cluster-name evexp1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p4_GRU_PPO_ns0p5_FAIR_PEN30_es30 --num_iters 400 --av_frac 0.4 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 2 --fair_reward --exit_history_seconds 30 --congest_penalty --congest_penalty_start 30" \
#--start --stop --cluster-name evexp2 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_PPO_ns0p5_FAIR_PEN20_es30 --num_iters 300 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 1 --fair_reward --exit_history_seconds 30 --congest_penalty --congest_penalty_start 20" \
#--start --stop --cluster-name evexp3 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p4_GRU_PPO_ns0p5_FAIR_PEN20_es30 --num_iters 400 --av_frac 0.4 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 2 --fair_reward --exit_history_seconds 30 --congest_penalty --congest_penalty_start 20" \
#--start --stop --cluster-name evexp4 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_PPO_ns0p5_FAIR_PEN30_es90 --num_iters 300 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 1 --fair_reward --exit_history_seconds 90 --congest_penalty --congest_penalty_start 30" \
#--start --stop --cluster-name evexp5 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p4_GRU_PPO_ns0p5_FAIR_PEN30_es90 --num_iters 400 --av_frac 0.4 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 2 --fair_reward --exit_history_seconds 90 --congest_penalty --congest_penalty_start 30" \
#--start --stop --cluster-name evexp6 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_PPO_ns0p5_FAIR_PEN20_es90 --num_iters 300 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 1 --fair_reward --exit_history_seconds 90 --congest_penalty --congest_penalty_start 20" \
#--start --stop --cluster-name evexp7 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p4_GRU_PPO_ns0p5_FAIR_PEN20_es90 --num_iters 400 --av_frac 0.4 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 2 --fair_reward --exit_history_seconds 90 --congest_penalty --congest_penalty_start 20" \
#--start --stop --cluster-name evexp8 --tmux

####################################################################################################################################################
####################################################################################################################################################
# 11/26/19 we run both the imitation and the basic decentralized exps

# ray exec ray_autoscale.yaml \
# "python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py high_in_NCN_AGG_NLSTM_Past_0pen_04frac --num_iters 500 --checkpoint_freq 50 --av_frac 0.4 \
# --num_samples 2 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 1000 --low_inflow 2300 --high_inflow 2301 --aggregate_info \
# --keep_past_actions --multi_node --vf_loss_coeff .0001 --life_penalty 0.0 --sim_step 0.5" \
# --start --stop --cluster-name kp_high_in_NCN_AGG_NLSTM_Past_0pen_04frac --tmux

# ray exec ray_autoscale.yaml \
# "python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py high_in_NCN_AGG_NLSTM_Past_0pen_04frac_im --num_iters 500 --checkpoint_freq 50 --av_frac 0.4 \
# --num_samples 2 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 1000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --keep_past_actions \
# --imitate --multi_node --vf_loss_coeff .0001 --life_penalty 0.0 --sim_step 0.5" \
# --start --stop --cluster-name kp_high_in_NCN_AGG_NLSTM_Past_0pen_04frac_im --tmux

# ray exec ray_autoscale.yaml \
# "python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py high_in_NCN_AGG_NLSTM_Past_2pen_04frac --num_iters 500 --checkpoint_freq 50 --av_frac 0.4 \
# --num_samples 2 --grid_search --n_cpus 3 --use_s3 --rollout_scale_factor 1.0 --horizon 1000 --low_inflow 2300 --high_inflow 2301 --aggregate_info \
# --keep_past_actions --multi_node --vf_loss_coeff .0001 --life_penalty 3.0 --sim_step 0.5" \
# --start --stop --cluster-name kp_high_in_NCN_AGG_NLSTM_Past_2pen_04frac --tmux

# ray exec ray_autoscale.yaml \
# "python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py high_in_NCN_AGG_NLSTM_Past_2pen_04frac_im --num_iters 500 --checkpoint_freq 50 --av_frac 0.4 \
# --num_samples 2 --grid_search --n_cpus 3 --use_s3 --rollout_scale_factor 1.0 --horizon 1000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --keep_past_actions \
# --imitate --multi_node --vf_loss_coeff .0001 --life_penalty 3.0 --sim_step 0.5" \
# --start --stop --cluster-name kp_high_in_NCN_AGG_NLSTM_Past_2pen_04frac_im --tmux

####################################################################################################################################################
####################################################################################################################################################

# python examples/rllib/multiagent_exps/multiagent_bottleneck.py high_in_NCN_AGG_NLSTM_Past_0pen_04frac --num_iters 1 --checkpoint_freq 50 --av_frac 0.4 \
# --num_samples 2 --grid_search --n_cpus 2 --rollout_scale_factor 1.0 --horizon 1000 --low_inflow 2300 --high_inflow 2301 --aggregate_info \
# --keep_past_actions --vf_loss_coeff .0001 --life_penalty 0.0 --sim_step 0.5

# ray exec scripts/ray_autoscale.yaml \
# "python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py high_in_NCN_AGG_NLSTM_Past_0pen_04frac --num_iters 500 --checkpoint_freq 50 --av_frac 0.4 \
# --num_samples 1 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 1000 --low_inflow 2300 --high_inflow 2301 --aggregate_info \
# --keep_past_actions --multi_node --vf_loss_coeff .0001 --life_penalty 0.0 --sim_step 0.5" \
# --start --stop --cluster-name kp_high_in_NCN_AGG_NLSTM_Past_0pen_04frac --tmux

# ray exec scripts/ray_autoscale.yaml \
# "python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py high_in_NCN_AGG_NLSTM_Past_0pen_04frac_im --num_iters 500 --checkpoint_freq 50 --av_frac 0.4 \
# --num_samples 1 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 1000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --keep_past_actions \
# --imitate --multi_node --vf_loss_coeff .0001 --life_penalty 0.0 --sim_step 0.5" \
# --start --stop --cluster-name kp_high_in_NCN_AGG_NLSTM_Past_0pen_04frac_im --tmux

# ray exec scripts/ray_autoscale.yaml \
# "python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py high_in_NCN_AGG_NLSTM_Past_2pen_04frac --num_iters 500 --checkpoint_freq 50 --av_frac 0.4 \
# --num_samples 1 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 1000 --low_inflow 2300 --high_inflow 2301 --aggregate_info \
# --keep_past_actions --multi_node --vf_loss_coeff .0001 --life_penalty 3.0 --sim_step 0.5" \
# --start --stop --cluster-name kp_high_in_NCN_AGG_NLSTM_Past_3pen_04frac --tmux

# ray exec scripts/ray_autoscale.yaml \
# "python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py high_in_NCN_AGG_NLSTM_Past_2pen_04frac_im --num_iters 500 --checkpoint_freq 50 --av_frac 0.4 \
# --num_samples 1 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 1000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --keep_past_actions \
# --imitate --multi_node --vf_loss_coeff .0001 --life_penalty 3.0 --sim_step 0.5" \
# --start --stop --cluster-name kp_high_in_NCN_AGG_NLSTM_Past_3pen_04frac_im --tmux

# ray exec scripts/ray_autoscale.yaml \
# "python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py high_in_NCN_AGG_NLSTM_2pen_04frac --num_iters 500 --checkpoint_freq 50 --av_frac 0.4 \
# --num_samples 1 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 1000 --low_inflow 2300 --high_inflow 2301 --aggregate_info \
# --multi_node --vf_loss_coeff .0001 --life_penalty 3.0 --sim_step 0.5" \
# --start --stop --cluster-name kp_high_in_NCN_AGG_NLSTM_2pen_04frac --tmux

# ray exec scripts/ray_autoscale.yaml \
# "python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py high_in_NCN_AGG_NLSTM_2pen_04frac_im --num_iters 500 --checkpoint_freq 50 --av_frac 0.4 \
# --num_samples 1 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 1000 --low_inflow 2300 --high_inflow 2301 --aggregate_info \
# --imitate --multi_node --vf_loss_coeff .0001 --life_penalty 3.0 --sim_step 0.5" \
# --start --stop --cluster-name kp_high_in_NCN_AGG_NLSTM_2pen_04frac_im --tmux

####################################################################################################################################################
####################################################################################################################################################

#ray exec scripts/ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py Past_0pen_im --num_iters 500 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 1000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --keep_past_actions \
#--imitate --multi_node --vf_loss_coeff .0001 --life_penalty 0.0 --sim_step 0.5" \
#--start --stop --cluster-name kp_Past_0pen_im --tmux
#
#ray exec scripts/ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py Past_2pen_im --num_iters 500 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 1000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --keep_past_actions \
#--imitate --multi_node --vf_loss_coeff .0001 --life_penalty 3.0 --sim_step 0.5" \
#--start --stop --cluster-name kp_Past_3pen_im --tmux
#
#ray exec scripts/ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 3pen_im --num_iters 500 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 1000 --low_inflow 2300 --high_inflow 2301 --aggregate_info \
#--imitate --multi_node --vf_loss_coeff .0001 --life_penalty 3.0 --sim_step 0.5" \
#--start --stop --cluster-name kp_3pen_im --tmux
#
#ray exec scripts/ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 3pen_im_200s --num_iters 500 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 1000 --low_inflow 2300 --high_inflow 2301 --aggregate_info \
#--imitate --multi_node --vf_loss_coeff .0001 --life_penalty 3.0 --sim_step 0.5 --num_imitation_iters 300" \
#--start --stop --cluster-name kp_3pen_im_200s --tmux
#
#ray exec scripts/ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py past_3pen_im200s --num_iters 500 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 1000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --keep_past_actions \
#--imitate --multi_node --vf_loss_coeff .0001 --life_penalty 3.0 --sim_step 0.5 --num_imitation_iters 300" \
#--start --stop --cluster-name kp_Past_3pen_im200s --tmux

#####################################################################################################################################################
#####################################################################################################################################################
## 12/14/19 exps
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py Past_0pen_im --num_iters 500 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 1000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --keep_past_actions \
#--imitate --multi_node --vf_loss_coeff .0001 --life_penalty 0.0 --sim_step 0.5" \
#--start --stop --cluster-name ev_Past_0pen_im --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py Past_2pen_im --num_iters 500 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 1000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --keep_past_actions \
#--imitate --multi_node --vf_loss_coeff .0001 --life_penalty 3.0 --sim_step 0.5" \
#--start --stop --cluster-name ev_Past_3pen_im --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 3pen_im --num_iters 500 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 1000 --low_inflow 2300 --high_inflow 2301 --aggregate_info \
#--imitate --multi_node --vf_loss_coeff .0001 --life_penalty 3.0 --sim_step 0.5" \
#--start --stop --cluster-name ev_3pen_im --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 3pen_im_200s --num_iters 500 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 1000 --low_inflow 2300 --high_inflow 2301 --aggregate_info \
#--imitate --multi_node --vf_loss_coeff .0001 --life_penalty 3.0 --sim_step 0.5 --num_imitation_iters 300" \
#--start --stop --cluster-name ev_3pen_im_200s --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py past_3pen_im200s --num_iters 500 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 1000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --keep_past_actions \
#--imitate --multi_node --vf_loss_coeff .0001 --life_penalty 3.0 --sim_step 0.5 --num_imitation_iters 300" \
#--start --stop --cluster-name ev_Past_3pen_im200s --tmux

#####################################################################################################################################################
#####################################################################################################################################################
## 12/15/19 exps
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck_dqfd.py dqfd_Past_0pen_im_2300i --num_iters 500 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 1000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --keep_past_actions \
#--life_penalty 0.0 --sim_step 0.5 --fingerprinting" \
#--start --stop --cluster-name ev_exp1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck_dqfd.py dqfd_0pen_im_2300i --num_iters 500 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 1000 --low_inflow 1900 --high_inflow 1900 --aggregate_info \
#--life_penalty 0.0 --sim_step 0.5 --fingerprinting" \
#--start --stop --cluster-name ev_exp2 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck_dqfd.py dqfd_2pen_im_kp_1900i --num_iters 500 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 1000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --keep_past_actions \
#--life_penalty 2.0 --sim_step 0.5 --fingerprinting" \
#--start --stop --cluster-name ev_exp3 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck_dqfd.py dqfd_2pen_im_1900i --num_iters 500 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 1000 --low_inflow 1900 --high_inflow 1900 --aggregate_info \
#--life_penalty 2.0 --sim_step 0.5 --fingerprinting" \
#--start --stop --cluster-name ev_exp4 --tmux


####################################################################################################################################################
####################################################################################################################################################
# 12/18/19 exps

#ray exec scripts/ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py Past_0pen_im --num_iters 500 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 1000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --keep_past_actions \
#--imitate --multi_node --vf_loss_coeff .0001 --life_penalty 0.0 --sim_step 0.5" \
#--start --stop --cluster-name kp_Past_0pen_im --tmux
#
#ray exec scripts/ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py Past_2pen_im --num_iters 500 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 1000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --keep_past_actions \
#--imitate --multi_node --vf_loss_coeff .0001 --life_penalty 3.0 --sim_step 0.5" \
#--start --stop --cluster-name kp_Past_3pen_im --tmux
#
#ray exec scripts/ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 3pen_im --num_iters 500 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 1000 --low_inflow 2300 --high_inflow 2301 --aggregate_info \
#--imitate --multi_node --vf_loss_coeff .0001 --life_penalty 3.0 --sim_step 0.5" \
#--start --stop --cluster-name kp_3pen_im --tmux
#
#ray exec scripts/ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 3pen_im_200s --num_iters 500 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 1000 --low_inflow 2300 --high_inflow 2301 --aggregate_info \
#--imitate --multi_node --vf_loss_coeff .0001 --life_penalty 3.0 --sim_step 0.5 --num_imitation_iters 300" \
#--start --stop --cluster-name kp_3pen_im_200s --tmux
#
#ray exec scripts/ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py past_3pen_im200s --num_iters 500 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 1000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --keep_past_actions \
#--imitate --multi_node --vf_loss_coeff .0001 --life_penalty 3.0 --sim_step 0.5 --num_imitation_iters 300" \
#--start --stop --cluster-name kp_Past_3pen_im200s --tmux

####################################################################################################################################################
####################################################################################################################################################
## 2/19/20 exps
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 3pen_im --num_iters 500 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 1000 --low_inflow 1900 --high_inflow 1900 --aggregate_info \
#--imitate --multi_node  --life_penalty 3.0 --sim_step 0.5" \
#--start --stop --cluster-name kp_3pen_im --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 3pen_im_cvf --num_iters 500 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 1000 --low_inflow 1900 --high_inflow 1900 --aggregate_info \
#--imitate --multi_node  --life_penalty 3.0 --sim_step 0.5 --centralized_vf" \
#--start --stop --cluster-name kp_3pen_im_cvf --tmux

####################################################################################################################################################
####################################################################################################################################################
### 2/20/20 exps
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 3pen_im_i2600 --num_iters 500 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2600 --high_inflow 2600 --aggregate_info \
#--imitate --multi_node  --life_penalty 3.0 --sim_step 0.5" \
#--start --stop --cluster-name kp_3pen_im1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 3pen_im_cvf_i2600 --num_iters 500 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2600 --high_inflow 2600 --aggregate_info \
#--imitate --multi_node  --life_penalty 3.0 --sim_step 0.5 --centralized_vf --max_num_agents 200" \
#--start --stop --cluster-name kp_3pen_im_cvf2 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 3pen_im_i2600_f0p1 --num_iters 500 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2600 --high_inflow 2600 --aggregate_info \
#--imitate --multi_node  --life_penalty 3.0 --sim_step 0.5 --final_imitation_weight 0.1" \
#--start --stop --cluster-name kp_3pen_im3 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 3pen_im_cvf_i2600_f0p1 --num_iters 500 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2600 --high_inflow 2600 --aggregate_info \
#--imitate --multi_node  --life_penalty 3.0 --sim_step 0.5 --centralized_vf --final_imitation_weight 0.1 --max_num_agents 200" \
#--start --stop --cluster-name kp_3pen_im_cvf4 --tmux

####################################################################################################################################################
#####################################################################################################################################################
### 2/26/20 exps
## Imitation
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 0pen_im_i1900 --num_iters 400 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 6 --use_s3 --rollout_scale_factor 1.0 --horizon 200 --low_inflow 1900 --high_inflow 1900 --aggregate_info \
#--imitate --multi_node --sim_step 0.5 --life_penalty 0.0" \
#--start --stop --cluster-name kp_0pen_im1 --tmux
#
## Imitation with CVF
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 0pen_im_cvf_i1900 --num_iters 400 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 6 --use_s3 --rollout_scale_factor 1.0 --horizon 200 --low_inflow 1900 --high_inflow 1900 --aggregate_info \
#--imitate --multi_node --sim_step 0.5 --centralized_vf --max_num_agents 200 --life_penalty 0.0" \
#--start --stop --cluster-name kp_0pen_im_cvf2 --tmux
#
## Imitation with prierarchy
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 0pen_im_i1900_f0p1 --num_iters 400 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 6 --use_s3 --rollout_scale_factor 1.0 --horizon 200 --low_inflow 1900 --high_inflow 1900 --aggregate_info \
#--imitate --multi_node --sim_step 0.5 --final_imitation_weight 0.1 --life_penalty 0.0" \
#--start --stop --cluster-name kp_0pen_im3 --tmux
#
## Imitation with prierarchy and CVF
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 0pen_im_cvf_i1900_f0p1 --num_iters 400 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 6 --use_s3 --rollout_scale_factor 1.0 --horizon 200 --low_inflow 1900 --high_inflow 1900 --aggregate_info \
#--imitate --multi_node --sim_step 0.5 --centralized_vf --final_imitation_weight 0.1 --max_num_agents 200 --life_penalty 0.0" \
#--start --stop --cluster-name kp_0pen_im_cvf4 --tmux
#
## Just CVF
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 0pen_cvf_i1900 --num_iters 400 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 6 --use_s3 --rollout_scale_factor 1.0 --horizon 200 --low_inflow 1900 --high_inflow 1900 --aggregate_info \
#--multi_node --sim_step 0.5 --centralized_vf --max_num_agents 200 --life_penalty 0.0" \
#--start --stop --cluster-name kp_0pen_cvf5 --tmux
#
## NO CVF
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 0pen_i1900 --num_iters 400 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 6 --use_s3 --rollout_scale_factor 1.0 --horizon 200 --low_inflow 1900 --high_inflow 1900 --aggregate_info \
#--multi_node --sim_step 0.5 --max_num_agents 200 --life_penalty 0.0" \
#--start --stop --cluster-name kp_0pen_6 --tmux

####################################################################################################################################################
## 3/02/20 exps
# Imitation
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 2pen_im_i1900 --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 19 --use_s3 --rollout_scale_factor 1.0 --horizon 200 --low_inflow 1900 --high_inflow 1900 --aggregate_info \
#--imitate --multi_node --sim_step 0.5 --life_penalty 2.0 --num_imitation_iters 150" \
#--start --stop --cluster-name kp_0pen_im1 --tmux

## Imitation with CVF
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 2pen_im_cvf_i1900 --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 19 --use_s3 --rollout_scale_factor 1.0 --horizon 200 --low_inflow 1900 --high_inflow 1900 --aggregate_info \
#--imitate --multi_node --sim_step 0.5 --centralized_vf --max_num_agents 200 --life_penalty 2.0 --num_imitation_iters 150" \
#--start --stop --cluster-name kp_0pen_im_cvf2 --tmux
#
## Imitation with prierarchy
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 2pen_im_i1900_f0p02 --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 19 --use_s3 --rollout_scale_factor 1.0 --horizon 200 --low_inflow 1900 --high_inflow 1900 --aggregate_info \
#--imitate --multi_node --sim_step 0.5 --final_imitation_weight 0.02 --life_penalty 2.0 --num_imitation_iters 300" \
#--start --stop --cluster-name kp_0pen_im3 --tmux
#
## Imitation with prierarchy and CVF
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 2pen_im_cvf_i1900_f0p02 --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 19 --use_s3 --rollout_scale_factor 1.0 --horizon 200 --low_inflow 1900 --high_inflow 1900 --aggregate_info \
#--imitate --multi_node --sim_step 0.5 --centralized_vf --final_imitation_weight 0.02 --max_num_agents 200 --life_penalty 2.0 --num_imitation_iters 150" \
#--start --stop --cluster-name kp_0pen_im_cvf4 --tmux
#
## Just CVF
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 2pen_cvf_i1900 --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 19 --use_s3 --rollout_scale_factor 1.0 --horizon 200 --low_inflow 1900 --high_inflow 1900 --aggregate_info \
#--multi_node --sim_step 0.5 --centralized_vf --max_num_agents 200 --life_penalty 2.0 --num_imitation_iters 150" \
#--start --stop --cluster-name kp_0pen_cvf5 --tmux
#
## NO CVF
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 2pen_i1900 --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 19 --use_s3 --rollout_scale_factor 1.0 --horizon 200 --low_inflow 1900 --high_inflow 1900 --aggregate_info \
#--multi_node --sim_step 0.5 --max_num_agents 200 --life_penalty 2.0 --num_imitation_iters 150" \
#--start --stop --cluster-name kp_0pen_6 --tmux

## Imitation with hard negative mining
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 2pen_im_i1900_neg --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 19 --use_s3 --rollout_scale_factor 1.0 --horizon 200 --low_inflow 1900 --high_inflow 1900 --aggregate_info \
#--imitate --multi_node --sim_step 0.5 --life_penalty 2.0 --num_imitation_iters 150 --hard_negative_mining" \
#--start --stop --cluster-name kp_0pen_im7 --tmux
#
#
## Imitation with prierarchy and hard negative mining
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 2pen_im_i1900_f0p02_neg --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 19 --use_s3 --rollout_scale_factor 1.0 --horizon 200 --low_inflow 1900 --high_inflow 1900 --aggregate_info \
#--imitate --multi_node --sim_step 0.5 --final_imitation_weight 0.02 --life_penalty 2.0 --num_imitation_iters 300 --hard_negative_mining" \
#--start --stop --cluster-name kp_0pen_im8 --tmux

#####################################################################################################################################################
### 3/03/20 exps
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 2pen_i2400 --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 2.0 --horizon 200 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 2.0 --create_inflow_graph" \
#--start --stop --cluster-name ev_2pen_1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 2pen_il400h3500 --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 2.0 --horizon 200 --low_inflow 400 --high_inflow 3500 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 2.0 --create_inflow_graph" \
#--start --stop --cluster-name ev_2pen_2 --tmux

####################################################################################################################################################
## 3/04/20 exps

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 2pen_i2400 --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 2.0 --horizon 200 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 2.0 --create_inflow_graph --sims_per_step 10" \
#--start --stop --cluster-name ev_2pen_3 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 2pen_il400h3500 --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 2.0 --horizon 200 --low_inflow 400 --high_inflow 3500 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 2.0 --create_inflow_graph --sims_per_step 10" \
#--start --stop --cluster-name ev_2pen_4 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 2pen_i2400_imitate --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 2.0 --horizon 200 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 2.0 --create_inflow_graph --sims_per_step 10 --imitate" \
#--start --stop --cluster-name ev_2pen_5 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 2pen_i2400_imitate_neg --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 2.0 --horizon 200 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 2.0 --create_inflow_graph --sims_per_step 10 --imitate --hard_negative_mining" \
#--start --stop --cluster-name ev_2pen_6 --tmux

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 1pen_i2400_imitate_neg --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 2.0 --horizon 200 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 1.0 --create_inflow_graph --sims_per_step 10 --imitate --hard_negative_mining" \
#--start --stop --cluster-name ev_1pen_7 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 0pen_i2400_imitate_neg --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 2.0 --horizon 200 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 10 --imitate --hard_negative_mining" \
#--start --stop --cluster-name ev_1pen_8 --tmux

####################################################################################################################################################
## 3/05/20 exps

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 0pen_i2400 --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 2.0 --horizon 200 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 10" \
#--start --stop --cluster-name ev_2pen_1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 0pen_i2400_cvf --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 2.0 --horizon 200 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 10 --centralized_vf" \
#--start --stop --cluster-name ev_2pen_2 --tmux

#ray exec ray_autoscale.yaml \
#" python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py dqfd_pen0_f0p4_nofinger_i2400 --num_iters 20 --av_frac 0.4 \
#--num_samples 1 --rollout_scale_factor 1.0 --grid_search --use_s3 --horizon 200 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--dqfd --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 10 --num_expert_steps 50000" \
#--start --stop --cluster-name ev_dqfd2 --tmux
#
#ray exec ray_autoscale.yaml \
#" python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py dqfd_pen0_f0p4_finger_i2400 --num_iters 20 --av_frac 0.4 \
#--num_samples 1 --rollout_scale_factor 1.0 --grid_search --use_s3 --horizon 200 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--dqfd --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 10 --num_expert_steps 50000 --fingerprinting" \
#--start --stop --cluster-name ev_dqfd3 --tmux

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 0pen_i2400_im --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 2.0 --horizon 200 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 10 --imitate" \
#--start --stop --cluster-name ev_2pen_im_1 --tmux

#####################################################################################################################################################
#### 3/07/20 exps
##ray exec ray_autoscale.yaml \
##"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 0pen_i2400 --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
##--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 2.0 --horizon 200 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
##--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 10" \
##--start --stop --cluster-name ev_2pen_1 --tmux
##
##ray exec ray_autoscale.yaml \
##"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 0pen_i2400_cvf --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
##--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 2.0 --horizon 200 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
##--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 10 --centralized_vf" \
##--start --stop --cluster-name ev_2pen_2 --tmux

####################################################################################################################################################
## 3/07/20 exps
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 0pen_i2400_trial2 --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 2.0 --horizon 200 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 10 --terminal_reward" \
#--start --stop --cluster-name ev_2pen_1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 0pen_i2400_cvf_trial2 --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 2.0 --horizon 200 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 10 --centralized_vf --max_num_agents 200 --terminal_reward" \
#--start --stop --cluster-name ev_2pen_2 --tmux

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 0pen_i2400_no_out_rew --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 2.0 --horizon 200 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 10 --terminal_reward --num_sample_seconds 0.0" \
#--start --stop --cluster-name ev_2pen_3 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 0pen_i2400_no_out_rew --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 2.0 --horizon 200 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 10 --centralized_vf --max_num_agents 200 --terminal_reward --num_sample_seconds 0.0" \
#--start --stop --cluster-name ev_2pen_4 --tmux

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 0pen_i2400_no_out_rew_true --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 2.0 --horizon 200 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 10 --terminal_reward --num_sample_seconds 0.0" \
#--start --stop --cluster-name ev_2pen_5 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 0pen_i2400_no_out_rew_truecv --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 2.0 --horizon 200 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 10 --centralized_vf --max_num_agents 200 --terminal_reward --num_sample_seconds 0.0" \
#--start --stop --cluster-name ev_2pen_6 --tmux

####################################################################################################################################################
## 3/08/20 exps
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 0pen_i2400_imitate_term_f0p01 --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 2.0 --horizon 200 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 10 --centralized_vf --max_num_agents 200 \
#--terminal_reward --num_sample_seconds 0.0 --imitate --num_imitation_iters 10 --final_imitation_weight 0.01" \
#--start --stop --cluster-name ev_2pen_7 --tmux

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 0pen_i2400_imitate_term_f0p1 --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 2.0 --horizon 200 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 10 --centralized_vf --max_num_agents 200 \
#--terminal_reward --num_sample_seconds 0.0 --imitate --num_imitation_iters 10 --final_imitation_weight 0.1" \
#--start --stop --cluster-name ev_2pen_8 --tmux

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 0pen_i2400_imitate_term_f1p0 --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 2.0 --horizon 200 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 10 --centralized_vf --max_num_agents 200 \
#--terminal_reward --num_sample_seconds 0.0 --imitate --num_imitation_iters 0 --final_imitation_weight 1.0" \
#--start --stop --cluster-name ev_2pen_8 --tmux

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 0pen_i2400_imitate_term_f1p0_hard --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 2.0 --horizon 200 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 10 --centralized_vf --max_num_agents 200 \
#--terminal_reward --num_sample_seconds 0.0 --imitate --num_imitation_iters 0 --final_imitation_weight 1.0 --hard_negative_mining --mining_frac 0.05" \
#--start --stop --cluster-name ev_2pen_9 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 0pen_i2400_imitate_term_f1p0_hard_senv --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 2.0 --horizon 200 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 10 --centralized_vf --max_num_agents 200 \
#--terminal_reward --num_sample_seconds 0.0 --imitate --num_imitation_iters 0 --final_imitation_weight 1.0 --hard_negative_mining --mining_frac 0.1 --simple_env" \
#--start --stop --cluster-name ev_2pen_10 --tmux

####################################################################################################################################################
## 3/09/20 exps

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 0pen_i2400_imitate_term_f1p0_hard_senv --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 1 --centralized_vf --max_num_agents 200 \
#--terminal_reward --num_sample_seconds 0.0 --imitate --num_imitation_iters 30 --final_imitation_weight 1.0 --hard_negative_mining --mining_frac 0.1 --simple_env" \
#--start --stop --cluster-name ev_0pen_1 --tmux

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 0pen_i2400_imitate_term_f0p5_hard_senv --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 1 --centralized_vf --max_num_agents 200 \
#--terminal_reward --num_sample_seconds 0.0 --imitate --num_imitation_iters 30 --final_imitation_weight 0.5 --hard_negative_mining --mining_frac 0.1 --simple_env" \
#--start --stop --cluster-name ev_0pen_2 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 0pen_i2400_imitate_term_f0p1_hard_senv --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 1 --centralized_vf --max_num_agents 200 \
#--terminal_reward --num_sample_seconds 0.0 --imitate --num_imitation_iters 30 --final_imitation_weight 0.1 --hard_negative_mining --mining_frac 0.1 --simple_env" \
#--start --stop --cluster-name ev_0pen_3 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 0pen_i2400_imitate_term_f0p01_hard_senv --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 1 --centralized_vf --max_num_agents 200 \
#--terminal_reward --num_sample_seconds 0.0 --imitate --num_imitation_iters 30 --final_imitation_weight 0.01 --hard_negative_mining --mining_frac 0.1 --simple_env" \
#--start --stop --cluster-name ev_0pen_4 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 0pen_i2400_imitate_term_f0p1_hard_senv_iter --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 1 --centralized_vf --max_num_agents 200 \
#--terminal_reward --num_sample_seconds 0.0 --imitate --num_imitation_iters 0 --final_imitation_weight 0.1 --hard_negative_mining --mining_frac 0.1 --simple_env" \
#--start --stop --cluster-name ev_0pen_5 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 0pen_i2400_imitate_term_f0p1_hard_iter --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 8 --use_s3 --rollout_scale_factor 1.0 --horizon 1000 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 2 --centralized_vf --max_num_agents 200 \
#--terminal_reward --num_sample_seconds 0.0 --imitate --num_imitation_iters 0 --final_imitation_weight 0.1 --hard_negative_mining --mining_frac 0.1" \
#--start --stop --cluster-name ev_0pen_6 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 0pen_i2400_imitate_term_f0p02_hard_iter --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 8 --use_s3 --rollout_scale_factor 1.0 --horizon 1000 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 2 --centralized_vf --max_num_agents 200 \
#--terminal_reward --num_sample_seconds 0.0 --imitate --num_imitation_iters 0 --final_imitation_weight 0.02 --hard_negative_mining --mining_frac 0.1" \
#--start --stop --cluster-name ev_0pen_7 --tmux

####################################################################################################################################################
## 3/10/20 exps

# Idea here is to try making the imitation weight of the same magnitude as the policy loss
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py i2400_im_term_f0p001_hard_senv_iter10_e0p01 --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 1 --centralized_vf --max_num_agents 200 \
#--terminal_reward --num_sample_seconds 0.0 --imitate --num_imitation_iters 10 --final_imitation_weight 0.001 --hard_negative_mining --mining_frac 0.1 --simple_env --entropy_coeff 0.01" \
#--start --stop --cluster-name ev_0pen_8 --tmux
#
## Idea here is to try making the imitation weight of the 10% magnitude as the policy loss
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py i2400_im_term_f0p0001_hard_senv_iter10_e0p01 --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 1 --centralized_vf --max_num_agents 200 \
#--terminal_reward --num_sample_seconds 0.0 --imitate --num_imitation_iters 10 --final_imitation_weight 0.0001 --hard_negative_mining --mining_frac 0.1 --simple_env --entropy_coeff 0.01" \
#--start --stop --cluster-name ev_0pen_9 --tmux

# Idea here is to try making the imitation weight of the same magnitude as the policy loss
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py i2400_im_term_f0p001_hard_senv_iter10_e0p1 --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 1 --centralized_vf --max_num_agents 200 \
#--terminal_reward --num_sample_seconds 0.0 --imitate --num_imitation_iters 10 --final_imitation_weight 0.001 --hard_negative_mining --mining_frac 0.1 --simple_env --entropy_coeff 0.1" \
#--start --stop --cluster-name ev_0pen_10 --tmux
#
## Idea here is to try making the imitation weight of the 10% magnitude as the policy loss
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py i2400_im_term_f0p0001_hard_senv_iter10_e0p1 --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 1 --centralized_vf --max_num_agents 200 \
#--terminal_reward --num_sample_seconds 0.0 --imitate --num_imitation_iters 10 --final_imitation_weight 0.0001 --hard_negative_mining --mining_frac 0.1 --simple_env --entropy_coeff 0.1" \
#--start --stop --cluster-name ev_0pen_11 --tmux

####################################################################################################################################################
## 3/11/20 exps

# minor entropy bonus to allow the policies to actually learn something

# Idea here is to try making the imitation weight of the same magnitude as the policy loss
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py i2400_im_term_f0p001_hard_iter10_e0p01 --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 6 --use_s3 --rollout_scale_factor 1.0 --horizon 1000 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 2 --centralized_vf --max_num_agents 200 \
#--terminal_reward --num_sample_seconds 0.0 --imitate --num_imitation_iters 10 --final_imitation_weight 0.001 --hard_negative_mining --mining_frac 0.1 --entropy_coeff 0.01" \
#--start --stop --cluster-name ev_0pen_10 --tmux
#
## Idea here is to try making the imitation weight of the 10% magnitude as the policy loss
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py i2400_im_term_f0p0001_hard_iter10_e0p01 --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 6 --use_s3 --rollout_scale_factor 1.0 --horizon 1000 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 2 --centralized_vf --max_num_agents 200 \
#--terminal_reward --num_sample_seconds 0.0 --imitate --num_imitation_iters 10 --final_imitation_weight 0.0001 --hard_negative_mining --mining_frac 0.1 --entropy_coeff 0.01" \
#--start --stop --cluster-name ev_0pen_11 --tmux

####################################################################################################################################################
## 3/12/20 exps

# rerun of 3/11 experiments but with bigger sgd batchsizes for speed

# minor entropy bonus to allow the policies to actually learn something

# Idea here is to try making the imitation weight of the same magnitude as the policy loss
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py i2400_im_term_f0p001_hard_iter10_e0p01 --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 6 --use_s3 --rollout_scale_factor 1.0 --horizon 1000 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 2 --centralized_vf --max_num_agents 200 \
#--terminal_reward --num_sample_seconds 0.0 --imitate --num_imitation_iters 10 --final_imitation_weight 0.001 --hard_negative_mining --mining_frac 0.1 --entropy_coeff 0.01" \
#--start --stop --cluster-name ev_0pen_10 --tmux
#
## Idea here is to try making the imitation weight of the 10% magnitude as the policy loss
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py i2400_im_term_f0p0001_hard_iter10_e0p01 --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 6 --use_s3 --rollout_scale_factor 1.0 --horizon 1000 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 2 --centralized_vf --max_num_agents 200 \
#--terminal_reward --num_sample_seconds 0.0 --imitate --num_imitation_iters 10 --final_imitation_weight 0.0001 --hard_negative_mining --mining_frac 0.1 --entropy_coeff 0.01" \
#--start --stop --cluster-name ev_0pen_11 --tmux

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py i2400_im_term_f0p001_hard_iter0_e0 --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 6 --use_s3 --rollout_scale_factor 1.0 --horizon 1000 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 2 --centralized_vf --max_num_agents 200 \
#--terminal_reward --num_sample_seconds 0.0 --imitate --num_imitation_iters 0 --final_imitation_weight 0.001 --hard_negative_mining --mining_frac 0.1" \
#--start --stop --cluster-name ev_0pen_12 --tmux
#
## Idea here is to try making the imitation weight of the 10% magnitude as the policy loss
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py i2400_im_term_f0p0001_hard_iter0_e0 --num_iters 450 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --grid_search --n_cpus 6 --use_s3 --rollout_scale_factor 1.0 --horizon 1000 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 2 --centralized_vf --max_num_agents 200 \
#--terminal_reward --num_sample_seconds 0.0 --imitate --num_imitation_iters 0 --final_imitation_weight 0.0001 --hard_negative_mining --mining_frac 0.1" \
#--start --stop --cluster-name ev_0pen_13 --tmux

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py i2400_term --num_iters 450 --checkpoint_freq 50 --av_frac 0.1 \
#--num_samples 1 --grid_search --n_cpus 19 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 2 --max_num_agents 200 \
#--terminal_reward --num_sample_seconds 0.0" \
#--start --stop --cluster-name ev_0pen_10 --tmux

####################################################################################################################################################
## 3/19/20 exps
# Hack to force the actions to their extremes so that it's more likely the vehicles learn to stop
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py i2400_no_out_rew_true_pe10 --num_iters 350 --checkpoint_freq 50 --av_frac 0.2 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 2.0 --horizon 200 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 10 --terminal_reward --num_sample_seconds 0.0 --post_exit_rew_len 10" \
#--start --stop --cluster-name ev_2pen_1 --tmux

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py i2400_no_out_rew_true_pe40 --num_iters 350 --checkpoint_freq 50 --av_frac 0.2 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 2.0 --horizon 200 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 10 --terminal_reward --num_sample_seconds 0.0 --post_exit_rew_len 40" \
#--start --stop --cluster-name ev_2pen_2 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py i2400_no_out_rew_true_pe80 --num_iters 350 --checkpoint_freq 50 --av_frac 0.2 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 2.0 --horizon 200 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 10 --terminal_reward --num_sample_seconds 0.0 --post_exit_rew_len 80" \
#--start --stop --cluster-name ev_2pen_3 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 2pen_i2400_no_out_rew_true_ns10_lp2 --num_iters 350 --checkpoint_freq 50 --av_frac 0.2 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 2.0 --horizon 200 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 2.0 --create_inflow_graph --sims_per_step 10 --num_sample_seconds 10.0" \
#--start --stop --cluster-name ev_2pen_4 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py i2400_no_out_rew_true_pe10_senv --num_iters 350 --checkpoint_freq 50 --av_frac 0.2 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 2.0 --horizon 200 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 10 --terminal_reward --num_sample_seconds 0.0 --post_exit_rew_len 10 --simple_env" \
#--start --stop --cluster-name ev_2pen_5 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py i2400_no_out_rew_true_pe40_senv --num_iters 350 --checkpoint_freq 50 --av_frac 0.2 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 2.0 --horizon 200 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 10 --terminal_reward --num_sample_seconds 0.0 --post_exit_rew_len 40 --simple_env" \
#--start --stop --cluster-name ev_2pen_6 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py i2400_no_out_rew_true_pe80_senv --num_iters 350 --checkpoint_freq 50 --av_frac 0.2 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 2.0 --horizon 200 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 10 --terminal_reward --num_sample_seconds 0.0 --post_exit_rew_len 80 --simple_env" \
#--start --stop --cluster-name ev_2pen_7 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py 2pen_i2400_no_out_rew_true_ns10_lp2_senv --num_iters 350 --checkpoint_freq 50 --av_frac 0.2 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 2.0 --horizon 200 --low_inflow 2400 --high_inflow 2400 --aggregate_info \
#--multi_node --sim_step 0.5 --life_penalty 2.0 --create_inflow_graph --sims_per_step 10 --num_sample_seconds 10.0 --simple_env" \
#--start --stop --cluster-name ev_2pen_8 --tmux

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py i2400_no_out_rew_true_pe5_senv --num_iters 350 --checkpoint_freq 50 --av_frac 0.2 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 2.0 --horizon 200 --low_inflow 2400 --high_inflow 2400 --aggregate_info --simple_env \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 10 --terminal_reward --num_sample_seconds 0.0 --post_exit_rew_len 5 \
#--curriculum --num_curr_iters 100 --min_horizon 40" \
#--start --stop --cluster-name ev_0pen_1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py i2400_no_out_rew_true_pe2_senv --num_iters 350 --checkpoint_freq 50 --av_frac 0.2 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 2.0 --horizon 200 --low_inflow 2400 --high_inflow 2400 --aggregate_info --simple_env \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 10 --terminal_reward --num_sample_seconds 0.0 --post_exit_rew_len 2 \
#--curriculum --num_curr_iters 100 --min_horizon 40" \
#--start --stop --cluster-name ev_0pen_2 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py i2400_no_out_rew_true_pe10_senv --num_iters 350 --checkpoint_freq 50 --av_frac 0.2 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 2.0 --horizon 200 --low_inflow 2400 --high_inflow 2400 --aggregate_info --simple_env \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 10 --terminal_reward --num_sample_seconds 0.0 --post_exit_rew_len 10 \
#--curriculum --num_curr_iters 100 --min_horizon 40" \
#--start --stop --cluster-name ev_0pen_3 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py i2400_no_out_rew_true_pe5_senv_cvf --num_iters 350 --checkpoint_freq 50 --av_frac 0.2 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 2.0 --horizon 200 --low_inflow 2400 --high_inflow 2400 --aggregate_info --simple_env \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 10 --terminal_reward --num_sample_seconds 0.0 --post_exit_rew_len 5 \
#--curriculum --num_curr_iters 100 --min_horizon 40 --centralized_vf" \
#--start --stop --cluster-name ev_0pen_4 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py i2400_no_out_rew_true_pe2_senv_cvf --num_iters 350 --checkpoint_freq 50 --av_frac 0.2 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 2.0 --horizon 200 --low_inflow 2400 --high_inflow 2400 --aggregate_info --simple_env \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 10 --terminal_reward --num_sample_seconds 0.0 --post_exit_rew_len 2 \
#--curriculum --num_curr_iters 100 --min_horizon 40 --centralized_vf" \
#--start --stop --cluster-name ev_0pen_5 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py i2400_no_out_rew_true_pe10_senv_cvf --num_iters 350 --checkpoint_freq 50 --av_frac 0.2 \
#--num_samples 1 --grid_search --n_cpus 12 --use_s3 --rollout_scale_factor 2.0 --horizon 200 --low_inflow 2400 --high_inflow 2400 --aggregate_info --simple_env \
#--multi_node --sim_step 0.5 --life_penalty 0.0 --create_inflow_graph --sims_per_step 10 --terminal_reward --num_sample_seconds 0.0 --post_exit_rew_len 10 \
#--curriculum --num_curr_iters 100 --min_horizon 40 --centralized_vf" \
#--start --stop --cluster-name ev_0pen_6 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py i2400_td3senv --num_iters 350 --checkpoint_freq 50 --av_frac 0.2 \
#--num_samples 1 --grid_search --use_s3 --rollout_scale_factor 2.0 --horizon 200 --low_inflow 2400 --high_inflow 2400 --aggregate_info --simple_env \
#--multi_node --sim_step 0.5 --life_penalty 2.0 --create_inflow_graph --sims_per_step 10 --num_sample_seconds 10.0 \
#--curriculum --num_curr_iters 100 --min_horizon 40 --td3" \
#--start --stop --cluster-name ev_0pen_7 --tmux

# 3/21
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py i2400_td3senv --num_iters 350 --checkpoint_freq 50 --av_frac 0.2 \
#--num_samples 1 --rollout_scale_factor 1.0 --horizon 200 --low_inflow 2400 --high_inflow 2400 --aggregate_info --simple_env \
#--sim_step 0.5 --life_penalty 2.0 --create_inflow_graph --sims_per_step 10 --num_sample_seconds 10.0 \
#--curriculum --num_curr_iters 100 --min_horizon 40 --td3 --grid_search" \
#--start --stop --cluster-name ev_0pen_8 --tmux

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py i2400_td3senv_ncrit8 --num_iters 350 --checkpoint_freq 50 --av_frac 0.2 \
#--num_samples 1 --rollout_scale_factor 1.0 --horizon 400 --low_inflow 2400 --high_inflow 2400 --aggregate_info --simple_env \
#--sim_step 0.5 --create_inflow_graph --sims_per_step 5 --rew_n_crit 8 \
#--td3 --grid_search --use_s3" \
#--start --stop --cluster-name ev_0pen_9 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py i2400_td3senv_ncrit10 --num_iters 350 --checkpoint_freq 50 --av_frac 0.2 \
#--num_samples 1 --rollout_scale_factor 1.0 --horizon 400 --low_inflow 2400 --high_inflow 2400 --aggregate_info --simple_env \
#--sim_step 0.5 --create_inflow_graph --sims_per_step 5 --rew_n_crit 10 \
#--td3 --grid_search --use_s3" \
#--start --stop --cluster-name ev_0pen_10 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py i2400_td3senv_ncrit12 --num_iters 350 --checkpoint_freq 50 --av_frac 0.2 \
#--num_samples 1 --rollout_scale_factor 1.0 --horizon 400 --low_inflow 2400 --high_inflow 2400 --aggregate_info --simple_env \
#--sim_step 0.5 --create_inflow_graph --sims_per_step 5 --rew_n_crit 12 \
#--td3 --grid_search --use_s3" \
#--start --stop --cluster-name ev_0pen_11 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py i2400_td3senv_ncrit14 --num_iters 350 --checkpoint_freq 50 --av_frac 0.2 \
#--num_samples 1 --rollout_scale_factor 1.0 --horizon 400 --low_inflow 2400 --high_inflow 2400 --aggregate_info --simple_env \
#--sim_step 0.5 --create_inflow_graph --sims_per_step 5 --rew_n_crit 14 \
#--td3 --grid_search --use_s3" \
#--start --stop --cluster-name ev_0pen_12 --tmux

#####################################################
# 3/22, simple env w/ reduced observation space sweep

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py l800_h3000_td3senv_ncrit12_av0p1 --num_iters 200 --checkpoint_freq 50 --av_frac 0.1 \
#--num_samples 1 --horizon 2000 --low_inflow 800 --high_inflow 3000 --simple_env \
#--sim_step 0.5 --create_inflow_graph --sims_per_step 1 --rew_n_crit 12 \
#--td3 --grid_search --use_s3" \
#--start --stop --cluster-name ev_0pen1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py l800_h3000_td3senv_ncrit12_av0p2 --num_iters 200 --checkpoint_freq 50 --av_frac 0.2 \
#--num_samples 1 --horizon 2000 --low_inflow 2400 --high_inflow 2400 --simple_env \
#--sim_step 0.5 --create_inflow_graph --sims_per_step 1 --rew_n_crit 12 \
#--td3 --grid_search --use_s3" \
#--start --stop --cluster-name ev_0pen_2 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py l800_h3000_td3senv_ncrit12_av0p4 --num_iters 200 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 1 --horizon 2000 --low_inflow 2400 --high_inflow 2400 --simple_env \
#--sim_step 0.5 --create_inflow_graph --sims_per_step 1 --rew_n_crit 12 \
#--td3 --grid_search --use_s3" \
#--start --stop --cluster-name ev_0pen_3 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py l800_h3000_td3senv_ncrit12_av1p0 --num_iters 200 --checkpoint_freq 50 --av_frac 1.0 \
#--num_samples 1 --horizon 2000 --low_inflow 2400 --high_inflow 2400 --simple_env \
#--sim_step 0.5 --create_inflow_graph --sims_per_step 1 --rew_n_crit 12 \
#--td3 --grid_search --use_s3" \
#--start --stop --cluster-name ev_0pen_4 --tmux

#######################################################################################
## 3/31
## Rerunning code to check that it works
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py i2400_td3senv_ncrit12 --num_iters 350 --checkpoint_freq 50 --av_frac 0.2 \
#--num_samples 1 --rollout_scale_factor 1.0 --horizon 400 --low_inflow 2400 --high_inflow 2400 --aggregate_info --simple_env \
#--sim_step 0.5 --create_inflow_graph --sims_per_step 5 --rew_n_crit 12 \
#--td3 --grid_search --use_s3" \
#--start --stop --cluster-name ev_0pen_11 --tmux

######################################################################################
# 4/01
# Rerunning code to check that it works

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py i2400_td3senv_ncrit12_0p1 --num_iters 350 --checkpoint_freq 50 --av_frac 0.1 \
#--num_samples 1 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2400 --high_inflow 2400 --aggregate_info --simple_env \
#--sim_step 0.5 --create_inflow_graph --sims_per_step 1 --rew_n_crit 12 \
#--td3 --grid_search --use_s3" \
#--start --stop --cluster-name ev_0pen_10 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py i2400_td3senv_ncrit12_0p2 --num_iters 350 --checkpoint_freq 50 --av_frac 0.2 \
#--num_samples 1 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2400 --high_inflow 2400 --aggregate_info --simple_env \
#--sim_step 0.5 --create_inflow_graph --sims_per_step 1 --rew_n_crit 12 \
#--td3 --grid_search --use_s3" \
#--start --stop --cluster-name ev_0pen_11 --tmux

ray exec ray_autoscale.yaml \
"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py i2400_td3senv_ncrit12_0p1 --num_iters 350 --checkpoint_freq 50 --av_frac 0.1 \
--num_samples 1 --rollout_scale_factor 1.0 --horizon 400 --low_inflow 2400 --high_inflow 2400 --aggregate_info --simple_env \
--sim_step 0.5 --create_inflow_graph --sims_per_step 5 --rew_n_crit 12 \
--td3 --grid_search --use_s3" \
--start --stop --cluster-name ev_0pen_10 --tmux

ray exec ray_autoscale.yaml \
"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py i2400_td3senv_ncrit12_0p2 --num_iters 350 --checkpoint_freq 50 --av_frac 0.2 \
--num_samples 1 --rollout_scale_factor 1.0 --horizon 400 --low_inflow 2400 --high_inflow 2400 --aggregate_info --simple_env \
--sim_step 0.5 --create_inflow_graph --sims_per_step 5 --rew_n_crit 12 \
--td3 --grid_search --use_s3" \
--start --stop --cluster-name ev_0pen_11 --tmux

ray exec ray_autoscale.yaml \
"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py i2400_td3senv_ncrit12_0p4 --num_iters 350 --checkpoint_freq 50 --av_frac 0.4 \
--num_samples 1 --rollout_scale_factor 1.0 --horizon 400 --low_inflow 2400 --high_inflow 2400 --aggregate_info --simple_env \
--sim_step 0.5 --create_inflow_graph --sims_per_step 5 --rew_n_crit 12 \
--td3 --grid_search --use_s3" \
--start --stop --cluster-name ev_0pen_12 --tmux