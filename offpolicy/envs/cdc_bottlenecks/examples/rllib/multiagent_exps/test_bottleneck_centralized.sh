#!/usr/bin/env bash

python multi_bottleneck_centralized.py test --num_iters 30 --checkpoint_freq 50 \
    --num_samples 1 --n_cpus 2 --rollout_scale_factor 1.0 --horizon 500