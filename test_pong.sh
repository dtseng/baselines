#!/bin/bash

python3 train_pong.py pong_results/prior_exploration_0_50_exp2/run1 softq 1e-4 0.50 exp_prior

python3 train_pong.py pong_results/act_exploration_epsilon_0_50/run0 softq 1e-4 0.50 exp_act_eps

python3 train_pong.py pong_results/act_exploration_epsilon_0_10/run0 softq 1e-4 0.10 exp_act_eps

python3 train_pong.py pong_results/prior_exploration_0_10_exp2/run1 softq 1e-4 0.10 exp_prior

python3 train_pong.py pong_results/act_exploration_softmax/run0 softq 1e-4 0.10 exp_act_softmax
