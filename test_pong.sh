#!/bin/bash


CUDA_VISIBLE_DEVICES=1 python3 train_pong.py pong_results/sanity_check/run0 softq 1e-4
CUDA_VISIBLE_DEVICES=1 python3 train_pong.py pong_results/sanity_check/run1 softq 1e-4
