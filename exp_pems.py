import os
import torch
import numpy as np
import argparse
from model.gnbeats import GNBEATS
from utils.util_functions import init_seed, init_params


parser = argparse.ArgumentParser(description="GNBEATS on PEMS dataset")

parser.add_argument('--dataset', type=str, default='PEMS07', choices=['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08']) 
parser.add_argument('--norm_method', type=str, default='z_score')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--seed', type=int, default=4321)

parser.add_argument('--window', type=int, default=12)
parser.add_argument('--horizon', type=int, default=12)
parser.add_argument('--embed_dim', type=int, default=8)

parser.add_argument('--train_percent', type=float, default=0.6)
parser.add_argument('--valid_percent', type=float, default=0.2)
parser.add_argument('--loss', type=str, default='mse')

# Whether to perform node modifications
parser.add_argument('--node_mod', action=argparse.BooleanOptionalAction)
# Whether to share parameters across blocks
parser.add_argument('--share_params', action=argparse.BooleanOptionalAction)
# Regularization settings
parser.add_argument('--block_dropout_prob', type=float, default=0)
parser.add_argument('--l1_penalty_linear', type=float, default=1e-5)
parser.add_argument('--l1_penalty_conv', type=float, default=1e-5)
# Ordering of blocks
parser.add_argument('--ar_block_order', nargs='+', type=int, default=[0, 1, 2, 0, 0, 1, 1, 2, 2])
parser.add_argument('--graph_block_order', nargs='+', type=int, default=[0, 1, 2, 0, 0, 1, 1, 2, 2])
# Degree of time bases
parser.add_argument('--trend_deg', type=int, default=3)
parser.add_argument('--season_deg', type=int, default=12)
# Whether to include a column of 1's in time components
parser.add_argument('--trend_id', action=argparse.BooleanOptionalAction)
parser.add_argument('--season_id', action=argparse.BooleanOptionalAction)
# Which network to use for basis coefficients
parser.add_argument('--ar_theta_net', type=str, default='TCN', choices=['FC', 'TCN', 'SCINet'])
parser.add_argument('--graph_theta_net', type=str, default='TCN')


args = parser.parse_args()

if __name__ == '__main__':
    init_seed(args.seed)
    # model = 
