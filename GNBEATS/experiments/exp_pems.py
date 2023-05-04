import os
import torch
import numpy as np
import argparse
from GNBEATS.model.gnbeats import GNBEATS
from GNBEATS.utils.data_utils import load_data_npz, get_loader_normalizer, get_data_from_loader
from GNBEATS.utils.util_functions import init_seed, plot_decomposition, plot_prediction


parser = argparse.ArgumentParser(description="GNBEATS on PEMS dataset")

parser.add_argument('--dataset', type=str, default='PEMS07', choices=['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08']) 
parser.add_argument('--norm_method', type=str, default='z_score')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--seed', type=int, default=4321)

parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--window', type=int, default=12)
parser.add_argument('--horizon', type=int, default=12)
parser.add_argument('--embed_dim', type=int, default=4)

parser.add_argument('--train_percent', type=float, default=0.6)
parser.add_argument('--valid_percent', type=float, default=0.2)
parser.add_argument('--train_interval', type=int, default=1)
parser.add_argument('--test_interval', type=int, default=1)
parser.add_argument('--loss', type=str, default='mse')
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--early_stop_patience', type=int, default=15)

# Whether to perform node modifications
parser.add_argument('--node_mod', action=argparse.BooleanOptionalAction, default=False)
# Whether to share parameters across blocks
parser.add_argument('--share_params', action=argparse.BooleanOptionalAction, default=False)
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
parser.add_argument('--trend_id', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--season_id', action=argparse.BooleanOptionalAction, default=True)
# Which network to use for basis coefficients
parser.add_argument('--ar_theta_net', type=str, default='TCN', choices=['FC', 'TCN', 'SCINet'])
parser.add_argument('--graph_theta_net', type=str, default='TCN', choices=['FC', 'TCN', 'SCINet'])

# Restrict data size for testing
parser.add_argument('--restrict_node_dim', type=int, default=0)
parser.add_argument('--restrict_time_steps', type=int, default=0)
# Path to load model
# parser.add_argument('--fit', action=argparse.BooleanOptionalAction)
# parser.add_argument('--model_path', type=str, default="models/2023-05-02_1548_exp.pth")



args = parser.parse_args()

if __name__ == '__main__':
    init_seed(args.seed)
    data = load_data_npz(ds_folder="PEMS", ds_name=args.dataset).T # (V, T)
    if args.restrict_node_dim > 0:
        data = data[:args.restrict_node_dim]
    if args.restrict_time_steps > 0:
        data = data[:, :args.restrict_time_steps]
    num_nodes = len(data) # V
    loader, normalizer = get_loader_normalizer(
        data, batch_size=args.batch_size, window=args.window, horizon=args.horizon, 
        train_interval=args.train_interval, test_interval=args.test_interval,
        normalization=args.norm_method, train=args.train_percent, valid=args.valid_percent, 
        device=args.device)
    model = GNBEATS(
        window=args.window, horizon=args.horizon, num_nodes=num_nodes, embed_dim=args.embed_dim, 
        node_mod=args.node_mod, share_params=args.share_params, block_dropout_prob=args.block_dropout_prob, 
        l1_penalty_linear=args.l1_penalty_linear, l1_penalty_conv=args.l1_penalty_conv, 
        ar_block_order=args.ar_block_order, graph_block_order=args.graph_block_order, 
        trend_deg=args.trend_deg, season_deg=args.season_deg, 
        trend_include_id=args.trend_id, season_include_id=args.season_id, 
        ar_theta_net=args.ar_theta_net, graph_theta_net=args.graph_theta_net, 
        device=args.device, loss=args.loss)
    print(f"Model initialized with {model.count_params()} parameters...")
    print("Fitting model...")
    model.fit(loader, normalizer, num_epochs=args.num_epochs, early_stop_patience=args.early_stop_patience)
    # if args.fit:
    #     print("Fitting model...")
    #     model.fit(loader, normalizer, num_epochs=args.num_epochs, early_stop_patience=args.early_stop_patience)
    # else:
    #     print("Loading model...")
    #     model.load_state_dict(torch.load(args.model_path), strict=False)
    print("Testing model...")
    test_output = model.test(loader, normalizer, return_output=False)
    inputs, targets = get_data_from_loader(loader["test"])
    preds, preds_decomp = model.predict(inputs, normalizer=normalizer["test"])
    plot_decomposition(preds_decomp, nodes=range(min(num_nodes, 5)))
    plot_prediction(preds, targets, nodes=range(min(num_nodes, 5)))
