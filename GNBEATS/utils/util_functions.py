import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union, Optional, Dict, List, Type, Iterable
from matplotlib import pyplot as plt
from itertools import product
import time
import random


##################################################
## Complex Activation Functions
##################################################
class ComplexActivation(nn.Module):
    """
    method: One of "real", "real_imag", "arg_bdd", and "phase_amp"
        real (Real):
            1[z.real >= 0]*z
            = ReLU(z.real) + i*1[z.real >= 0]*z.imag
        real_imag (Real-imaginary):
            ReLU(z.real) + i*ReLU(z.imag)
        arg_bdd (Argument bound):
            z if -pi/2 <= arg(z) < pi/2, 0 otherwise
        phase_amp (Phase-amplitude):
            tanh(|z|)*exp(i*arg(z))
    """
    def __init__(self, method: str="arg_bdd"):
        super().__init__()
        self.method = method
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.method == "real":
            return x*(x.real >= 0)
        elif self.method == "real_imag":
            return F.relu(x.real) + 1.j*F.relu(x.imag)
        elif self.method == "arg_bdd":
            return torch.where(
                (-torch.pi/2 <= torch.angle(x)) & (torch.angle(x) < torch.pi/2), x, 0)
        elif self.method == "phase_amp":
            return torch.tanh(x.abs())*torch.exp(1.j*x.angle())


##################################################
## Plotting
##################################################
def plot_decomposition(
    preds_decomp: Union[torch.Tensor, np.array], nodes: Union[Iterable[int], int]=0, 
    batches: Union[Iterable[int], int]=0, plot_path: str="plots"):
    """
    -------Arguments-------
    preds_decomp: (B, V, T, 8) or (V, T, 8)
        preds_decomp[v, :, g] is the gth component for series/node v, where g is
        6 = Static Trend, 7 = Static Seasonality,
        0 = AR Trend, 1 = AR Seasonality, 2 = AR Identity,
        3 = Graph Trend, 4 = Graph Seasonality, 5 = Graph Identity
    nodes: List of nodes in {0, ..., V - 1} to plot
    batches: List of batches in {0, ..., B - 1} to plot
    plot_path: Directory to save the plots to
    --------Outputs--------
    len(nodes)*len(batches) number of plots will be saved to plot_path, 
    each named {current_time}_batch{b}_node{v}_decomp.jpeg
    """
    basis_types = {0: "trend", 1: "season", 2: "identity"}
    coef_types = {0: "AR", 1: "Graph", 2: "Static"}
    current_time = time.strftime("%Y-%m-%d_%H%M", time.localtime(time.time()))
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    if len(preds_decomp.shape) == 3: # (V, T, 8)
        preds_decomp = preds_decomp.reshape(1, *preds_decomp.shape) # (B=1, V, T, 8)
    nodes = [nodes] if not hasattr(nodes, '__len__') else nodes
    batches = [batches] if not hasattr(batches, '__len__') else batches
    for b, v in product(batches, nodes):
        fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(12, 9))
        fig.subplots_adjust(bottom=1, top=2)
        for i in range(8):
            basis, coef = i % 3, i // 3
            ax = axs[basis][coef]
            ax.plot(preds_decomp[b, v, :, i])
            ax.grid(visible=True, which='major', linestyle='--')
            ax.set_title(f"{coef_types[coef]} {basis_types[basis]}")
        fig_name = f"{current_time}_batch{b}_node{v}_decomp.jpeg"
        fig.savefig(os.path.join(plot_path, fig_name), bbox_inches='tight')
    

def plot_prediction(
    preds: Union[torch.Tensor, np.array], targets: Union[torch.Tensor, np.array], 
    nodes: Union[Iterable[int], int]=0, batches: Union[Iterable[int], int]=0, 
    plot_path: str="plots"):
    """
    -------Arguments-------
    preds: (B, V, T) or (V, T)
    targets: (B, V, T) or (V, T)
    nodes: List of nodes in {0, ..., V - 1} to plot
    batches: List of batches in {0, ..., B - 1} to plot
    plot_path: Directory to save the plots to
    --------Outputs--------
    len(nodes)*len(batches) number of plots will be saved to plot_path, 
    each named {current_time}_batch{b}_node{v}.jpeg
    """
    current_time = time.strftime("%Y-%m-%d_%H%M", time.localtime(time.time()))
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    if len(preds.shape) == 2: # (V, T)
        preds = preds.reshape(1, *preds.shape) # (B=1, V, T)
    if len(targets.shape) == 2: # (V, T)
        targets = targets.reshape(1, *targets.shape) # (B=1, V, T)
    assert preds.shape == targets.shape
    nodes = [nodes] if not hasattr(nodes, '__len__') else nodes
    batches = [batches] if not hasattr(batches, '__len__') else batches
    for b, v in product(batches, nodes):
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(9, 5))
        ax.plot(preds[b, v], label="prediction")
        ax.plot(targets[b, v], label="target")
        ax.grid(visible=True, which='major', linestyle='--')
        ax.legend(loc=2)
        fig_name = f"{current_time}_batch{b}_node{v}.jpeg"
        fig.savefig(os.path.join(plot_path, fig_name), bbox_inches='tight')


def visualize_adj_mat():
    None


##################################################
## Initialization & Reproducibility
##################################################
def init_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def init_params(model: nn.Module):
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
        else:
            nn.init.uniform_(param)
