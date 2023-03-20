from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union
from .layers_temporal import ParallelConv1d, ParallelCausalConv1d, NodeDependentMod, FullyConnectedNet
from .tcn import TCN
from .scinet import StackedSCINet
from .layers_graph import UndirGraphConv, MagGraphConv


class Block(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Node dependent modification
        self.node_mod = NodeDependentMod() if args.node_mod else None
        # Theta network
        if args.theta_net == "FC":
            self.theta_net = FullyConnectedNet(*args.theta_net_args)
        elif args.theta_net == "TCN":
            self.theta_net = TCN(*args.theta_net_args)
        elif args.theta_net == "SCINet":
            self.theta_net = StackedSCINet(*args.theta_net_args)

    def forward(
            self, x:torch.Tensor, node_embeddings: torch.Tensor
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        -------Arguments-------
        x: (B, V, W)
        node_embeddings: (V, D)
        --------Outputs--------
        backcast: (B, V, W)
        forecast: (B, V, H)
        """
        theta = self.theta_net(x) # (B, V)


##################################################
## Basis Expansion: Identity (AR)
##################################################
class IdentityComponent(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size # W
        self.forecast_size = forecast_size # H

    def forward(self, theta:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        -------Arguments-------
        theta: (B, V, W + H)
        --------Outputs--------
        backcast: (B, V, W)
        forecast: (B, V, H)
        """
        backcast, forecast = theta[:, :, :self.backcast_size], theta[:, :, self.backcast_size:]
        assert forecast.shape[-1] == self.forecast_size
        return backcast, forecast


##################################################
## Basis Expansion: Time
##################################################
class TimeComponent(nn.Module):
    def __init__(
            self, backcast_size: int, forecast_size: int, deg: int, include_identity: bool):
        super().__init__()
        self.sizes = (backcast_size, forecast_size) # (W, H)
        self.deg = deg
        self.include_identity = include_identity
        self.time_vecs = [np.arange(size, dtype=float) / size for size in self.sizes]
        self.backcast_basis, self.forecast_basis = self.get_basis()

    def get_basis(self):
        pass
        
    def forward(self, theta:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        -------Arguments-------
        theta: (B, V, 2P)
            - Trend
            P = deg + 1  if include_identity 
                deg      else
            - Seasonality
            P = H - 1 (even) / H - 2 (odd)  if include_identity
                H - 2 (even) / H - 3 (odd)  else
        self.backcast_basis: (P, W)
        self.forecast_basis: (P, H)
        --------Outputs--------
        backcast: (B, V, W)
        forecast: (B, V, H)
        """
        poly_dim = len(self.backcast_basis)
        assert poly_dim == len(self.forecast_basis)
        backcast = torch.einsum("bvp,pt->bvt", theta[:, :, :poly_dim], self.backcast_basis)
        forecast = torch.einsum("bvp,pt->bvt", theta[:, :, poly_dim:], self.forecast_basis)
        return backcast, forecast


class TrendComponent(TimeComponent):
    def __init__(self, backcast_size: int, forecast_size: int, deg: int, include_identity: bool=True):
        super().__init__(backcast_size, forecast_size, deg, include_identity)

    def get_basis(self) -> torch.Tensor:
        return (
            nn.Parameter(torch.tensor(
            np.concatenate([
            np.power(time_vec, p)[np.newaxis, :]
            for p in range(0 if self.include_identity else 1, self.deg + 1)
            ]), dtype=torch.float), requires_grad=False)
            for time_vec in self.time_vecs)


class SeasonalityComponent(TimeComponent):
    def __init__(
            self, backcast_size: int, forecast_size: int, 
            deg: Union[int, None]=None, include_identity: bool=True):
        assert include_identity or (deg > 3), "Empty basis"
        if deg is None:  # N-BEATS default: deg = forecast_size
            deg = forecast_size
        super().__init__(
            backcast_size, forecast_size, deg, include_identity)

    def get_basis(self) -> torch.Tensor:
        deg_half = self.deg // 2
        return (
            nn.Parameter(torch.tensor(
            np.stack(
            ([np.ones(self.sizes[i])] if self.include_identity else []) + 
            [np.cos(2*np.pi*p*time_vec) for p in range(1, deg_half)] +
            [np.sin(2*np.pi*p*time_vec) for p in range(1, deg_half)]
            ), dtype=torch.float), requires_grad=False)
            for i, time_vec in enumerate(self.time_vecs)
            )


##################################################
## Basis Expansion: Graph
##################################################




