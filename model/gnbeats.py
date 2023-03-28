import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union, Dict
from .layers_temporal import ParallelConv1d, ParallelCausalConv1d, NodeDependentMod, FullyConnectedNet
from .tcn import TCN
from .scinet import StackedSCINet
from .layers_graph import UndirGraphConv, MagGraphConv, MPGraphConv


class Block(nn.Module):
    """
    backcast_size: Length of backcasts, aka window, W
    forecast_size: Length of forecasts, aka horizon, H
    embed_dim: Embedded dimension D, node embeddings have shape (V, 2*D)
    basis: One of "identity", "trend", and "season"
    node_mod: Whether to modify the input by node embeddings in forward()
    kwargs: Dictionary including the following arguments, where (val) means optional with default value "val" 
        - basis="trend": deg, include_identity (True)
        - basis="season": deg (None), include_identity (True)
    """
    def __init__(
            self, backcast_size: int, forecast_size: int, embed_dim: int, basis: str, 
            node_mod: bool=True, kwargs: Union[Dict, None]=None): 
            # kwargs instead of **kwargs to enable pass by reference 
            # (some elements will be popped before kwargs is being used in children classes)
        super().__init__()
        self.node_mod = node_mod
        if node_mod:
            self.in_shape = (2*embed_dim, backcast_size) # (2*D, W)
            self.in_channels = 2*embed_dim # 2*D
        else:
            self.in_shape = backcast_size # W
            self.in_channels = 1
        # Basis type
        if basis == "identity":
            self.out_shape = backcast_size + forecast_size # W + H
            self.basis = IdentityComponent(backcast_size, forecast_size)
        else:
            basis_args = {arg: kwargs.pop(arg) for arg in ("deg", "include_identity") if arg in kwargs}
            poly_dim = basis_args["deg"]
            if basis == "trend":
                assert "deg" in basis_args
                self.basis = TrendComponent(backcast_size, forecast_size, **basis_args)
            elif basis == "season":
                poly_dim -= 2 if poly_dim % 2 == 0 else 3
                self.basis = SeasonalityComponent(backcast_size, forecast_size, **basis_args)
            if "include_identity" not in basis_args or basis_args["include_identity"]:
                poly_dim += 1
            self.out_shape = 2*poly_dim # 2*P


class AutoregressiveBlock(Block):
    """
    theta_net: One of "FC", "TCN", and "SCINet"
    kwargs: Dictionary including the following arguments, where (val) means optional with default value "val" 
        - basis="trend"/"season": deg, include_identity as in "Block"
        - theta_net="FC": hidden_dims, activation ("selu"), batch_norm (False), dropout_prob (0)
        - theta_net="TCN": kernel_size, hidden_channels (None), padding_mode ("zeros"), dropout_prob (0.2)
        - theta_net="SCINet": hidden_channels, kernel_sizes, num_stacks (2), num_levels (3), 
            padding_mode ("replicate"), causal_conv (True), dropout_prob (0.25)
    """
    def __init__(
            self, backcast_size: int, forecast_size: int, embed_dim: int, 
            basis: str, theta_net: str, node_mod: bool=True, **kwargs):
        super().__init__(backcast_size, forecast_size, embed_dim, basis, node_mod, kwargs)
        if theta_net == "FC": # N-BEATS originial
            self.theta_net = FullyConnectedNet(self.in_shape, self.out_shape, **kwargs)
        elif theta_net == "TCN":
            self.theta_net = TCN(
                self.in_channels, backcast_size, out_shape=self.out_shape, **kwargs)
        elif theta_net == "SCINet":
            self.theta_net = StackedSCINet(
                self.in_channels, backcast_size, out_shape=self.out_shape, **kwargs)
    
    def forward(
            self, x:torch.Tensor, node_embeddings: Union[torch.Tensor, None]=None
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        -------Arguments-------
        x: (B, V, W)
        node_embeddings: (V, 2*D)
        --------Outputs--------
        backcast: (B, V, W)
        forecast: (B, V, H)
        """
        if self.node_mod:
            assert node_embeddings is not None and node_embeddings.shape == (x.shape[1], self.in_channels)
            x = torch.einsum("bvt,vd->bvdt", x, node_embeddings) # (B, V, 2*D, W)
        theta = self.theta_net(x) # (B, V, W + H)
        return self.basis(theta) # (B, V, W), (B, V, H)


class GraphBlock(Block):
    """
    theta_net: One of "FC", "TCN", and "SCINet"
    adj_mat_activation, update_activation, self_loops, thresh, normalization: 
        Defined under MPGraphConv in layers_graph.py
    kwargs: Dictionary including the following arguments, where (val) means optional with default value "val" 
        - basis="trend"/"season": deg, include_identity as in "Block"
        - theta_net="FC": hidden_dims, activation ("selu"), batch_norm (False), dropout_prob (0)
        - theta_net="TCN": kernel_size, hidden_channels (None), padding_mode ("zeros"), dropout_prob (0.2)
        - theta_net="SCINet": hidden_channels, kernel_sizes, num_stacks (2), num_levels (3), 
            padding_mode ("replicate"), causal_conv (True), dropout_prob (0.25)
    """
    def __init__(
            self, backcast_size: int, forecast_size: int, embed_dim: int, 
            basis: str, theta_net: str, node_mod: bool=True, 
            adj_mat_activation: Union[str, None]="tanh", update_activation: Union[str, None]=None, 
            self_loops: bool=False, thresh: float=0.2, normalization: Union[str, None]="frob", **kwargs):
        super().__init__(backcast_size, forecast_size, embed_dim, basis, node_mod, kwargs)
        self.theta_net = MPGraphConv(
            self.in_shape, self.out_shape, embed_dim, theta_net, adj_mat_activation, update_activation, 
            self_loops, thresh, normalization, **kwargs)
        
    def forward(
            self, x:torch.Tensor, node_embeddings: Union[torch.Tensor, None]=None
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        -------Arguments-------
        x: (B, V, W)
        node_embeddings: (V, 2*D)
        --------Outputs--------
        backcast: (B, V, W)
        forecast: (B, V, H)
        """
        if self.node_mod:
            x = torch.einsum("bvt,vd->bvdt", x, node_embeddings) # (B, V, 2*D, W)
        theta = self.theta_net(x, node_embeddings) # (B, V, W + H)
        return self.basis(theta) # (B, V, W), (B, V, H)


##################################################
## Basis Expansion: Identity
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
            P = deg - 1 (even) / deg - 2 (odd)  if include_identity
                deg - 2 (even) / deg - 3 (odd)  else
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
        if not deg:  # N-BEATS default: deg = forecast_size
            deg = forecast_size
        assert include_identity or (deg > 3), "Empty basis"
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






