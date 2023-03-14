import math
from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import Tuple


class Block(nn.Module):
    def __init__(self, args):
        super(Block, self).__init__()


##################################################
## Basis Expansion Components
##################################################
class IdentityComponent(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int):
        super(IdentityComponent, self).__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        -------Arguments-------
        theta: (B, V, backcast_size + forecast_size)
        --------Outputs--------
        backcast: (B, V, backcast_size)
        forecast: (B, V, forecast_size)
        """
        backcast, forecast = theta[:, :, :self.backcast_size], theta[:, :, self.backcast_size:]
        assert forecast.shape[-1] == self.forecast_size
        return backcast, forecast


class TimeComponent(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int, deg: int, basis_type: str):
        super(TimeComponent, self).__init__()
        self.sizes = [backcast_size, forecast_size]
        self.deg = deg
        self.basis_type = basis_type
        self.backcast_basis, self.forecast_basis = self.get_basis()

    def get_basis(self) -> torch.Tensor:
        time_vecs = [
            np.arange(size, dtype=float) / size 
            for size in self.sizes]
        if self.basis_type == "trend":
            return (
                nn.Parameter(torch.tensor(
                np.concatenate([
                np.power(time_vec, p)[np.newaxis, :]
                for p in range(1, self.deg + 1)
                # for p in range(self.deg + 1)
                ]), dtype=torch.float), requires_grad=False)
                for time_vec in time_vecs)
        elif self.basis_type == "seasonality":
            deg_half = self.deg // 2
            return (
                nn.Parameter(torch.Tensor(
                np.stack(
                # [np.ones(self.sizes[i])] + 
                [np.cos(2*np.pi*p*time_vec) for p in range(1, deg_half)] +
                [np.sin(2*np.pi*p*time_vec) for p in range(1, deg_half)]
                ), dtype=torch.float), requires_grad=False)
                for time_vec in time_vecs
                # for i, time_vec in enumerate(time_vecs)
                )
        
    def forward(self, theta:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        -------Arguments-------
        theta: (B, V, 2P)
        self.backcast_basis: (P, backcast_size)
        self.forecast_basis: (P, forecast_size)
        --------Outputs--------
        backcast: (B, V, backcast_size)
        forecast: (B, V, forecast_size)
        """
        poly_dim = len(self.backcast_basis)
        assert poly_dim == len(self.forecast_basis)
        backcast = torch.einsum("bvp,pt->bvt", theta[:, :, :poly_dim], self.backcast_basis)
        forecast = torch.einsum("bvp,pt->bvt", theta[:, :, poly_dim:], self.forecast_basis)
        return backcast, forecast























