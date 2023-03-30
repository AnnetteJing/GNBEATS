import numpy as np
import torch
import torch.nn as nn
# from torch.nn.utils import weight_norm
from typing import Optional, Union, Tuple
from .layers_temporal import ParallelConv1d, ParallelCausalConv1d


class TemporalBlock(nn.Module):
    def __init__(
            self, in_channels: int, out_channels: int,  kernel_size: int, dilation: int,
            padding_mode: str="zeros", dropout_prob: int=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding_mode = padding_mode
        self.dropout_prob = dropout_prob
        conv1 = ParallelCausalConv1d(
            in_channels, out_channels, kernel_size, 
            dilation=dilation, padding_mode=padding_mode, param_norm="weight")
        conv2 = ParallelCausalConv1d(
            out_channels, out_channels, kernel_size, 
            dilation=dilation, padding_mode=padding_mode, param_norm="weight")
        if dropout_prob > 0:
            self.layers = nn.Sequential(
                conv1, nn.ReLU(), nn.Dropout(dropout_prob), 
                conv2, nn.ReLU(), nn.Dropout(dropout_prob))
        else:
            self.layers = nn.Sequential(conv1, nn.ReLU(), conv2, nn.ReLU())
        # For matching shapes of input & output (may have different numbers of channels)
        if in_channels != out_channels:
            self.downsample = ParallelConv1d(
                in_channels, out_channels, kernel_size=1, padding_mode=padding_mode)
        else:
            self.downsample = None
        # Activation for residual connection
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        -------Arguments-------
        x: (B, V, C_i, T)
        --------Outputs--------
        out: (B, V, C_o, T)
        """
        out = self.layers(x) # (B, V, C_o, T)
        res = x if self.downsample is None else self.downsample(x) # (B, V, C_o, T)
        out = self.relu(out + res)
        return out


class TCN(nn.Module):
    def __init__(
            self, in_channels: int, in_timesteps: int, kernel_size: int, 
            hidden_channels: Optional[list]=None, padding_mode: str="zeros", 
            dropout_prob: int=0.2, out_shape: Union[None, int, Tuple]=None):
        super().__init__()
        self.in_timesteps = in_timesteps # T
        self.channels = [in_channels] # [C_i]
        if not hidden_channels:
            self.channels += [in_channels] # [C_i, C_i]
        else:
            self.channels += hidden_channels  # [C_i, ...]
        self.kernel_size = kernel_size
        self.padding_mode = padding_mode
        self.dropout_prob = dropout_prob
        layers = []
        for i in range(len(self.channels) - 1):
            layers.append(TemporalBlock(
                self.channels[i], self.channels[i + 1], kernel_size, 
                dilation=2**i, padding_mode=padding_mode, dropout_prob=dropout_prob
            ))
        self.layers = nn.Sequential(*layers)
        self.out_shape = out_shape
        if out_shape is not None:
            self.out_layer = nn.Linear(self.channels[-1]*in_timesteps, np.prod(out_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        -------Arguments-------
        x: (B, V, C_i, T) OR (B, V, C_i*T)
        --------Outputs--------
        out: (B, V, hidden_channels[-1], T) OR (B, V, out_shape)
        """
        if len(x.shape) == 3:
            x = x.reshape(x.shape[0], x.shape[1], self.channels[0], self.in_timesteps)
        out = self.layers(x) # (B, V, channels[-1], T)
        if self.out_shape is not None:
            out = self.out_layer(out.flatten(2, 3)) # (B, V, prod(out_shape))
            if hasattr(self.out_shape, '__len__'):
                out = out.reshape(x.shape[0], x.shape[1], *self.out_shape) # (B, V, out_shape)
        return out
    
    