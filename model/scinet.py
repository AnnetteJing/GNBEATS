import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Union
from .layers_temporal import ParallelConv1d, ParallelCausalConv1d, NodeDependentMod


class SubseqTransformation(nn.Module):
    """
    Transformations applied to subsequences before & during the interation phase
    ----------------------
    channels: Input/Output channels; C
    hidden_channels: Hidden channels; C_h
    kernel_size: Kernel size for the 1st and 2nd convolutions; (Ker_1, Ker_2)
    causal_conv: Whether to use causal convolutions
    """
    def __init__(
            self, channels: int, hidden_channels: int, kernel_size: Union[Tuple[int, int], int], 
            padding_mode: str="replicate", causal_conv: bool=True, dropout_prob: float=0.25):
        super().__init__()
        if type(kernel_size) is int:
            kernel_size = (kernel_size, kernel_size)
        if causal_conv:
            conv1 = ParallelCausalConv1d(
                channels, hidden_channels, kernel_size[0], padding_mode=padding_mode)
            conv2 = ParallelCausalConv1d(
                hidden_channels, channels, kernel_size[1], padding_mode=padding_mode)
        else:
            if kernel_size[0] % 2 == 0:
                padding = ((kernel_size[0] - 2) // 2 + 1, kernel_size[0] // 2 + 1)
            else:
                padding = ((kernel_size[0] - 1) // 2 + 1, (kernel_size[0] - 1) // 2 + 1)
            conv1 = ParallelConv1d(
                channels, hidden_channels, kernel_size[0], 
                padding=padding, padding_mode=padding_mode)
            conv2 = ParallelConv1d(hidden_channels, channels, kernel_size[1])
        if dropout_prob > 0:
            self.layers = nn.Sequential(
                conv1, nn.LeakyReLU(), nn.Dropout(dropout_prob), conv2, nn.Tanh())
        else:
            self.layers = nn.Sequential(
                conv1, nn.LeakyReLU(), conv2, nn.Tanh())
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        -------Arguments-------
        x: (B, V, C, T)
        --------Outputs--------
        out: (B, V, C, T)
        """
        return self.layers(x)


class SCIBlock(nn.Module):
    def __init__(
            self, channels: int, hidden_channels: int, kernel_size: Tuple[int, int], 
            padding_mode: str="replicate", causal_conv: bool=True, dropout_prob: float=0.25):
        super().__init__()
        subseq_trans_args = (channels, hidden_channels, kernel_size, padding_mode, causal_conv, dropout_prob)
        self.scale_trans_even = SubseqTransformation(*subseq_trans_args)
        self.scale_trans_odd = SubseqTransformation(*subseq_trans_args)
        self.shift_trans_even = SubseqTransformation(*subseq_trans_args)
        self.shift_trans_odd = SubseqTransformation(*subseq_trans_args)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        -------Arguments-------
        x: (B, V, C, T);
            T is divisible by 2
        --------Outputs--------
        out_even: (B, V, C, T/2)
        out_odd: (B, V, C, T/2)
        """
        x_even, x_odd = x[:, :, :, ::2], x[:, :, :, 1::2]
        assert x_even.shape[-1] == x_odd.shape[-1], "Even & odd shape mismatch"
        x_scaled_even = torch.mul(x_even, torch.exp(self.scale_trans_odd(x_odd)))
        x_scaled_odd = torch.mul(x_odd, torch.exp(self.scale_trans_even(x_even)))
        out_even = x_scaled_even + self.shift_trans_odd(x_scaled_odd)
        out_odd = x_scaled_odd + self.shift_trans_even(x_scaled_even)
        return out_even, out_odd


class LevelSCINet(nn.Module):
    def __init__(
            self, level: int, channels: int, hidden_channels: int, kernel_size: Tuple[int, int], 
            padding_mode: str="replicate", causal_conv: bool=True, dropout_prob: float=0.25):
        super().__init__()
        self.level = level
        self.blocks = [
            SCIBlock(channels, hidden_channels, kernel_size, padding_mode, causal_conv, dropout_prob) 
            for i in range(2**level)]
    
    def forward(self, x_list: list) -> list:
        """
        -------Arguments-------
        x_list: [(B, V, C, T/2**level)] x 2**level
        --------Outputs--------
        out_list: [(B, V, C, T/2**(level + 1))] x 2**(level + 1)
        """
        assert len(x_list) == len(self.blocks), "Input size does not match #blocks"
        out_list = []
        for i, x in enumerate(x_list):
            x_even, x_odd = self.blocks[i](x)
            out_list.extend((x_even, x_odd))
        return out_list


class SCINet(nn.Module):
    """
    num_levels: Number of levels; L
    in_channels: Input channels; C_i
    in_timesteps: Input timesteps; T_i
    hidden_channels: Hidden channels for each layer
    out_shape: Output shape
    """
    def __init__(
            self, num_levels: int, in_channels: int, in_timesteps: int, 
            hidden_channels: Union[int, list], out_shape: Union[int, Tuple], 
            kernel_size: Tuple[int, int], padding_mode: str="replicate", 
            causal_conv: bool=True, dropout_prob: float=0.25):
        super().__init__()
        self.num_levels = num_levels # L
        self.in_channels = in_channels # C_i
        self.in_timesteps = in_timesteps # T_i
        self.out_shape = out_shape
        if not hasattr(hidden_channels, '__len__'): # [C_{h1}, ..., C_{hL}]
            hidden_channels = [hidden_channels]*num_levels
        assert len(hidden_channels) == num_levels, "Hidden channel & level mismatch"
        self.kernel_size = kernel_size # (Ker_1, Ker_2)
        self.padding_mode = padding_mode
        self.causal_conv = causal_conv
        self.dropout_prob = dropout_prob
        self.subseq_idx = self.get_subseq_idx([i for i in range(2**num_levels)], num_levels)
        self.levels = [
            LevelSCINet(
                l, in_channels, hidden_channels[l], kernel_size, 
                padding_mode, causal_conv, dropout_prob)
            for l in range(num_levels)]
        if hasattr(out_shape, '__len__'):
            self.output_layer = nn.Linear(in_channels*in_timesteps, np.prod(out_shape))
        else:
            self.output_layer = nn.Linear(in_channels*in_timesteps, out_shape)

    def get_subseq_idx(self, seq: list, level: int) -> list:
        if level == 0:
            return seq
        seq_even = self.get_subseq_idx(seq[::2], level - 1)
        seq_odd = self.get_subseq_idx(seq[1::2], level - 1)
        return seq_even + seq_odd
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        -------Arguments-------
        x: (B, V, C_i, T_i)
        --------Outputs--------
        out: (B, V, *out_shape)
        """
        x_list = [x]
        for level in self.levels:
            x_list = level(x_list) # [(B, V, C_i, T_i/2**num_levels)] x 2**num_levels
        assert len(x_list) == 2**self.num_levels
        out_list = []
        for t in range(x_list[0].shape[-1]):
            for i in self.subseq_idx:
                out_list.append(x_list[i][:, :, :, t])
        out = torch.stack(out_list, dim=-1) # (B, V, C_i, T_i)
        # Residual connection
        out += x # (B, V, C_i, T_i)
        # Fully connected layer
        out = self.output_layer(out.flatten(-2, -1)) # (B, V, prod(*out_shape))
        if hasattr(self.out_shape, '__len__'):
            return out.reshape((x.shape[0], x.shape[1], *self.out_shape))
        else:
            return out


class StackedSCINet(nn.Module):
    """
    in_channels: Input channels; C_i
    in_timesteps: Input timesteps; T_i
    out_shape: Output shape of the final stack
    hidden_channels: Hidden channels for each layer in each stack
        If list, must satisfy len(hidden_channels) == num_levels
    num_stacks: Number of SCINet stacks
    num_levels: Number of levels in each SCINet stack; L
    """
    def __init__(
            self, in_channels: int, in_timesteps: int, out_shape: Union[int, Tuple], 
            hidden_channels: Union[int, list], kernel_size: Tuple[int, int], 
            num_stacks: int=2, num_levels: int=3, 
            padding_mode: str="replicate", causal_conv: bool=True, dropout_prob: float=0.25):
        super().__init__()
        self.num_stacks = num_stacks
        self.in_channels = in_channels
        self.in_timesteps = in_timesteps
        self.out_shape = out_shape
        self.scinets = []
        for s in range(num_stacks - 1):
            self.scinets.append(
                SCINet(
                num_levels, in_channels, in_timesteps, hidden_channels, (in_channels, in_timesteps), 
                kernel_size, padding_mode, causal_conv, dropout_prob))
        self.scinets.append(
            SCINet(
                num_levels, in_channels, in_timesteps, hidden_channels, out_shape, 
                kernel_size, padding_mode, causal_conv, dropout_prob))
        
    def forward(
            self, x: torch.Tensor) -> torch.Tensor:
        """
        -------Arguments-------
        x: (B, V, C_i, T_i) OR (B, V, C_i*T_i)
        --------Outputs--------
        out: (B, V, out_shape)
        """
        if len(x.shape) == 3:
            x = x.reshape(x.shape[0], x.shape[1], self.in_channels, self.in_timesteps)
        for scinet in self.scinets:
            x = scinet(x)
        return x
        
