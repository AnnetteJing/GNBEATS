import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from .layers_temporal import ParallelConv1d, ParallelCausalConv1d


class TemporalBlock(nn.Module):
    def __init__(
            self, in_channels: int, out_channels: int,  kernel_size: int, dilation: int,
            padding_mode: str="zeros", dropout_prob: int=0.2):
        super(TemporalBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding_mode = padding_mode
        self.dropout_prob = dropout_prob
        conv1 = weight_norm(
            ParallelCausalConv1d(
            in_channels, out_channels, kernel_size, 
            dilation=dilation, padding_mode=padding_mode
            ))
        conv2 = weight_norm(
            ParallelCausalConv1d(
            out_channels, out_channels, kernel_size, 
            dilation=dilation, padding_mode=padding_mode
            ))
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
            self, channels: list, kernel_size: int, 
            padding_mode: str="zeros", dropout_prob: int=0.2):
        super(TCN, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding_mode = padding_mode
        self.dropout_prob = dropout_prob
        layers = []
        for i in range(len(channels)):
            layers.append(TemporalBlock(
                channels[i], channels[i + 1], kernel_size, 
                dilation=2**i, padding_mode=padding_mode, dropout_prob=dropout_prob
            ))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        -------Arguments-------
        x: (B, V, channels[0], T)
        --------Outputs--------
        out: (B, V, channels[-1], T)
        """
        return self.layers(x)
    
    