import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm
import torch.nn.init as init
from typing import Optional, Tuple, Union


##################################################
## Node-dependent Modification
##################################################
class NodeDependentMod(nn.Module):
    """
    Modifies each input series by its corresponding node embedding,
    which facilitates parameter sharing in other layers
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        -------Arguments-------
        x: (B, V, T_i)
        node_embeddings: (V, D)
        --------Outputs--------
        out: (B, V, D, T_i)
        """
        return torch.einsum("bvt,vd->bvdt", x, node_embeddings)


##################################################
## Parallel Conv1d
##################################################
class ParallelConv1d(nn.Module):
    """
    Applies the same 1D convolution to multiple sets of the same size
    """
    def __init__(
            self, in_channels: int, out_channels: int, kernel_size: int, 
            stride: int=1, dilation: int=1, 
            padding: Tuple[int, int]=(0, 0), padding_mode: str="zeros", 
            param_norm: Optional[str]=None):
        super().__init__()
        self.in_channels = in_channels # C_i
        self.out_channels = out_channels # C_o
        self.kernel_size = kernel_size # Ker
        self.stride = stride # Str
        self.dilation = dilation # Dil
        padding = (*padding, 0, 0) # (Pad_l, Pad_r, 0, 0)
        if padding_mode == "zeros":
            self.pad = nn.ZeroPad2d(padding)
        elif padding_mode == "replicate":
            self.pad = nn.ReplicationPad2d(padding)
        elif padding_mode == "reflection":
            self.pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(
            in_channels=1, out_channels=out_channels, kernel_size=(in_channels, kernel_size), 
            stride=(in_channels, stride), dilation=(1, dilation))
        if param_norm == "weight":
            self.conv = weight_norm(self.conv)
        elif param_norm == "spectral":
            self.conv = spectral_norm(self.conv)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        -------Arguments-------
        x: (B, V, C_i, T_i)
        --------Outputs--------
        out: (B, V, C_o, T_o)
            T_o = int((T_i + Pad_l + Pad_r - Dil*(Ker - 1) - 1)/Str + 1)
        """
        batch_size, sets, in_channels, in_timesteps = x.shape
        x = x.reshape(batch_size, 1, sets*in_channels, in_timesteps)
        out = self.conv(self.pad(x)).permute(0, 2, 1, 3)
        assert out.shape[1] == sets, f"{out.shape}"
        return out


##################################################
## Parallel Causal Conv1d
##################################################
class ParallelCausalConv1d(ParallelConv1d):
    """
    Applies the same 1D causal convolution to multiple sets of the same size
    """
    def __init__(
            self, in_channels: int, out_channels: int, kernel_size: int, 
            stride: int=1, dilation: int=1, padding_mode: str="replicate",
            param_norm: Optional[str]=None):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, dilation, 
            padding=(dilation*(kernel_size - 1), 0), padding_mode=padding_mode, 
            param_norm=param_norm)
        
    # def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        -------Arguments-------
        x: (B, V, C_i, T_i)
        --------Outputs--------
        out: (B, V, C_o, T_o)
            T_o = int((T_i - 1)/Str + 1)    since Pad_l = Dil*(Ker - 1), Pad_r = 0
            T_i = T_o if Str=1 (default)
        """


##################################################
## Fully Connected Network
##################################################
class FullyConnectedNet(nn.Module):
    """
    Network of linear layers + activations and/or normalizations
    ----------------------
    dims: Input, hidden, and output dimensions
    """
    def __init__(
            self, in_shape: Union[Tuple[int], int], out_shape: Union[Tuple[int], int], 
            hidden_dims: Union[Tuple, list, None], activation: str="selu", 
            batch_norm: bool=False, dropout_prob: float=0):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.hidden_dims = hidden_dims
        self.activation = {
            "relu": nn.ReLU(), 
            "softplus": nn.Softplus(),
            "tanh": nn.Tanh(), 
            "selu": nn.SELU(), 
            "lrelu": nn.LeakyReLU(), 
            "prelu": nn.PReLU(), 
            "sigmoid": nn.Sigmoid()
        }[activation]
        self.batch_norm = batch_norm
        self.dropout_prob = dropout_prob
        if hasattr(hidden_dims, '__len__') and len(hidden_dims) > 0:
            layers = [nn.Linear(np.prod(in_shape), hidden_dims[0])]
            num_hidden_layers = len(hidden_dims) - 1
            for i in range(num_hidden_layers):
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                if batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
                if dropout_prob > 0:
                    layers.append(nn.Dropout(dropout_prob))
            layers.append(nn.Linear(hidden_dims[num_hidden_layers], np.prod(out_shape)))
        else:
            layers = [nn.Linear(np.prod(in_shape), np.prod(out_shape))]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        -------Arguments-------
        x: (B, V, in_shape)
        --------Outputs--------
        out: (B, V, out_shape)
        """
        out = self.layers(x.reshape(x.shape[0], x.shape[1], -1)) # (B, V, prod(out_shape))
        if hasattr(self.out_shape, '__len__'):
            out = out.reshape(x.shape[0], x.shape[1], *self.out_shape)
        return out




