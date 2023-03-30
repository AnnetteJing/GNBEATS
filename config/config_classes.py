from typing import Optional, Tuple, Union
from dataclasses import dataclass

@dataclass
class ConfigFC:
    in_shape: Union[Tuple[int], int]
    out_shape: Union[Tuple[int], int]
    hidden_dims: Union[Tuple, list, None]
    activation: str = "selu"
    batch_norm: bool = False
    dropout_prob: float = 0

@dataclass
class ConfigTCN:
    in_channels: int
    in_timesteps: int
    kernel_size: int
    hidden_channels: Optional[list]=None
    padding_mode: str = "zeros"
    dropout_prob: int = 0.2
    out_shape: Union[None, int, Tuple] = None

@dataclass
class ConfigSCINet:
    in_channels: int
    in_timesteps: int
    out_shape: Union[int, Tuple]
    hidden_channels: Union[int, list]
    kernel_size: Tuple[int, int]
    num_stacks: int = 2
    num_levels: int = 3
    padding_mode: str = "replicate"
    causal_conv: bool = True
    dropout_prob: float = 0.25

@dataclass
class ConfigGraphBlock:
    adj_mat_activation: Optional[str] = "tanh"
    update_activation: Optional[str] = None
    self_loops: bool = False
    thresh: float = 0.2
    normalization: Optional[str] = "frob"

