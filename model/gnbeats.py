import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Union, Iterable, Optional
from dataclasses import asdict
from .gnbeats_components import AutoregressiveBlock, GraphBlock, DoubleResStack
from config.config_classes import *


class GNBEATS(nn.Module):
    """
    Graphical N-BEATS Model
    ----------------------
    FLAGS
    Block type: 0 = Trend, 1 = Seasonality, 2 = Identity
    Network type: 0 = FC, 1 = TCN, 2 = SCINet
    ----------------------
    window: Lookback window, W
    horizon: Forecast horizon, H
    embed_dim: Embedded dimension, D
        Node embeddings E have shape (V, 2*D), and adjacency matrices are approximately E[:D] @ E[D:].T
    node_mod: Modify pooled parameters by node embeddings if True
    share_params: Repeat the same block (same parameters) if True
    num_ar_blocks / num_graph_blocks: Integer tuple of length 3
        If receives a single int, then initialize the same number of blocks for all three types.
        e.g. [2, 2, 1] = [Trend, Trend, Seasonality, Seasonality, Identity].
        e.g. 2 = [2, 2, 2] = [Trend, Trend, Seasonality, Seasonality, Identity, Identity].
        Specify only when ar_block_order / graph_block_order are not specified.
    ar_block_order / graph_block_order: Iterable taking values in {0, 1, 2}
        e.g. [0, 0, 1, 2, 1] = [Trend, Trend, Seasonality, Identity, Seasonality].
        Graph blocks always come after AR blocks.
        Used only when num_ar_blocks / num_graph_blocks are None.
    ar_theta_net / graph_theta_net: String tuple of length 3 taking values in {"FC", "TCN", "SCINet"}
        If receives a single string, then use the same network for all three kinds of blocks.
        e.g. ["FC", "TCN", "FC"] indicates using FC as theta_net for Trend & Identity blocks, 
        and using TCN for Seasonality blocks. 
        e.g. "SCINet" = ["SCINet", "SCINet", "SCINet"].
    config_graph_block: 
    """
    def __init__(
            self, window: int, horizon: int, embed_dim: int, 
            node_mod: bool=True, share_params: bool=True,
            num_ar_blocks: Optional[Union[int, Tuple[int]]]=None, 
            num_graph_blocks: Optional[Union[int, Tuple[int]]]=None, 
            ar_block_order: Optional[Iterable]=[0, 1, 2, 0, 0, 1, 1, 2, 2], 
            graph_block_order: Optional[Iterable]=[0, 1, 2, 0, 0, 1, 1, 2, 2],
            ar_theta_net: Union[str, Tuple[str]]="SCINet", 
            graph_theta_net: Union[str, Tuple[str]]="SCINet", 
            fc_net_kwargs: Optional[ConfigFC]=None,
            tcn_kwargs: Optional[ConfigTCN]=None,
            scinet_kwargs: Optional[ConfigSCINet]=None,
            graph_block_kwargs: Optional[ConfigGraphBlock]=None
            ): 
        super().__init__()
        self.window = window # W
        self.horizon = horizon # H
        self.embed_dim = embed_dim # D
        self.node_mod = node_mod
        self.share_params = share_params
        if num_ar_blocks:
            num_ar_blocks = [num_ar_blocks]*3 if isinstance(num_ar_blocks, int) else num_ar_blocks
            ar_block_order = (j for i, num_block in enumerate(num_ar_blocks) for j in [i]*num_block)
        if num_graph_blocks:
            num_graph_blocks = [num_graph_blocks]*3 if isinstance(num_graph_blocks, int) else num_graph_blocks
            graph_block_order = (j for i, num_block in enumerate(num_graph_blocks) for j in [i]*num_block)
        assert ar_block_order, "Neither num_ar_blocks nor ar_block_order are specified"
        assert graph_block_order, "Neither num_graph_blocks nor graph_block_order are specified"
        ar_theta_net = (
            ar_theta_net for i in range(3)) if isinstance(ar_theta_net, str) else ar_theta_net
        graph_theta_net = (
            graph_theta_net for i in range(3)) if isinstance(graph_theta_net, str) else graph_theta_net
        self.theta_net_kwargs = (fc_net_kwargs, tcn_kwargs, scinet_kwargs)


        ar_blocks = []
        if self.share_params:
            init_block_idx = (None for i in range(3))
        for i, block_type in enumerate(ar_block_order):
            if block_type == 0:
                if self.share_params and init_block_idx[block_type] is not None:
                    ar_blocks.append(ar_blocks[init_block_idx[block_type]])
                else:
                    if self.share_params:
                        init_block_idx[block_type] = i
                    ar_blocks.append(AutoregressiveBlock(
                        self.window, self.horizon, self.embed_dim, basis="trend", 
                        theta_net=ar_theta_net[block_type], node_mod=self.node_mod))



