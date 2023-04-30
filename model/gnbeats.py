import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, Union, Iterable, Optional, Dict, List, Type
from dataclasses import asdict
import copy
import time
from .gnbeats_components import AutoregressiveBlock, GraphBlock, DoubleResStack
from ..config.config_classes import *
from ..logging.logging import _get_logger
from ..utils.data_utils import Normalizer
from ..utils.loss_functions import _get_loss_func

# https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
def is_notebook() -> bool:
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class GNBEATS(nn.Module):
    """
    Graphical N-BEATS Model
    ----------------------
    FLAGS - 0 = Trend, 1 = Seasonality, 2 = Identity
    ----------------------
    window: Lookback window, W
    horizon: Forecast horizon, H
    num_nodes: Number of nodes, V
    embed_dim: Embedded dimension, D
        Node embeddings E have shape (V, 2*D), and the adjacency matrix = func(E[:D] @ E[D:].T),
        where func can include activation, normalization, removal of self-loops, etc
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
    """
    basis_types = {0: "trend", 1: "season", 2: "identity"}
    def __init__(
            self, window: int, horizon: int, num_nodes: int, embed_dim: int, 
            node_mod: bool=True, share_params: bool=True, block_dropout_prob: float=0,
            l1_penalty_linear: float=1e-5, l1_penalty_conv: float=1e-5, 
            num_ar_blocks: Optional[Union[int, Tuple[int]]]=None, 
            num_graph_blocks: Optional[Union[int, Tuple[int]]]=None, 
            ar_block_order: Optional[Iterable]=[0, 1, 2, 0, 0, 1, 1, 2, 2], 
            graph_block_order: Optional[Iterable]=[0, 1, 2, 0, 0, 1, 1, 2, 2],
            trend_deg: int=3, season_deg: Optional[int]=None,
            trend_include_id: bool=True, season_include_id: bool=True, 
            ar_theta_net: Union[str, List[str]]="TCN", 
            graph_theta_net: Union[str, List[str]]="TCN", 
            fc_net_kwargs: Optional[ConfigFC]=None,
            tcn_kwargs: Optional[ConfigTCN]=None,
            scinet_kwargs: Optional[ConfigSCINet]=None,
            graph_block_kwargs: Optional[ConfigGraphBlock]=None, 
            node_embeddings: Optional[torch.Tensor]=None,
            device: Optional[str]=None,
            optimizer: Optional[torch.optim.Optimizer]=None, 
            loss: str="mse", 
            main_seasonality: Optional[int]=None, 
            debug: bool=False, 
            log_filename: Optional[str]="output.log",
            model_path: Optional[str]="models"
            ): 
        super().__init__()
        self.window = window # W
        self.horizon = horizon # H
        self.num_nodes = num_nodes # V
        self.embed_dim = embed_dim # D
        self.node_mod = node_mod
        self.share_params = share_params
        if num_ar_blocks:
            num_ar_blocks = [num_ar_blocks]*3 if isinstance(num_ar_blocks, int) else num_ar_blocks
            ar_block_order = (j for i, num_block in enumerate(num_ar_blocks) for j in [i]*num_block)
        if num_graph_blocks:
            num_graph_blocks = [num_graph_blocks]*3 if isinstance(num_graph_blocks, int) else num_graph_blocks
            graph_block_order = (j for i, num_block in enumerate(num_graph_blocks) for j in [i]*num_block)
        ar_theta_net = [
            ar_theta_net for i in range(3)] if isinstance(ar_theta_net, str) else ar_theta_net
        graph_theta_net = [
            graph_theta_net for i in range(3)] if isinstance(graph_theta_net, str) else graph_theta_net
        self.theta_net_kwargs = {
            "FC": fc_net_kwargs if fc_net_kwargs else ConfigFC(hidden_dims=[window, window, horizon]), 
            "TCN": tcn_kwargs if tcn_kwargs else ConfigTCN(
                kernel_size=4, hidden_channels=[2*embed_dim, embed_dim, 1]), 
            "SCINet": scinet_kwargs if scinet_kwargs else ConfigSCINet(
                hidden_channels=[2*embed_dim, embed_dim, 1], kernel_size=(3, 5))}
        self.graph_block_kwargs = graph_block_kwargs if graph_block_kwargs else ConfigGraphBlock()
        deg, include_id = [trend_deg, season_deg, None], [trend_include_id, season_include_id, None]
        blocks = self._get_blocks(
            ar_block_order, ar_theta_net, deg, include_id, graph_blocks=False)
        blocks += self._get_blocks(
            graph_block_order, graph_theta_net, deg, include_id, graph_blocks=True)
        blocks = nn.ModuleList(blocks)
        # 0 = AR Trend, 1 = AR Seasonality, 2 = AR Identity
        # 3 = Graph Trend, 4 = Graph Seasonality, 5 = Graph Identity
        block_grouping = torch.cat((torch.tensor(ar_block_order), (3 + torch.tensor(graph_block_order))))
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        node_embeddings = node_embeddings if node_embeddings else torch.randn(num_nodes, 2*embed_dim) # (V, 2*D)
        self.model = DoubleResStack(
            blocks, node_embeddings, block_grouping, block_dropout_prob).to(self.device)
        self.optimizer = optimizer if optimizer else torch.optim.Adam(params=self.model.parameters())
        self.loss = loss
        self.regularization = {
            "block_dropout_prob": block_dropout_prob, 
            "l1_penalty_linear": l1_penalty_linear, 
            "l1_penalty_conv": l1_penalty_conv}
        self.loss_func = self._loss_func(loss, main_seasonality)
        self.logger = _get_logger(debug=debug, filename=log_filename)
        self.model_path = model_path

    def _get_block_reg(self):
        l1_reg = 0
        for block in self.model.blocks:
            for layer in block.modules():
                if self.regularization["l1_penalty_linear"] > 0 and isinstance(layer, nn.Linear):
                    l1_reg += self.regularization["l1_penalty_linear"] * torch.sum(torch.abs(layer.weight))
                elif self.regularization["l1_penalty_conv"] > 0 and isinstance(layer, nn.Conv2d):
                    l1_reg += self.regularization["l1_penalty_conv"] * torch.sum(torch.abs(layer.weight))
        return l1_reg
    
    def _loss_func(self, loss, main_seasonality):
        base_loss = _get_loss_func(loss, self.device, main_seasonality)
        def reg_loss(targets, preds):
            return base_loss(targets, preds) + self._get_block_reg()
        return reg_loss
        
    def _get_blocks(
            self, block_order: Iterable, theta_net: List[str], 
            deg: List[int], include_id: List[bool], graph_blocks: bool
            ) -> List:
        blocks = []
        if self.share_params:
            init_block_idx = [None for i in range(3)]
        for i, block_type in enumerate(block_order):
            for num in range(3):
                if block_type == num:
                    if self.share_params and init_block_idx[block_type] is not None:
                        blocks.append(blocks[init_block_idx[block_type]])
                    else:
                        if self.share_params:
                            init_block_idx[block_type] = i
                        if graph_blocks:
                            block = GraphBlock(
                                self.window, self.horizon, self.embed_dim, 
                                basis=self.basis_types[block_type],
                                theta_net=theta_net[block_type], node_mod=self.node_mod, 
                                deg=deg[block_type], include_identity=include_id[block_type],
                                **asdict(self.graph_block_kwargs), 
                                **asdict(self.theta_net_kwargs[theta_net[block_type]]))
                        else:
                            block = AutoregressiveBlock(
                                self.window, self.horizon, self.embed_dim, 
                                basis=self.basis_types[block_type],
                                theta_net=theta_net[block_type], node_mod=self.node_mod, 
                                deg=deg[block_type], include_identity=include_id[block_type],
                                **asdict(self.theta_net_kwargs[theta_net[block_type]]))
                        blocks.append(block)
        return blocks
    
    def _train_epoch(
            self, train_loader: DataLoader, train_normalizer: Normalizer, 
            summary_writer: Optional[SummaryWriter]=None) -> List[float]:
        self.model.train()
        epoch_losses = []
        for inputs, targets in train_loader:
            self.optimizer.zero_grad()
            preds = train_normalizer.unnormalize(self.model(inputs)).to(self.device)
            loss = self.loss_func(preds, targets)
            loss.backward()
            self.optimizer.step()
            epoch_losses.append(loss.item())
            if summary_writer:
                summary_writer.add_scalar(f"Loss ({self.loss}) / Train", loss.item())
        return epoch_losses
    
    def _valid_epoch(
            self, valid_loader: DataLoader, valid_normalizer: Normalizer, 
            summary_writer: Optional[SummaryWriter]=None) -> List[float]:
        self.model.eval()
        epoch_losses = []
        with torch.no_grad():
            for inputs, targets in valid_loader:
                preds = valid_normalizer.unnormalize(self.model(inputs)).to(self.device)
                loss = self.loss_func(preds, targets)
                epoch_losses.append(loss.item())
                if summary_writer:
                    summary_writer.add_scalar(f"Loss ({self.loss}) / Validate", loss.item())
        return epoch_losses

    def fit(
            self, loader: Dict[str, DataLoader], normalizer: Dict[str, Normalizer], 
            num_epochs: int=100, early_stop_patience: int=15, save_model: bool=True,
            use_tensor_board: bool=True):
        """
        -------Arguments-------
        loader: Dictionary of dataloaders. Key "train" required, key "valid" optional
        normalizer: Dictionary of normalizers. Key "train" required, key "valid" optional
        num_epochs: Number of epochs / passes through the data
        early_stop_patience: #epochs of valid loss not decreasing allowed before training is stopped
        save_model: Save self.model.state_dict() to disk (model with lowest valid loss) if True
        use_tensor_board: Use tensorboard if True
        """
        summary_writer = SummaryWriter() if use_tensor_board else None
        self.losses = {"train": []}
        valid_loader, valid_normalizer = loader.get("valid"), normalizer.get("valid") # None if no "valid" key
        validate = (valid_loader is not None) and (valid_normalizer is not None)
        best_loss, not_improved_epochs = float('inf'), 0
        start_time = time.time()
        for epoch in tqdm(range(num_epochs), desc="Epoch", position=0):
            # Train
            train_epoch_losses = self._train_epoch(loader["train"], normalizer["train"], summary_writer)
            train_epoch_avg_loss = np.mean(train_epoch_losses)
            if train_epoch_avg_loss > 1e9:
                self.logger.warning("Gradient explosion detected. Ending...")
                break
            self.losses["train"].extend(train_epoch_losses)
            # Validate
            if validate:
                valid_epoch_losses = self._valid_epoch(valid_loader, valid_normalizer, summary_writer)
                valid_epoch_avg_loss = np.mean(valid_epoch_losses)
                self.losses.setdefault("valid", []).extend(valid_epoch_losses)
                if valid_epoch_avg_loss < best_loss:
                    best_loss, not_improved_epochs = valid_epoch_avg_loss, 0
                    best_model = copy.deepcopy(self.model.state_dict())
                else:
                    not_improved_epochs += 1
                if not_improved_epochs == early_stop_patience:
                    self.logger.info(
                        f"Validation performance did not improve for {early_stop_patience} epochs. Ending...")
                    break
        train_time = time.time() - start_time
        hrs = train_time // 3600
        mins = (train_time - hrs*3600) // 60
        secs = train_time - hrs*3600 - mins*60
        self.logger.info(f"Total training time: {hrs}h{mins}m{secs}s, Best loss: {best_loss :.4f}")
        self.model.load_state_dict(best_model) # Use the best model as the final model
        if save_model:
            current_time = time.strftime("%Y-%m-%d_%H:%M", time.localtime(time.time()))
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
            model_path = os.path.join(self.model_path, f"{current_time}_exp.pth")
            self.logger.info("Saving current best model to " + model_path)
            torch.save(best_model, model_path)
    
    def test(
            self, test_loader: Union[DataLoader, Dict[str, DataLoader]], 
            test_normalizer: Union[Normalizer, Dict[str, Normalizer]], 
            metrics: Union[List[str], str]=["rmse", "mae", "mape"], 
            seasonality: Optional[int]=None, verbose: bool=False) -> Dict[str, float]:
        """
        -------Arguments-------
        test_loader, test_normalizer: Dataloader & normalizer for test data
        metrics: List of evaluation metrics to use
            A subset of "mse", "rmse", "mae", "mape", "wmape", "smape", "mase"
        seasonality: Optional argument for "mase"
        verbose: Print out contents of output if True
        --------Outputs--------
        output: Dictionary of the average loss & metrics
        """
        test_loader = test_loader["test"] if isinstance(test_loader, dict) else test_loader
        test_normalizer = test_normalizer["test"] if isinstance(test_normalizer, dict) else test_normalizer
        if not hasattr(metrics, '__len__'):
            metrics = [metrics]
        metrics = {metric: _get_loss_func(metric, self.device, seasonality) for metric in metrics}
        output = dict()
        self.model.eval()
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc="Batch", position=0):
                preds = test_normalizer.unnormalize(self.model(inputs)).to(self.device)
                output.setdefault("loss", []).append(self.loss_func(preds, targets).item())
                for metric, metric_func in metrics.items():
                    output.setdefault(metric, []).append(metric_func(preds, targets).item())
        output = {key: np.average(vals) for key, vals in output.items()}
        output_msg = "Loss = {avg_loss:.2f}".format(avg_loss=output["loss"])
        for metric in metrics.keys():
            output_msg += f" | {str.upper(metric)} = {output[metric]:.2f}"
        self.logger.info(output_msg)
        if verbose:
            print(output_msg)
        return output
    
    def predict(
            self, data: Union[np.array, torch.Tensor], decompose: bool=True, verbose: bool=True, 
            normalizer: Optional[Normalizer]=None, norm_method: Optional[str]="z_score"
            ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        -------Arguments-------
        data: (B, V, T) or (V, T), T >= W
        decompose: Return preds_decomp if True
        --------Outputs--------
        preds: (B, V, H) or (V, H)
        preds_decomp: (B, V, H, 6) or (V, H, 6)
            preds_decomp[v, :, g] is the gth component for series/node v, where g is
            0 = AR Trend, 1 = AR Seasonality, 2 = AR Identity
            3 = Graph Trend, 4 = Graph Seasonality, 5 = Graph Identity
        """
        if len(data.shape) == 2: # (V, T)
            data = data.reshape(1, *data.shape) # (B=1, V, T)
        _, num_nodes, time_steps = data.shape
        assert num_nodes == self.num_nodes, f"Input nodes {num_nodes} does not equal {self.num_nodes}"
        assert time_steps >= self.window, f"Input time steps {time_steps} < {self.window}"
        if time_steps > self.window:
            time_step_msg = f"""\
            Input time steps {time_steps} > {self.window}, using last {self.window} observations as input.\
            """
            self.logger.info(time_step_msg)
            if verbose:
                print(time_step_msg)
            data = data[:, :, -self.window:] # (B, V, W)
        if type(data).__module__ == np.__name__:
            data = torch.tensor(data)
        if not normalizer:
            normalizer = Normalizer(norm_method)
        if decompose:
            # B > 1: (B, V, H), (B, V, H, 6); B = 1: (V, H), (V, H, 6)
            preds, preds_decomp = self.model(normalizer.normalize(data), decompose)
            preds = normalizer.unnormalize(preds)
            preds_decomp = normalizer.unnormalize(preds_decomp.transpose(-1, -2)).transpose(-1, -2)
            return preds, preds_decomp
        else: 
            # B > 1: (B, V, H); B = 1: (V, H)
            preds = normalizer.unnormalize(self.model(normalizer.normalize(data), decompose))
            return preds
    
    





