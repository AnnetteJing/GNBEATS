import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Union, Dict
from pathlib import Path


def load_data_npz(ds_folder: str, ds_name: str):
    data_file = os.path.join(
        Path(os.path.abspath(os.path.dirname(__file__))), 
        "..", "datasets", ds_folder, ds_name + ".npz")
    data = np.load(data_file, allow_pickle=True)["data"].squeeze()
    return data


class Normalizer:
    def __init__(self, method: str):
        self.method = method
        self.norm_statistic = None

    # Should only be called once 
    def normalize(self, data: torch.Tensor):
        """
        -------Arguments-------
        data: (*shape, T)
        --------Outputs--------
        out: (*shape, T)
        """
        if self.method == "identity":
            return data
        elif self.method == "z_score":
            mean = data.mean(axis=-1).unsqueeze(-1)
            std = data.std(axis=-1).unsqueeze(-1)
            std[std == 0] = 1
            self.norm_statistic = mean, std
            return (data - mean)/std
        elif self.method == "min_max":
            min = data.min(axis=-1).values.unsqueeze(-1)
            max = data.max(axis=-1).values.unsqueeze(-1)
            scale = max - min
            scale[scale == 0] = 1
            self.norm_statistic = min, scale
            return (data - min)/scale
        
    def unnormalize(self, data: torch.Tensor):
        """
        -------Arguments-------
        data: (*shape, T)
        --------Outputs--------
        out: (*shape, T)
        """
        if self.method == "identity":
            return data
        elif self.method == "z_score":
            mean, std = self.norm_statistic
            return data * std + mean
        elif self.method == "min_max":
            min, scale = self.norm_statistic
            return data * scale + min
    

class TS_Forecast_Dataset(Dataset):
    """
    df: (V, T)
    window: Size of lookback window, W (input = data[t-W:t])
    horizon: Size of forecast horizon, H (target = data[t:t+H])
    interval: 
    node_dim: df.shape=(V, T) if node_dim=0; (T, V) if node_dim=1
    """
    def __init__(
            self, df: Union[np.array, torch.Tensor], window: int, horizon: int, 
            interval: int=1, device: Optional[str]=None, normalizer: Optional[str]=None):
        self.window = window # W
        self.horizon = horizon # H
        if not isinstance(df, torch.Tensor):
            df = torch.tensor(df)
        device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.df = df.type(torch.float).to(device)
        self.df_norm = normalizer.normalize(self.df) if normalizer else self.df
        range_t = range(self.window, self.df.shape[1] - self.horizon + 1)
        self.idx_mid = [range_t[i * interval] for i in range(len(range_t) // interval)]

    def __getitem__(self, index: int):
        """
        -------Arguments-------
        self.df: (V, T)
        --------Outputs--------
        inputs: (V, W)
        targets: (V, H)
        """
        mid = self.idx_mid[index]
        lo, hi = mid - self.window, mid + self.horizon
        inputs = self.df_norm[:, lo:mid] # (V, W), (V, H)
        targets = self.df[:, mid:hi] # (V, H)
        return inputs, targets
    
    def __len__(self):
        return len(self.idx_mid)


def split_data(
        data: Union[np.array, torch.Tensor], 
        train: Union[int, float], valid: Union[int, float], by_ratio: bool=True):
    """
    -------Arguments-------
    data: (V, T)
    --------Outputs--------
    df_train: (V, T_train)
    df_valid: (V, T_valid)
    df_test: (V, T_test)
    """
    n = data.shape[1]
    if by_ratio:
        lo, hi = int(n * round(train, 4)), int(n * round(train + valid, 4))
    else:
        lo, hi = int(train), int(train + valid)
    assert lo > 0 and hi < n and lo != hi
    return data[:, :lo], data[:, lo:hi], data[:, hi:]


def get_loader_normalizer(
        data: Union[np.array, torch.Tensor], batch_size: int, window: int, horizon: int, 
        train_interval: int=1, test_interval: int=1, node_dim: int=0,
        train_shuffle: bool=True, normalization: str="z_score",
        train: Union[int, float]=0.6, valid: Union[int, float]=0.2, 
        by_ratio: bool=True, device: Optional[str]=None, return_df: bool=False):
    data = data if node_dim == 0 else data.T # (V, T)
    df_train, df_valid, df_test = split_data(data, train, valid, by_ratio)
    normalizer = {
        "train": Normalizer(normalization), 
        "valid": Normalizer(normalization), 
        "test": Normalizer(normalization)}
    df_train = TS_Forecast_Dataset(df_train, window, horizon, train_interval, device, normalizer["train"])
    df_valid = TS_Forecast_Dataset(df_valid, window, horizon, train_interval, device, normalizer["valid"])
    df_test = TS_Forecast_Dataset(df_test, window, horizon, test_interval, device, normalizer["test"])
    loader = {
        "train": DataLoader(df_train, batch_size=batch_size, shuffle=train_shuffle), 
        "valid": DataLoader(df_valid, batch_size=batch_size, shuffle=False), 
        "test": DataLoader(df_test, batch_size=batch_size, shuffle=False)}
    if return_df:
        return loader, normalizer, {"train": df_train, "valid": df_valid, "test": df_test}
    else:
        return loader, normalizer


def get_data_from_loader(loader: DataLoader, batch_idx: Optional[int]=None):
    """
    -------Arguments-------
    loader: Dataloader
    batch_idx: Optional integer of selecting 
    --------Outputs--------
    inputs: (B, V, W) if batch_idx=None, else (1, V, W)
    targets: (B, V, H) if batch_idx=None, else (1, V, H)
    """
    inputs, targets = iter(loader).next()
    if batch_idx is not None:
        inputs, targets = inputs[batch_idx].unsqueeze(0), targets[batch_idx].unsqueeze(0)
    return inputs, targets


