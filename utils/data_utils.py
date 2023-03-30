import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Union


def load_data_npz(ds_folder: str, ds_name: str):
    data_file = os.path.join(
        "/Users/sleeper/Desktop/DL_Forecasting/GNBEATS/datasets", 
        ds_folder, ds_name + ".npz")
    data = np.load(data_file, allow_pickle=True)["data"].squeeze()
    return data


class Normalizer:
    def __init__(self, method: str):
        self.method = method
        self.norm_statistic = None

    def normalize(self, data: np.array):
        if self.method == "z_score":
            mean = data.mean(axis=0)
            std = data.std(axis=0)
            std[std == 0] = 1
            self.norm_statistic = mean, std
            return (data - mean)/std
        elif self.method == "min_max":
            min = data.min(axis=0)
            max = data.max(axis=0)
            scale = max - min
            scale[scale == 0] = 1
            self.norm_statistic = min, scale
            return (data - min)/scale
        
    def unnormalize(self, data: np.array):
        if self.method == "z_score":
            mean, std = self.norm_statistic
            return data * std + mean
        elif self.method == "min_max":
            min, scale = self.norm_statistic
            return data * scale + min


def split_data(
        data: np.array, train: Union[int, float], valid: Union[int, float], by_ratio: bool=True):
    n = len(data)
    if by_ratio:
        lo, hi = int(n * train), int(n * (train + valid))
    else:
        lo, hi = int(train), int(train + valid)
    assert lo > 0 and hi < n and lo != hi
    return data[:lo], data[lo:hi], data[hi:]
    

class TS_Forecast_Dataset(Dataset):
    def __init__(
            self, df: np.array, window: int, horizon: int, interval: int=1, 
            device: Optional[str]=None, normalizer: Optional[str]=None):
        self.window = window # Size of lookback window (input = data[t-window:t])
        self.horizon = horizon # Size of forecast horizon (target = data[t:t+horizon])
        if normalizer is not None:
            self.df = normalizer.normalize(df)
        else:
            self.df = df
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        range_t = range(self.window, len(self.df) - self.horizon + 1)
        self.idx_mid = [range_t[i * interval] for i in range(len(range_t) // interval)]

    def __getitem__(self, index: int):
        mid = self.idx_mid[index]
        lo, hi = mid - self.window, mid + self.horizon
        inputs, targets = self.df[lo:mid], self.df[mid:hi]
        inputs = torch.from_numpy(inputs).type(torch.float).to(self.device)
        targets = torch.from_numpy(targets).type(torch.float).to(self.device)
        return inputs, targets
    
    def __len__(self):
        return len(self.idx_mid)


def get_loader_normalizer(
        data: np.array, batch_size: int, window: int, horizon: int, 
        train_interval: int, test_interval: int,
        train_shuffle: bool=True, normalization: Optional[str]="z_score",
        train: Union[int, float]=0.7, valid: Union[int, float]=0.2, 
        by_ratio: bool=True, device: Optional[str]=None):
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
    return loader, normalizer





