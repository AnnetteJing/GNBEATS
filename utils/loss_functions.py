import torch
import torch.nn as nn
from typing import Optional

def _get_loss_func(loss: str, device: str, seasonality: Optional[int]=None):
    if loss == "mse":
        return nn.MSELoss().to(device)
    elif loss == "mae":
        return nn.L1Loss().to(device)
    elif loss == "mape":
        def MAPE(yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.mean(torch.abs(yhat - y)/torch.abs(torch.where(y < 1e-5, 1e-5, y)))
        return MAPE.to(device)
    elif loss == "wmape":
        def WMAPE(yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            diff_sum = torch.sum(torch.abs(yhat - y))
            y_sum = torch.sum(torch.abs(y))
            return diff_sum/torch.where(y_sum < 1e-5, 1e-5, y_sum)
        return WMAPE.to(device)
    elif loss == "smape":
        def SMAPE(yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            abs_avg = (torch.abs(y) + torch.abs(yhat))/2
            return torch.mean(torch.abs(yhat - y)/torch.abs(torch.where(abs_avg < 1e-5, 1e-5, abs_avg)))
        return SMAPE.to(device)
    elif loss == "mase": # https://en.wikipedia.org/wiki/Mean_absolute_scaled_error
        def MASE(
                yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            mae = torch.mean(torch.abs(yhat - y))
            if seasonality:
                denominator = torch.mean(torch.abs(y - y.roll(seasonality))[seasonality:])
            else:
                denominator = torch.mean(torch.abs(y - torch.mean(y)))
            return mae/denominator
        return MASE.to(device)




