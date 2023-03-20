import torch
import torch.nn as nn
import torch.nn.functional as F


##################################################
## Complex Activation Functions
##################################################
class ComplexActivation(nn.Module):
    """
    method: One of "real", "real_imag", "arg_bdd", and "phase_amp"
        real (Real):
            1[z.real >= 0]*z
            = ReLU(z.real) + i*1[z.real >= 0]*z.imag
        real_imag (Real-imaginary):
            ReLU(z.real) + i*ReLU(z.imag)
        arg_bdd (Argument bound):
            z if -pi/2 <= arg(z) < pi/2, 0 otherwise
        phase_amp (Phase-amplitude):
            tanh(|z|)*exp(i*arg(z))
    """
    def __init__(self, method: str="arg_bdd"):
        super().__init__()
        self.method = method
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.method == "real":
            return x*(x.real >= 0)
        elif self.method == "real_imag":
            return F.relu(x.real) + 1.j*F.relu(x.imag)
        elif self.method == "arg_bdd":
            return torch.where(
                (-torch.pi/2 <= torch.angle(x)) & (torch.angle(x) < torch.pi/2), x, 0)
        elif self.method == "phase_amp":
            return torch.tanh(x.abs())*torch.exp(1.j*x.angle())
        