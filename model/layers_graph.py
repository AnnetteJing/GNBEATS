import math
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

##################################################
## Undirected Graph Convolution
##################################################
class UndirGraphConv(nn.Module):
    def __init__(self, args):
        super(UndirGraphConv, self).__init__()
        self.num_nodes = args.num_nodes # V
        self.embed_dim = args.embed_dim # D
        self.c_in = args.c_in # C_i
        self.c_out = args.c_out # C_o
        self.cheb_k = args.cheb_k # K. 1st order approx if K = 1
        self.adj_mat_type = args.adj_mat_type
        self.weights_pool = nn.Parameter(
            torch.randn(self.embed_dim, self.cheb_k, self.c_in, self.c_out)) # (D, K, C_i, C_o)
        self.bias_pool = nn.Parameter(torch.randn(self.embed_dim, self.c_out))  # (D, C_o)

    def calc_adj_mat(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        -------Arguments-------
        node_embeddings: (V, D)
        --------Outputs--------
        normalized_adj_mat: (V, V)
        """
        adj_mat = node_embeddings @ node_embeddings.T
        if self.adj_mat_type == "Bai":
            return F.softmax(F.relu(adj_mat), dim=0)
        else:
            adj_mat_rowsum = torch.sum(adj_mat, axis=1)[:, None]
            return adj_mat/torch.sqrt(adj_mat_rowsum @ adj_mat_rowsum.T)

    def forward(self, x: torch.Tensor, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        -------Arguments-------
        x: (B, V, C_i)
        node_embeddings: (V, D)
        --------Outputs--------
        x_gconv: (B, V, C_o)
        """
        adj_mat = self.calc_adj_mat(node_embeddings)
        weights = torch.einsum("vd,dkio->vkio", node_embeddings, self.weights_pool) # (V, K, C_i, C_o)
        bias = torch.matmul(node_embeddings, self.bias_pool)  # (V, C_o)
        device = adj_mat.device
        if self.cheb_k == 1: # 1st order approx
            x_gconv = torch.einsum("uv,bvi,vio->buo", 
                (torch.eye(self.num_nodes) + adj_mat), x, torch.squeeze(weights, 1)) + bias # (B, V, C_o)
        else: # Chebshev polynomials approx 
            ## No scaling & normalizing, since the model can learn those intrinsically
            laplacian = torch.eye(self.num_nodes).to(device) - adj_mat # (V, V)
            cheb_polys = [torch.eye(self.num_nodes).to(device), laplacian]
            for k in range(2, self.cheb_k):
                cheb_polys.append(2*cheb_polys[k - 1] - cheb_polys[k - 2])
            cheb_polys = torch.stack(cheb_polys, dim=0) # (K, V, V)
            cheb_polys = torch.einsum("kuv,bvi->buki", cheb_polys, x) # (B, V, K, C_i)
            x_gconv = torch.einsum("bvki,vkio->bvo", cheb_polys, weights) # (B, V, C_o)
        return x_gconv + bias


##################################################
## Directed Graph Convolution
##################################################
class DirGraphConv(nn.Module):
    def __init__(self, args):
        super(DirGraphConv, self).__init__()
        self.num_nodes = args.num_nodes # V
        self.embed_dim = args.embed_dim # D
        self.c_in = args.c_in # C_i
        self.c_out = args.c_out # C_o
        self.cheb_k = args.cheb_k # K
        weights_shape = self.embed_dim, self.cheb_k, self.c_in, self.c_out
        bias_shape = self.embed_dim, self.c_out
        if args.param_type == "real":
            self.weights_pool = nn.Parameter(
                torch.randn(*weights_shape).type(torch.complex64)) # (D, K, C_i, C_o)
            self.bias_pool = nn.Parameter(
                torch.randn(*bias_shape).type(torch.complex64))  # (D, C_o)
        elif args.param_type == "complex":
            self.weights_pool = nn.Parameter(torch.complex(
                torch.randn(*weights_shape), torch.randn(*weights_shape)))
            self.bias_pool = nn.Parameter(torch.complex(
                torch.randn(*bias_shape), torch.randn(*bias_shape)))
        self.activation = (
            ComplexActivation(args.activation_type) if args.activation_type is not None else None)
        self.norm_laplacian = args.norm_laplacian

    def calc_laplacian(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        -------Arguments-------
        node_embeddings: (V, D)
        --------Outputs--------
        normalized_laplacian: (V, V)
        """
        laplacian = node_embeddings @ node_embeddings.conj().T
        if self.activation is not None:
            laplacian = self.activation(laplacian)
        if self.norm_laplacian == "spec": # Spectral (maximum singular value)
            norm_factor = torch.linalg.matrix_norm(laplacian, ord=2)
        elif self.norm_laplacian == "frob": # Frobenius
            norm_factor = torch.linalg.matrix_norm(laplacian)
        return laplacian/norm_factor
    
    def forward(
            self, x: torch.Tensor, node_embeddings: torch.Tensor, unwind: bool=False
            ) -> torch.Tensor:
        """
        -------Arguments-------
        x: (B, V, C_i)
        node_embeddings: (V, D)
        --------Outputs--------
        x_gconv: (B, V, C_o) complex OR (B, V, 2*C_o) real
        """
        laplacian = self.calc_laplacian(node_embeddings)
        weights = torch.einsum("vd,dkio->vkio", node_embeddings, self.weights_pool) # (V, K, C_i, C_o)
        bias = torch.matmul(node_embeddings, self.bias_pool)  # (V, C_o)
        cheb_polys = [torch.eye(self.num_nodes, dtype=laplacian.dtype).to(laplacian.device), laplacian]
        for k in range(2, self.cheb_k):
            cheb_polys.append(2*cheb_polys[k - 1] - cheb_polys[k - 2])
        cheb_polys = torch.stack(cheb_polys, dim=0) # (K, V, V)
        cheb_polys = torch.einsum("kuv,bvi->buki", cheb_polys, x.type(cheb_polys.dtype)) # (B, V, K, C_i)
        x_gconv = torch.einsum("bvki,vkio->bvo", cheb_polys, weights) + bias # (B, V, C_o)
        if unwind:
            return torch.view_as_real(x_gconv).flatten(2, 3) # (B, V, 2*C_o)
        else:
            return x_gconv


##################################################
## Complex Activation Functions
##################################################
class ComplexActivation(nn.Module):
    """
    method: One of "arg_bdd", "real_imag", and "phase_amp"
        arg_bdd (Argument bound):
            z if -pi/2 <= arg(z) < pi/2, 0 otherwise
        real_imag (Real-imaginary):
            ReLU(z.real) + i*ReLU(z.imag)
        phase_amp (Phase-amplitude):
            tanh(|z|)*exp(i*arg(z))
    """
    def __init__(self, method: str="arg_bdd"):
        super(ComplexActivation, self).__init__()
        self.method = method
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.method == "arg_bdd":
            return torch.where(
                (-torch.pi/2 <= torch.angle(x)) & (torch.angle(x) < torch.pi/2), x, 0)
        elif self.method == "real_imag":
            return F.relu(x.real) + 1.j*F.relu(x.imag)
        elif self.method == "phase_amp":
            return torch.tanh(x.abs())*torch.exp(1.j*x.angle())





