import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, dense_to_sparse
from .. utils.util_functions import ComplexActivation
from .layers_temporal import FullyConnectedNet
from .tcn import TCN
from .scinet import StackedSCINet


##################################################
## Undirected Graph Convolution
##################################################
class UndirGraphConv(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_nodes = args.num_nodes # V
        self.embed_dim = args.embed_dim # D
        self.c_in = args.c_in # C_i
        self.c_out = args.c_out # C_o
        self.cheb_k = args.cheb_k # K. 1st order approx if K = 1
        self.normalization = args.normalization
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
        if self.normalization == "Bai":
            return F.softmax(F.relu(adj_mat), dim=0)
        else:
            adj_mat_rowsum = torch.sum(adj_mat, axis=1)[:, None]
            return adj_mat/torch.sqrt(torch.abs(adj_mat_rowsum @ adj_mat_rowsum.T))

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
## Directed Graph Convolution: Magnetic
##################################################
class MagGraphConv(nn.Module):
    def __init__(self, args):
        super().__init__()
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
            ComplexActivation(args.activation) if args.activation is not None else None)
        self.normalization = args.normalization

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
        if self.normalization == "spec": # Spectral (maximum singular value)
            norm_factor = torch.linalg.matrix_norm(laplacian, ord=2)
        elif self.normalization == "frob": # Frobenius
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
## Directed Graph Convolution: Message Passing
##################################################
class MPGraphConv(MessagePassing):
    def __init__(self, args):
        super().__init__(aggr="add")
        self.num_nodes = args.num_nodes # V
        self.embed_dim = args.embed_dim # D
        self.c_in = args.c_in # C_i
        self.c_out = args.c_out # C_o
        self.activation = {
            "relu": nn.ReLU(), 
            "softplus": nn.Softplus(),
            "tanh": nn.Tanh(), 
            "selu": nn.SELU(), 
            "lrelu": nn.LeakyReLU(), 
            "prelu": nn.PReLU(), 
            "sigmoid": nn.Sigmoid()
        }[args.activation] if args.activation is not None else None
        self.self_loops = args.self_loops
        self.thresh = args.thresh
        self.normalization = args.normalization
        if args.message_net is None:
            self.message_net = nn.Linear(self.c_in, self.c_out)
        elif args.message_net == "FC":
            self.message_net = FullyConnectedNet(**args.message_net_args)
        elif args.message_net == "TCN":
            self.message_net = TCN(**args.message_net_args)
        elif args.message_net == "SCINet":
            self.message_net = StackedSCINet(**args.message_net_args)
        self.update_func = nn.Linear(self.c_in + self.c_out, self.c_out)

    def calc_adj_mat(self, node_embeddings: torch.Tensor, list_repr: bool=True) -> torch.Tensor:
        """
        -------Arguments-------
        node_embeddings: (V, 2*D)
        --------Outputs--------
        adj_mat: (V, V)  if list_repr == False
        adj_list, weights: (2, E), (E)  if list_repr == True
        """
        assert self.embed_dim == node_embeddings.shape[1] / 2
        adj_mat = node_embeddings[:, :self.embed_dim] @ node_embeddings[:, self.embed_dim:].T
        if self.activation is not None:
            adj_mat = self.activation(adj_mat)
        if not self.self_loops:
            adj_mat = adj_mat.fill_diagonal_(0) # Remove self loops
        if self.thresh > 0:
            adj_mat = torch.where(torch.abs(adj_mat) > 1, adj_mat, 0)
        if self.normalization == "spec": # Spectral (maximum singular value)
            adj_mat /= torch.linalg.matrix_norm(adj_mat, ord=2)
        elif self.normalization == "frob": # Frobenius
            adj_mat /= torch.linalg.matrix_norm(adj_mat)
        if list_repr:
            return dense_to_sparse(adj_mat)
        else:
            return adj_mat
        
    def forward(
            self, x: torch.Tensor, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        -------Arguments-------
        x: (V, C_i)
        node_embeddings: (V, 2*D)
        --------Outputs--------
        x_gconv: (V, C_o)
        """
        adj_list, adj_weight = self.calc_adj_mat(node_embeddings) # (2, E), (E)
        return self.propagate(adj_list, x=x, adj_weight=adj_weight)

    def message(self, x_j, adj_weight):
        """
        -------Arguments-------
        x_j: (E, C_i)
        --------Outputs--------
        message_i: (E, C_o)
        """
        message_i = self.message_net(x_j) # (E, C_o)
        return message_i * adj_weight[:, None] # (E, C_o)

    def update(self, aggr_out, x):
        """
        -------Arguments-------
        aggr_out: (V, C_o)
        x: (V, C_i)
        --------Outputs--------
        out: (V, C_o)
        """
        return self.update_func(torch.cat([x, aggr_out], dim=1))








