import math
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
            torch.FloatTensor(self.embed_dim, self.cheb_k, self.c_in, self.c_out)) # (D, K, C_i, C_o)
        self.bias_pool = nn.Parameter(torch.FloatTensor(self.embed_dim, self.c_out))  # (D, C_o)

    def calc_adj_mat(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        -------Arguments-------
        node_embeddings: (V, D)
        --------Outputs--------
        normalized_adj_mat: (V, V)
        """
        adj_mat = node_embeddings @ node_embeddings.T
        if self.adj_mat_type == "Bai":
            return F.softmax(F.relu(adj_mat))
        else:
            adj_mat_rowsum = torch.sum(adj_mat, axis=1).reshape(self.embed_dim, 1)
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
        return x_gconv

##################################################
## Directed Graph Convolution
##################################################
class DirGraphConv(nn.Module):
    def __init__(self, args):
        super(DirGraphConv, self).__init__()















