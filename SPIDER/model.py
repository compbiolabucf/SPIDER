import numpy as np, random
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
import random
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax


def mean_act(x):
    return torch.clamp(torch.exp(x), 1e-5, 1e6)
def disp_act(x):
    return torch.clamp(F.softplus(x), 1e-4, 1e4)
def pi_act(x):
    return torch.sigmoid(x)

class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None, device='cuda:0'):
        super().__init__()
        self.device = device
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels, device=self.device) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.sum() / (n_samples ** 2 - n_samples)
        return self.bandwidth

    def forward(self, X):
        X = X.to(self.device)
        L2_distances = torch.cdist(X, X) ** 2  
        bw = self.get_bandwidth(L2_distances)
        bw = torch.tensor(bw, device=self.device) if not isinstance(bw, torch.Tensor) else bw.to(self.device)
        K = torch.exp(-L2_distances[None, ...] / (bw * self.bandwidth_multipliers)[:, None, None])
        return K.sum(dim=0)
class MMDLoss(nn.Module):
    def __init__(self, kernel=None, device='cuda:0'):
        super().__init__()
        self.device = device
        self.kernel = kernel if kernel is not None else RBF(device=self.device)
    def forward(self, X, Y):
        X = X.to(self.device)
        Y = Y.to(self.device)
        K = self.kernel(torch.vstack([X, Y]))
        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY

def ZINB_loss(y_true, mean, disp, pi, device):
    """
    Computes the Zero-Inflated Negative Binomial (ZINB) loss.

    Args:
        y_true (torch.Tensor): Ground truth tensor.
        mean (torch.Tensor): Predicted mean tensor.
        disp (torch.Tensor): Predicted dispersion tensor.
        pi (torch.Tensor): Predicted zero-inflation probability tensor.
        device (torch.device): Device to perform the computation on.

    Returns:
        torch.Tensor: Computed ZINB loss.
    """
    eps = 1e-10
    r = torch.minimum(disp, torch.tensor(1e6, device=device))
    t1 = torch.lgamma(r + eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(y_true + r + eps)
    t2 = (r + y_true) * torch.log(1.0 + (mean / (r + eps))) + (y_true * (torch.log(r + eps) - torch.log(mean + eps)))

    NB = t1 + t2 - torch.log(1 - pi + eps)

    z1 = torch.pow(r / (mean + r + eps), r)
    zero_inf = -torch.log(pi + (1 - pi) * z1 + eps)

    return torch.mean(torch.where(y_true < 1e-8, zero_inf, NB))

class GATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper


    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        # if isinstance(in_channels, int):
        #     self.lin_src = Linear(in_channels, heads * out_channels,
        #                           bias=False, weight_initializer='glorot')
        #     self.lin_dst = self.lin_src
        # else:
        #     self.lin_src = Linear(in_channels[0], heads * out_channels, False,
        #                           weight_initializer='glorot')
        #     self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
        #                           weight_initializer='glorot')

        self.lin_src = nn.Parameter(torch.zeros(size=(in_channels, out_channels)))
        nn.init.xavier_normal_(self.lin_src.data, gain=1.414)
        self.lin_dst = self.lin_src


        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))
        nn.init.xavier_normal_(self.att_src.data, gain=1.414)
        nn.init.xavier_normal_(self.att_dst.data, gain=1.414)

        # if bias and concat:
        #     self.bias = Parameter(torch.Tensor(heads * out_channels))
        # elif bias and not concat:
        #     self.bias = Parameter(torch.Tensor(out_channels))
        # else:
        #     self.register_parameter('bias', None)

        self._alpha = None
        self.attentions = None

        # self.reset_parameters()

    # def reset_parameters(self):
    #     self.lin_src.reset_parameters()
    #     self.lin_dst.reset_parameters()
    #     glorot(self.att_src)
    #     glorot(self.att_dst)
    #     # zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights=None, attention=True, tied_attention = None):
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            # x_src = x_dst = self.lin_src(x).view(-1, H, C)
            x_src = x_dst = torch.mm(x, self.lin_src).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        if not attention:
            return x[0].mean(dim=1)
            # return x[0].view(-1, self.heads * self.out_channels)

        if tied_attention == None:
            # Next, we compute node-level attention coefficients, both for source
            # and target nodes (if present):
            alpha_src = (x_src * self.att_src).sum(dim=-1)
            alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
            alpha = (alpha_src, alpha_dst)
            self.attentions = alpha
        else:
            alpha = tied_attention


        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        # if self.bias is not None:
        #     out += self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        # Given egel-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        #alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = torch.sigmoid(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha  # Save for later use.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)




class SPIDER(nn.Module):
    """
    DUSTED: A dual-attention spatial transcriptomics enhanced denoiser model.

    Args:
        hidden_dims (list): List of dimensions for input, hidden, and output layers.
    """
    def __init__(self, hidden_dims,num_classes,alpha=1.5):
        super(SPIDER, self).__init__()
        [in_dim, num_hidden, out_dim] = hidden_dims
        self.conv1 = GATConv(in_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv2 = GATConv(num_hidden, out_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv3 = GATConv(out_dim*2, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.disp = GATConv(num_hidden, in_dim, heads=1, concat=False,
                            dropout=0, add_self_loops=False, bias=False)
        self.mean = GATConv(num_hidden, in_dim, heads=1, concat=False,
                            dropout=0, add_self_loops=False, bias=False)
        self.pi = GATConv(num_hidden, in_dim, heads=1, concat=False,
                          dropout=0, add_self_loops=False, bias=False)
        self.alpha = alpha

        self.conv_gene_psd1 = GATConv(in_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv_gene_psd2 = GATConv(num_hidden, out_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)

        self.conv_gene_std1 = GATConv(in_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv_gene_std2 = GATConv(num_hidden, out_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        
        self.conv_gene3 = GATConv(out_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.disp_gene = GATConv(num_hidden, in_dim, heads=1, concat=False,
                            dropout=0, add_self_loops=False, bias=False)
        self.mean_gene = GATConv(num_hidden, in_dim, heads=1, concat=False,
                            dropout=0, add_self_loops=False, bias=False)
        self.pi_gene = GATConv(num_hidden, in_dim, heads=1, concat=False,
                          dropout=0, add_self_loops=False, bias=False)

        self.mlp_classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], num_classes),
            nn.Softmax(dim=1)
        )
    def forward(self, features, featpsd, edge_index, edge_index_std, edge_index_psd, scale_factor):
        h1 = F.elu(self.conv1(features, edge_index))
        h2 = self.conv2(h1, edge_index)

        p1 = F.elu(self.conv_gene_psd1(featpsd, edge_index_psd))
        p2 = self.conv_gene_psd2(p1, edge_index_psd)
        cell_type_ratio = self.mlp_classifier(p2)

        s1 = F.elu(self.conv_gene_std1(features, edge_index_std))
        s2 = self.conv_gene_std2(s1, edge_index_std)

        h2s2 = torch.cat([h2,s2],dim=1)
        h3 = F.elu(self.conv3(h2s2, edge_index, attention=True,
                              tied_attention=self.conv1.attentions))
        
        pi = pi_act(self.pi(h3, edge_index))
        disp = disp_act(self.disp(h3, edge_index))
        mean = mean_act(self.mean(h3, edge_index))
        mean = (mean.T * scale_factor).T

        return mean, disp, pi, h2, s2, p2, h2s2, cell_type_ratio



