from typing import Optional, Tuple

from torch.nn.init import zeros_, xavier_uniform_
from torch_geometric.typing import Adj, OptTensor
import torch.nn as nn
import torch.nn.functional as F


import torch
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from torch_geometric.nn.inits import zeros, glorot
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (Tensor, OptTensor, Optional[int], bool, bool,
    #       Optional[int]) -> PairTensor  # noqa
    pass


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool,
    #        Optional[int]) -> SparseTensor  # noqa
    pass


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]



class AttentionDepthGCNConv(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, att_hidden_dim: int = None, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(AttentionDepthGCNConv, self).__init__(** kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        glorot(self.lin.weight)
        self.residual_proj = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

        self.att_hidden_dim = att_hidden_dim if att_hidden_dim is not None else out_channels // 2
        self.att_proj = nn.Linear(out_channels, self.att_hidden_dim)
        self.att_score = nn.Linear(self.att_hidden_dim, 1)
        xavier_uniform_(self.att_proj.weight)
        zeros_(self.att_proj.bias)
        xavier_uniform_(self.att_score.weight)
        zeros_(self.att_score.bias)

        self.m_offset_proj = None

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.bias is not None:
            zeros_(self.bias)
        if isinstance(self.residual_proj, nn.Linear):
            self.residual_proj.reset_parameters()
        self.att_proj.reset_parameters()
        self.att_score.reset_parameters()
        self.m_offset_proj = None
        self._cached_edge_index = None
        self._cached_adj_t = None

    def _compute_depth_attention(self, depth_features: Tensor) -> Tensor:
        L, num_nodes, _ = depth_features.shape
        proj_features = self.att_proj(depth_features)
        proj_features = torch.tanh(proj_features)
        depth_scores = self.att_score(proj_features)
        depth_weights = F.softmax(depth_scores, dim=0)  # [L, num_nodes, 1]
        weighted_features = (depth_features * depth_weights).sum(dim=0)  # [num_nodes, out_channels]
        return weighted_features

    def _compute_node_level_m(self, depth_features: Tensor) -> Tuple[Tensor, Tensor]:
        conv_time, num_nodes, out_dim = depth_features.shape
        depth_per_node = depth_features.permute(1, 0, 2)

        node_seq_stats = depth_per_node.reshape(num_nodes, -1)

        if self.m_offset_proj is None or self.m_offset_proj.in_features != node_seq_stats.shape[1]:
            self.m_offset_proj = nn.Linear(node_seq_stats.shape[1], 1).to(node_seq_stats.device)
            xavier_uniform_(self.m_offset_proj.weight)
            zeros_(self.m_offset_proj.bias)

        delta_m_cont_i = self.m_offset_proj(node_seq_stats)  # [num_nodes, 1]
        delta_m_cont_i = (conv_time - 1) * torch.sigmoid(delta_m_cont_i)

        global_mean = delta_m_cont_i.mean()
        global_std = delta_m_cont_i.std()
        delta_m_cont_i = torch.clamp(delta_m_cont_i, global_mean - 2*global_std, global_mean + 2*global_std)

        delta_m_int_i = torch.round(delta_m_cont_i)
        delta_m_i = delta_m_int_i + (delta_m_cont_i - delta_m_cont_i.detach())
        delta_m_global_cont = delta_m_cont_i.mean(dim=0)  # [1]
        delta_m_global_int = torch.round(delta_m_global_cont)
        m = torch.clamp(delta_m_global_int + 1, min=1, max=conv_time).long()

        return m, delta_m_i

    def forward(self, x: Tensor, edge_index: Adj,
                conv_time: int = 1, edge_weight: OptTensor = None, is_source: bool = True) -> Tensor:

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops
                    )
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]
            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops
                    )
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
        out = self.lin(x)

        if conv_time > 0:
            depth_features = []
            for _ in range(conv_time):
                out = self.propagate(edge_index, x=out, edge_weight=edge_weight, size=None)
                depth_features.append(out)
            depth_features = torch.stack(depth_features, dim=0)  # [conv_time, num_nodes, out_channels]

            m, _ = self._compute_node_level_m(depth_features)
            m_idx = m - 1

            filtered_features = depth_features[m_idx:, :, :]
            if filtered_features.shape[0] == 0:
                out = depth_features[-1]
            else:
                out = self._compute_depth_attention(filtered_features)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return torch.sparse.mm(adj_t, x) if self.aggr == 'add' else super().message_and_aggregate(adj_t, x)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels}, att_hidden_dim={self.att_hidden_dim})'



class GCNConv(MessagePassing):
    r"""
    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and compute
            symmetric normalization coefficients on the fly.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(GCNConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None


    def forward(self, x: Tensor, edge_index: Adj,
                conv_time = 1, edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        out = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        for i in range(conv_time):
            out = self.propagate(edge_index, x=out, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out += self.bias

        return out


    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)