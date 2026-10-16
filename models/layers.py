from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.conv import GCNConv
from torch_geometric.utils import degree, subgraph

from models.pna import PNAConv


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        if self.bias is not None:  # Only reset bias if it exists
            nn.init.zeros_(self.bias)

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class GCNCluster(torch.nn.Module):
    def __init__(self, dims, out_norm=False, in_norm=False): #L=nb_hidden_layers
        super().__init__()
        list_Conv_layers = [ GCNConv(dims[idx-1], dims[idx]) for idx in range(1,len(dims)) ]
        self.Conv_layers = nn.ModuleList(list_Conv_layers)
        self.hidden_layers = len(dims) - 2

        self.out_norm = out_norm
        self.in_norm = in_norm

        if self.out_norm:
            self.out_ln = LayerNorm(dims[-1])
        if self.in_norm:
            self.in_ln = LayerNorm(dims[0])

    def reset_parameters(self):
        for idx in range(self.hidden_layers+1):
            self.Conv_layers[idx].reset_parameters()
        if self.out_norm:
            self.out_ln.reset_parameters()
        if self.in_norm:
            self.in_ln.reset_parameters()

    def forward(self, x, edge_index):
        y = x
        # Input Layer Norm
        if self.in_norm:
            y = self.in_ln(y)

        for idx in range(self.hidden_layers):
            y = self.Conv_layers[idx](y, edge_index)
            y = F.relu(y)
        y = self.Conv_layers[-1](y, edge_index)

        if self.out_norm:
            y = self.out_ln(y)

        return y

class PosLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, init_value=0.2,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(PosLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # center_value = init_value
        # lower_bound = center_value - center_value/10
        # upper_bound = center_value + center_value/10

        lower_bound = init_value/2
        upper_bound = init_value
        weight = nn.init.uniform_(torch.empty((out_features, in_features),**factory_kwargs), a=lower_bound, b=upper_bound)
        # weight = nn.init.kaiming_uniform_(torch.empty((out_features, in_features),**factory_kwargs), a=math.sqrt(5))
        weight = torch.abs(weight)
        self.weight = nn.Parameter(weight.log())
        # self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()



    def reset_parameters(self) -> None:
        # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # self.weight = torch.abs(self.weight).log()
        if self.bias is not None:
            nn.init.uniform_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight.exp(), self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class MLP(nn.Module):

    def __init__(self, dims, out_norm=False, in_norm=False, bias=True): #L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [ nn.Linear(dims[idx-1], dims[idx], bias=bias) for idx in range(1,len(dims)) ]
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.hidden_layers = len(dims) - 2

        self.out_norm = out_norm
        self.in_norm = in_norm

        if self.out_norm:
            self.out_ln = LayerNorm(dims[-1])
        if self.in_norm:
            self.in_ln = LayerNorm(dims[0])

    def reset_parameters(self):
        for idx in range(self.hidden_layers+1):
            self.FC_layers[idx].reset_parameters()
        if self.out_norm:
            self.out_ln.reset_parameters()
        if self.in_norm:
            self.in_ln.reset_parameters()

    def forward(self, x):
        y = x
        # Input Layer Norm
        if self.in_norm:
            y = self.in_ln(y)

        for idx in range(self.hidden_layers):
            y = self.FC_layers[idx](y)
            y = F.relu(y)
        y = self.FC_layers[-1](y)

        if self.out_norm:
            y = self.out_ln(y)

        return y

class Drug_PNAConv(nn.Module):
    def __init__(self, mol_deg, hidden_channels, edge_channels,
                 pre_layers=2, post_layers=2,
                 aggregators=['sum', 'mean', 'min', 'max', 'std'],
                 scalers=['identity', 'amplification', 'attenuation'],
                 num_towers=4,
                 dropout=0.1):
        super(Drug_PNAConv, self).__init__()

        self.bond_encoder = torch.nn.Linear(edge_channels, hidden_channels)

        self.atom_conv = PNAConv(
            in_channels=hidden_channels, out_channels=hidden_channels,
            edge_dim=hidden_channels, aggregators=aggregators,
            scalers=scalers, deg=mol_deg, pre_layers=pre_layers,
            post_layers=post_layers,towers=num_towers,divide_input=True,
        )
        self.atom_norm = LayerNorm(hidden_channels)

        self.dropout = dropout

    def reset_parameters(self):
        self.atom_conv.reset_parameters()
        self.atom_norm.reset_parameters()


    def forward(self, atom_x, bond_x, atom_edge_index):
        atom_in = atom_x
        bond_x = self.bond_encoder(bond_x.squeeze())
        atom_x = atom_in + F.relu(self.atom_norm(self.atom_conv(atom_x, atom_edge_index, bond_x)))
        atom_x = F.dropout(atom_x, self.dropout, training=self.training)

        return atom_x


class Protein_PNAConv(nn.Module):
    def __init__(self, prot_deg, hidden_channels, edge_channels,
                 pre_layers=2, post_layers=2,
                 aggregators=['sum', 'mean', 'min', 'max', 'std'],
                 scalers=['identity', 'amplification', 'attenuation'],
                 num_towers=4,
                 dropout=0.1):
        super(Protein_PNAConv, self).__init__()

        self.conv = PNAConv(in_channels=hidden_channels,
                            out_channels=hidden_channels,
                            edge_dim=edge_channels,
                            aggregators=aggregators,
                            scalers=scalers,
                            deg=prot_deg,
                            pre_layers=pre_layers,
                            post_layers=post_layers,
                            towers=num_towers,
                            divide_input=True,
                            )
                            
        self.norm = LayerNorm(hidden_channels)
        self.dropout = dropout
        
    def reset_parameters(self):
        self.conv.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, x, prot_edge_index, prot_edge_attr):
        x_in = x
        x = x_in + F.relu(self.norm(self.conv(x, prot_edge_index, prot_edge_attr)))
        x = F.dropout(x, self.dropout, training=self.training)

        return x


def unbatch(src, batch, dim: int = 0):
    r"""Splits :obj:`src` according to a :obj:`batch` vector along dimension
    :obj:`dim`.

    Args:
        src (Tensor): The source tensor.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            entry in :obj:`src` to a specific example. Must be ordered.
        dim (int, optional): The dimension along which to split the :obj:`src`
            tensor. (default: :obj:`0`)

    :rtype: :class:`List[Tensor]`

    Example:

        >>> src = torch.arange(7)
        >>> batch = torch.tensor([0, 0, 0, 1, 1, 2, 2])
        >>> unbatch(src, batch)
        (tensor([0, 1, 2]), tensor([3, 4]), tensor([5, 6]))
    """
    sizes = degree(batch, dtype=torch.long).tolist()
    return src.split(sizes, dim)



def dropout_node(edge_index, p, num_nodes, batch, training):
    r"""Randomly drops nodes from the adjacency matrix
    :obj:`edge_index` with probability :obj:`p` using samples from
    a Bernoulli distribution.

    The method returns (1) the retained :obj:`edge_index`, (2) the edge mask
    indicating which edges were retained. (3) the node mask indicating
    which nodes were retained.

    Args:
        edge_index (LongTensor): The edge indices.
        p (float, optional): Dropout probability. (default: :obj:`0.5`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)

    :rtype: (:class:`LongTensor`, :class:`BoolTensor`, :class:`BoolTensor`)

    Examples:

        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> edge_index, edge_mask, node_mask = dropout_node(edge_index)
        >>> edge_index
        tensor([[0, 1],
                [1, 0]])
        >>> edge_mask
        tensor([ True,  True, False, False, False, False])
        >>> node_mask
        tensor([ True,  True, False, False])
    """
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or p == 0.0:
        node_mask = edge_index.new_ones(num_nodes, dtype=torch.bool)
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask, node_mask
    
    prob = torch.rand(num_nodes, device=edge_index.device)
    node_mask = prob > p
    
    ## ensure no graph is totally dropped out
    batch_tf = global_add_pool(node_mask.view(-1,1),batch).flatten()
    unbatched_node_mask = unbatch(node_mask, batch)
    node_mask_list = []
    
    for true_false, sub_node_mask in zip(batch_tf, unbatched_node_mask):
        if true_false.item():
            node_mask_list.append(sub_node_mask)
        else:
            perm = torch.randperm(sub_node_mask.size(0))
            idx = perm[:1]
            sub_node_mask[idx] = True
            node_mask_list.append(sub_node_mask)
            
    node_mask = torch.cat(node_mask_list)
    
    edge_index, _, edge_mask = subgraph(node_mask, edge_index,
                                        num_nodes=num_nodes,
                                        return_edge_mask=True)
    return edge_index, edge_mask, node_mask

def dropout_edge(edge_index: Tensor, p: float = 0.5,
                 force_undirected: bool = False,
                 training: bool = True) -> Tuple[Tensor, Tensor]:
    r"""Randomly drops edges from the adjacency matrix
    :obj:`edge_index` with probability :obj:`p` using samples from
    a Bernoulli distribution.

    The method returns (1) the retained :obj:`edge_index`, (2) the edge mask
    or index indicating which edges were retained, depending on the argument
    :obj:`force_undirected`.

    Args:
        edge_index (LongTensor): The edge indices.
        p (float, optional): Dropout probability. (default: :obj:`0.5`)
        force_undirected (bool, optional): If set to :obj:`True`, will either
            drop or keep both edges of an undirected edge.
            (default: :obj:`False`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)

    :rtype: (:class:`LongTensor`, :class:`BoolTensor` or :class:`LongTensor`)

    Examples:

        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> edge_index, edge_mask = dropout_edge(edge_index)
        >>> edge_index
        tensor([[0, 1, 2, 2],
                [1, 2, 1, 3]])
        >>> edge_mask # masks indicating which edges are retained
        tensor([ True, False,  True,  True,  True, False])

        >>> edge_index, edge_id = dropout_edge(edge_index,
        ...                                    force_undirected=True)
        >>> edge_index
        tensor([[0, 1, 2, 1, 2, 3],
                [1, 2, 3, 0, 1, 2]])
        >>> edge_id # indices indicating which edges are retained
        tensor([0, 2, 4, 0, 2, 4])
    """
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or p == 0.0:
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask

    row, col = edge_index

    edge_mask = torch.rand(row.size(0), device=edge_index.device) >= p

    if force_undirected:
        edge_mask[row > col] = False

    edge_index = edge_index[:, edge_mask]

    if force_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_mask = edge_mask.nonzero().repeat((2, 1)).squeeze()

    return edge_index, edge_mask

