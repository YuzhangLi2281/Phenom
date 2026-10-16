import math
from types import NoneType
from typing import List, Optional, Tuple, Union, overload

import torch
import torch.nn.functional as F
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, PairTensor, Size, SparseTensor
from torch_geometric.utils import softmax, to_dense_batch


from .layers import LayerNorm, MLP
from torch import Tensor

class InteractionLayer(torch.nn.Module):
    """Symmetric pocket-to-motif Transformer interaction."""

    def __init__(self, hidden_channels, heads, dropout_attn_score=0.2):
        super().__init__()
        self.norm = LayerNorm(hidden_channels)
        self.attention = AttentionConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels // heads,
            heads=heads,
            concat=True,
            attn_dropout=dropout_attn_score,
        )
        self.update = MLP(
            [hidden_channels, hidden_channels * 4, hidden_channels],
            out_norm=True,
        )

    def reset_parameters(self):
        self.norm.reset_parameters()
        self.attention.reset_parameters()
        self.update.reset_parameters()

    @staticmethod
    def connect_same_batch(pocket_batch, motif_batch):
        return (pocket_batch[:, None] == motif_batch[None, :]).nonzero(
            as_tuple=False
        ).t().contiguous()

    def forward(self, pocket_x, pocket_batch, motif_x, motif_batch):
        pocket_to_motif = self.connect_same_batch(pocket_batch, motif_batch)
        pocket_count = pocket_x.size(0)
        pocket_to_motif = pocket_to_motif.clone()
        pocket_to_motif[1] += pocket_count
        interaction_edges = torch.cat(
            [pocket_to_motif, pocket_to_motif.flip(0)], dim=1
        )

        combined = torch.cat([pocket_x, motif_x], dim=0)
        message, (_, attention) = self.attention(
            self.norm(combined), interaction_edges, return_attention_weights=True
        )
        combined = combined + message
        combined = combined + self.update(combined)

        edge_count = pocket_to_motif.size(1)
        return (
            combined[:pocket_count],
            combined[pocket_count:],
            pocket_to_motif,
            attention[:edge_count],
        )


class CatalyticInteraction(torch.nn.Module):
    """Ligand-guided residue–atom cross-attention for the Phenom interaction fingerprint."""

    def __init__(self, hidden_channels, heads, max_residues=64, dropout=0.1):
        super().__init__()
        self.max_residues = max_residues
        self.task_embedding = torch.nn.Embedding(6, hidden_channels)
        self.residue_norm = torch.nn.LayerNorm(hidden_channels)
        self.atom_norm = torch.nn.LayerNorm(hidden_channels)
        self.selector_key = torch.nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.selector_query = torch.nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.residue_from_atom = torch.nn.MultiheadAttention(
            hidden_channels, heads, dropout=dropout, batch_first=True
        )
        self.atom_from_residue = torch.nn.MultiheadAttention(
            hidden_channels, heads, dropout=dropout, batch_first=True
        )
        self.residue_update = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.Linear(hidden_channels, hidden_channels * 2),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels * 2, hidden_channels),
        )
        self.atom_update = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.Linear(hidden_channels, hidden_channels * 2),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels * 2, hidden_channels),
        )
        self.atom_score = torch.nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.output = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 4, hidden_channels * 4),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels * 4, hidden_channels * 2),
        )

    def forward(
        self, residue_x, residue_batch, atom_x, atom_batch, ligand_context, task_index
    ):
        residue, residue_mask = to_dense_batch(residue_x, residue_batch)
        atom, atom_mask = to_dense_batch(atom_x, atom_batch)
        query = self.selector_query(
            ligand_context + self.task_embedding(task_index.view(-1))
        )
        node_score = torch.sum(
            self.selector_key(self.residue_norm(residue_x)) * query[residue_batch], dim=-1
        ) / residue_x.size(-1) ** 0.5
        selection_score, _ = to_dense_batch(node_score, residue_batch)
        selection_score = selection_score.masked_fill(~residue_mask, -torch.inf)

        count = min(self.max_residues, residue.size(1))
        selected_score, selected_index = selection_score.topk(count, dim=1)
        selected_mask = residue_mask.gather(1, selected_index)
        selected = residue.gather(
            1, selected_index.unsqueeze(-1).expand(-1, -1, residue.size(-1))
        )
        residue_message, residue_atom_attention = self.residue_from_atom(
            self.residue_norm(selected), self.atom_norm(atom), self.atom_norm(atom),
            key_padding_mask=~atom_mask, need_weights=True, average_attn_weights=False,
        )
        selected = selected + residue_message
        selected = selected + self.residue_update(selected)
        atom_message, atom_residue_attention = self.atom_from_residue(
            self.atom_norm(atom), self.residue_norm(selected), self.residue_norm(selected),
            key_padding_mask=~selected_mask, need_weights=True, average_attn_weights=False,
        )
        atom = atom + atom_message
        atom = atom + self.atom_update(atom)

        residue_weight = torch.softmax(
            selected_score.masked_fill(~selected_mask, -torch.inf), dim=1
        )
        residue_pool = torch.sum(selected * residue_weight.unsqueeze(-1), dim=1)
        atom_weight = torch.einsum(
            "bad,bd->ba", self.atom_score(self.atom_norm(atom)),
            self.selector_query(residue_pool),
        ) / atom.size(-1) ** 0.5
        atom_weight = torch.softmax(atom_weight.masked_fill(~atom_mask, -torch.inf), dim=1)
        atom_pool = torch.sum(atom * atom_weight.unsqueeze(-1), dim=1)
        fingerprint = self.output(torch.cat([
            residue_pool, atom_pool, residue_pool * atom_pool,
            torch.abs(residue_pool - atom_pool),
        ], dim=-1))
        return fingerprint, residue_weight, atom_weight, {
            "selected_residue_index": selected_index,
            "residue_atom_attention": residue_atom_attention,
            "atom_residue_attention": atom_residue_attention,
        }


class AttentionConv(MessagePassing):
    r"""The graph transformer operator from the `"Masked Label Prediction:
    Unified Message Passing Model for Semi-Supervised Classification"
    <https://arxiv.org/abs/2009.03509>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \mathbf{W}_2 \mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed via
    multi-head dot product attention:

    .. math::
        \alpha_{i,j} = \textrm{softmax} \left(
        \frac{(\mathbf{W}_3\mathbf{x}_i)^{\top} (\mathbf{W}_4\mathbf{x}_j)}
        {\sqrt{d}} \right)

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
        beta (bool, optional): If set, will combine aggregation and
            skip information via

            .. math::
                \mathbf{x}^{\prime}_i = \beta_i \mathbf{W}_1 \mathbf{x}_i +
                (1 - \beta_i) \underbrace{\left(\sum_{j \in \mathcal{N}(i)}
                \alpha_{i,j} \mathbf{W}_2 \vec{x}_j \right)}_{=\mathbf{m}_i}

            with :math:`\beta_i = \textrm{sigmoid}(\mathbf{w}_5^{\top}
            [ \mathbf{W}_1 \mathbf{x}_i, \mathbf{m}_i, \mathbf{W}_1
            \mathbf{x}_i - \mathbf{m}_i ])` (default: :obj:`False`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). Edge features are added to the keys after
            linear transformation, that is, prior to computing the
            attention dot product. They are also added to final values
            after the same linear transformation. The model is:

            .. math::
                \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i +
                \sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \left(
                \mathbf{W}_2 \mathbf{x}_{j} + \mathbf{W}_6 \mathbf{e}_{ij}
                \right),

            where the attention coefficients :math:`\alpha_{i,j}` are now
            computed via:

            .. math::
                \alpha_{i,j} = \textrm{softmax} \left(
                \frac{(\mathbf{W}_3\mathbf{x}_i)^{\top}
                (\mathbf{W}_4\mathbf{x}_j + \mathbf{W}_6 \mathbf{e}_{ij})}
                {\sqrt{d}} \right)

            (default :obj:`None`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add the transformed root node features to the output and the
            option  :attr:`beta` is set to :obj:`False`. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        attn_dropout: float = 0.,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = False,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.attn_dropout = attn_dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        # Initialize query, key, value projections with Glorot uniform (fan_avg)
        torch.nn.init.xavier_uniform_(self.lin_key.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.lin_query.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.lin_value.weight, gain=1.0)
        torch.nn.init.zeros_(self.lin_key.bias)
        torch.nn.init.zeros_(self.lin_query.bias)
        torch.nn.init.zeros_(self.lin_value.bias)

        # Edge projection if exists
        if self.edge_dim:
            torch.nn.init.xavier_uniform_(self.lin_edge.weight, gain=1.0)

        # Skip connection layer (acts as final layer in residual path)
        torch.nn.init.zeros_(self.lin_skip.weight)
        if self.lin_skip.bias is not None:
            torch.nn.init.zeros_(self.lin_skip.bias)

        # Beta projection if exists
        if self.beta:
            torch.nn.init.zeros_(self.lin_beta.weight)

    @overload
    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        return_attention_weights: NoneType = None,
    ) -> Tensor:
        pass

    @overload
    def forward(  # noqa: F811
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Tensor,
        edge_attr: OptTensor = None,
        return_attention_weights: bool = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        pass

    @overload
    def forward(  # noqa: F811
        self,
        x: Union[Tensor, PairTensor],
        edge_index: SparseTensor,
        edge_attr: OptTensor = None,
        return_attention_weights: bool = None,
    ) -> Tuple[Tensor, SparseTensor]:
        pass

    def forward(  # noqa: F811
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        return_attention_weights: Optional[bool] = None,
    ) -> Union[
            Tensor,
            Tuple[Tensor, Tuple[Tensor, Tensor]],
            Tuple[Tensor, SparseTensor],
    ]:
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor or (torch.Tensor, torch.Tensor)): The input node
                features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            x = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor,
        #                  edge_attr: OptTensor)
        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out = out + x_r

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, #edge_weight: OptTensor, 
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        # if self.lin_edge is not None:
            # assert edge_attr is not None
            # edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
            #                                           self.out_channels)
            # key_j = key_j + edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = F.dropout(alpha, p=self.attn_dropout, training=self.training)
        
        # Apply edge weights after softmax if they exist
        if edge_attr is not None:
        # Add in log space (equivalent to multiplication in probability space)
            alpha = alpha + torch.log(edge_attr.view(-1, 1) + 1e-6)

        
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha

        out = value_j
        out = out * alpha.view(-1, self.heads, 1)
        
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
    


class SuperNode2TokenConv(MessagePassing):
    r"""message passing operator that performs (non-trainable)
    propagation.

    .. math::
        \mathbf{x}^{\prime}_i = \bigoplus_{j \in \mathcal{N(i)}} e_{ji} \cdot
        \mathbf{x}_j

    where :math:`\bigoplus` defines a custom aggregation scheme.

    Args:
        aggr (str or [str] or Aggregation, optional): The aggregation scheme
            to use, *e.g.*, :obj:`"add"`, :obj:`"sum"` :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"` or :obj:`"mul"`.
            In addition, can be any
            :class:`~torch_geometric.nn.aggr.Aggregation` module (or any string
            that automatically resolves to it). (default: :obj:`"sum"`)
        combine_root (str, optional): Specifies whether or how to combine the
            central node representation (one of :obj:`"sum"`, :obj:`"cat"`,
            :obj:`"self_loop"`, :obj:`None`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **inputs:**
          node features :math:`(|\mathcal{V}|, F)` or
          :math:`((|\mathcal{V_s}|, F), (|\mathcal{V_t}|, *))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **outputs:** node features :math:`(|\mathcal{V}|, F)` or
          :math:`(|\mathcal{V_t}|, F)` if bipartite
    """
    def __init__(
        self,
        aggr: Optional[Union[str, List[str], Aggregation]] = "sum",
        combine_root: Optional[str] = None,
        **kwargs,
    ):
        if combine_root not in ['sum', 'cat', 'self_loop', None]:
            raise ValueError(f"Received invalid value for 'combine_root' "
                             f"(got '{combine_root}')")

        super().__init__(aggr, **kwargs)
        self.combine_root = combine_root

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None, size: Size = None) -> Tensor:

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=size)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j



class PhenomInteraction(torch.nn.Module):
    """Progressive stack of bidirectional pocket-to-motif interactions."""

    def __init__(self, hidden_channels, heads, num_interaction_layers,
                 dropout_attn_score=0.2):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            InteractionLayer(hidden_channels, heads, dropout_attn_score)
            for _ in range(num_interaction_layers)
        ])

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, layer_index, pocket_x, pocket_batch, motif_x, motif_batch):
        return self.layers[layer_index](pocket_x, pocket_batch, motif_x, motif_batch)
