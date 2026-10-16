import torch

import torch.nn.functional as F
from torch_geometric.utils import softmax

from torch_geometric.utils import scatter
from torch_geometric.nn import global_add_pool
from models.layers import MLP, LayerNorm
from torch_geometric.utils import to_dense_batch


EPS = 1e-15

def dense_mincut_pool(x, adj, s, mask=None, cluster_drop_node=None):
    r"""The MinCut pooling operator from the `"Spectral Clustering in Graph
    Neural Networks for Graph Pooling" <https://arxiv.org/abs/1907.00481>`_
    paper

    .. math::
        \mathbf{X}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{X}

        \mathbf{A}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})

    based on dense learned assignments :math:`\mathbf{S} \in \mathbb{R}^{B
    \times N \times C}`.
    Returns the pooled node feature matrix, the coarsened and symmetrically
    normalized adjacency matrix and two auxiliary objectives: (1) The MinCut
    loss

    .. math::
        \mathcal{L}_c = - \frac{\mathrm{Tr}(\mathbf{S}^{\top} \mathbf{A}
        \mathbf{S})} {\mathrm{Tr}(\mathbf{S}^{\top} \mathbf{D}
        \mathbf{S})}

    where :math:`\mathbf{D}` is the degree matrix, and (2) the orthogonality
    loss

    .. math::
        \mathcal{L}_o = {\left\| \frac{\mathbf{S}^{\top} \mathbf{S}}
        {{\|\mathbf{S}^{\top} \mathbf{S}\|}_F} -\frac{\mathbf{I}_C}{\sqrt{C}}
        \right\|}_F.

    Args:
        x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
            \times N \times F}` with batch-size :math:`B`, (maximum)
            number of nodes :math:`N` for each graph, and feature dimension
            :math:`F`.
        adj (Tensor): Symmetrically normalized adjacency tensor
            :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
        s (Tensor): Assignment tensor :math:`\mathbf{S} \in \mathbb{R}^{B
            \times N \times C}` with number of clusters :math:`C`. The softmax
            does not have to be applied beforehand, since it is executed
            within this method.
        mask (BoolTensor, optional): Mask matrix
            :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
            the valid nodes for each graph. (default: :obj:`None`)

    :rtype: (:class:`Tensor`, :class:`Tensor`, :class:`Tensor`,
        :class:`Tensor`)
    """

    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    (batch_size, num_nodes, _), k = x.size(), s.size(-1)

    s = torch.softmax(s, dim=-1)

    if mask is not None:
        s = s * mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x_mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)

        if cluster_drop_node is not None:
            x_mask = cluster_drop_node.view(batch_size, num_nodes, 1).to(x.dtype)

        x = x * x_mask

    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    # MinCut regularization.
    mincut_num = _rank3_trace(out_adj)
    d_flat = torch.einsum('ijk->ij', adj)
    d = _rank3_diag(d_flat)
    mincut_den = _rank3_trace(
        torch.matmul(torch.matmul(s.transpose(1, 2), d), s))
    mincut_loss = -(mincut_num / (mincut_den + EPS))
    mincut_loss = torch.mean(mincut_loss)

    # Orthogonality regularization.
    ss = torch.matmul(s.transpose(1, 2), s)
    i_s = torch.eye(k).type_as(ss)
    ortho_loss = torch.norm(
        ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
        i_s / torch.norm(i_s), dim=(-1, -2))
    ortho_loss = torch.mean(ortho_loss)

    # Fix and normalize coarsened adjacency matrix.
    ind = torch.arange(k, device=out_adj.device)
    out_adj[:, ind, ind] = 0
    d = torch.einsum('ijk->ij', out_adj)
    d = torch.sqrt(d)[:, None] + EPS
    out_adj = (out_adj / d) / d.transpose(1, 2)

    # out_loss = mincut_loss + ortho_loss

    return s, out, out_adj, mincut_loss, ortho_loss

def _rank3_trace(x):
    return torch.einsum('ijj->i', x)


def _rank3_diag(x):
    eye = torch.eye(x.size(1)).type_as(x)
    out = eye * x.unsqueeze(2).expand(*x.size(), x.size(1))
    return out





class MotifPool(torch.nn.Module):
    def __init__(self, hidden_channels, heads, pe_walk_length=20, dropout_attn_score=0): 
        super().__init__()
        assert hidden_channels % heads == 0 

        self.pe_lin = torch.nn.Linear(pe_walk_length, hidden_channels, bias=False)
        self.pe_norm = LayerNorm(hidden_channels)

        self.atom_proj = torch.nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.atom_norm = LayerNorm(hidden_channels)

        self.attn = torch.nn.MultiheadAttention(
                hidden_channels,
                heads,
                batch_first=True
            )
        self.clique_norm = LayerNorm(hidden_channels)
        mh_hidden_channels = hidden_channels // heads
        
        self.score_proj = torch.nn.ModuleList()
        for _ in range(heads): 
            self.score_proj.append( MLP([ mh_hidden_channels, mh_hidden_channels*2, 1]) )
        
        self.heads = heads
        self.mh_hidden_channels = mh_hidden_channels
        self.dropout_attn_score = dropout_attn_score

    def reset_parameters(self):
        
        self.pe_lin.reset_parameters()
        self.pe_norm.reset_parameters()
        self.atom_proj.reset_parameters()
        self.atom_norm.reset_parameters()
        self.attn._reset_parameters()
        self.clique_norm.reset_parameters()
        for m in self.score_proj:
            m.reset_parameters()

    def forward(self, x, clique_x, clique_pe, atom2clique_index, clique_batch):
        row, col = atom2clique_index
        H = self.heads
        C = self.mh_hidden_channels

        clique_pe = self.pe_norm(self.pe_lin(clique_pe))
        clique_hx = scatter(x[row], col, dim=0, dim_size=clique_x.size(0), reduce='mean')
        clique_hx = self.atom_norm(self.atom_proj(clique_hx))
        clique_x = clique_x + clique_pe + clique_hx

        ## self attention
        h, mask = to_dense_batch(clique_x, clique_batch)
        h, _ = self.attn(h, h, h, key_padding_mask=~mask,
                             need_weights=False)
        h = h[mask]
        clique_x = clique_x + h
        clique_x = self.clique_norm(clique_x)
        ## GNN scoring
        score_clique = clique_x.view(-1, H, C)
        score = torch.cat([ mlp(score_clique[:, i]) for i, mlp in enumerate(self.score_proj) ], dim=-1)
        score = F.dropout(score, p=self.dropout_attn_score, training=self.training)
        alpha = softmax(score, clique_batch)    
        
        ## multihead aggregation of drug feature
        drug_feat = clique_x.view(-1, H, C) * alpha.view(-1, H, 1) 
        drug_feat = drug_feat.view(-1, H * C)
        drug_feat = global_add_pool(drug_feat, clique_batch)

        return drug_feat, clique_x, alpha
