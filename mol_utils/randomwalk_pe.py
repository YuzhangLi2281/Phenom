"""Dense random-walk positional encodings for ligand motif graphs."""

import torch
from torch_geometric.utils import scatter

def add_rw_pe(edge_index, num_nodes, edge_weight=None, walk_length=20, device = 'cpu'):
    """Return diagonal transition probabilities for walks of length 1..L."""
    if edge_index is None or num_nodes < 1 or walk_length < 1:
        raise ValueError("edge_index, num_nodes and walk_length must be valid")
    num_edges = edge_index.size(1)
    edge_index = edge_index.to(device)
    
    row, col = edge_index
    N = num_nodes
    if edge_weight is None:
        value = torch.ones(num_edges, device=row.device)
    else:
        value = edge_weight.to(device=device, dtype=torch.float)
        if value.numel() != num_edges:
            raise ValueError("edge_weight must contain one value per edge")
    value = value / scatter(value, row, dim_size=N, reduce='sum').clamp(min=1)[row]

    adj = torch.zeros((N, N), device=device)
    adj[row, col] = value
    loop_index = torch.arange(N, device=device)

    out = adj
    pe_list = [out[loop_index, loop_index]]
    for _ in range(walk_length - 1):
        out = out @ adj
        pe_list.append(out[loop_index, loop_index])

    pe = torch.stack(pe_list, dim=-1)

    return pe
