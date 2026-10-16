"""Build the atom and motif graph consumed by the released PHENOM model."""

from __future__ import annotations

import torch
from rdkit import Chem

from .mol_to_graph import from_rdmol
from .randomwalk_pe import add_rw_pe
from .tree_decomposition import tree_decomposition


def smiles_to_graph(smiles: str, one_hot: bool = True) -> dict:
    """Encode one valid SMILES using the checkpoint-compatible graph schema."""
    mol = Chem.MolFromSmiles(smiles)
    atom_ids, atomic_nums, atom_feat, edge_index, edge_attr = from_rdmol(mol, one_hot)
    tree_edge_index, atom2clique_index, num_cliques, x_clique = tree_decomposition(
        mol, return_vocab=True
    )
    if atom2clique_index.nelement() == 0:
        num_cliques = len(mol.GetAtoms())
        x_clique = torch.tensor([3] * num_cliques)
        atom2clique_index = torch.stack(
            [torch.arange(num_cliques), torch.arange(num_cliques)]
        )

    if num_cliques == 1:
        clique_pe = torch.zeros(1, 20).float()
    else:
        clique_pe = add_rw_pe(
            tree_edge_index.long(), num_cliques,
            edge_weight=None, walk_length=20, device="cpu",
        )

    return {
        "smiles": smiles,
        "atom_idx": atom_ids,
        "atom_types": atomic_nums,
        "atom_feature": atom_feat,
        "atom_edge_index": edge_index,
        "atom_edge_attr": edge_attr,
        "atom_num_nodes": atom_ids.shape[0],
        "tree_edge_index": tree_edge_index.long(),
        "atom2clique_index": atom2clique_index.long(),
        "x_clique": x_clique.long().view(-1, 1),
        "clique_num_nodes": num_cliques,
        "clique_pe": clique_pe,
    }
