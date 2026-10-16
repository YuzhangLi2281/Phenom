from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data, Dataset

from models.enzyme_prediction import ESP, MUTATION, PH
from mol_utils import MoleculeNormalizer, smiles_to_graph


REQUIRED_COLUMNS = ("Protein", "task_index", "label", "is_binary")


class ProteinMoleculeDataset(Dataset):
    """One row is one PHENOME supervision target."""

    def __init__(
        self,
        sequence_data: pd.DataFrame,
        molecule_dict: dict | None = None,
        molecule_folder: str | Path = "",
        protein_dict: dict | None = None,
        protein_folder: str | Path = "",
        standardize: bool = True,
        timeout: int = 10,
        protein_embedding_dim: int | None = None,
        ligand_language_embedding: bool = False,
    ) -> None:
        super().__init__()
        missing = [column for column in REQUIRED_COLUMNS if column not in sequence_data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        frame = sequence_data.reset_index(drop=True).copy()
        frame["task_index"] = frame["task_index"].astype(int)
        if not frame["task_index"].between(0, 5).all():
            raise ValueError("task_index must be between 0 and 5")
        if ((frame["task_index"] == ESP) & ~frame["is_binary"].astype(bool)).any():
            raise ValueError("ESP rows must set is_binary=true")
        if (frame["task_index"] == MUTATION).any():
            missing_wt = frame.loc[frame["task_index"] == MUTATION, "WT_Protein"].isna()
            if missing_wt.any():
                raise ValueError("Mutation rows require WT_Protein")

        frame["ID"] = frame["ID"].astype(str) if "ID" in frame else [f"Row_{i + 1}" for i in range(len(frame))]
        frame["Ligand"] = frame["Ligand"] if "Ligand" in frame else pd.NA
        frame["source"] = frame["source"].astype(str) if "source" in frame else "unknown"
        frame["sample_weight"] = (
            pd.to_numeric(frame["sample_weight"], errors="raise") if "sample_weight" in frame else 1.0
        )
        frame["has_ligand"] = frame["Ligand"].notna() & frame["Ligand"].astype(str).str.strip().ne("")
        if (frame.loc[~frame["task_index"].isin([PH, MUTATION]), "has_ligand"] == 0).any():
            raise ValueError("kcat, Km, kcat/Km, and ESP rows require Ligand")

        self.frame = frame
        self.frame["protein_group_index"] = pd.factorize(frame["Protein"], sort=True)[0]
        ligand_group = pd.factorize(frame["Ligand"], sort=True)[0]
        self.frame["ligand_group_index"] = ligand_group
        self.mol_dict = molecule_dict or {}
        self.molecule_folder = Path(molecule_folder)
        self.prot_dict = protein_dict or {}
        self.protein_folder = Path(protein_folder)
        self.standardize = standardize
        self.timeout = timeout
        self.protein_embedding_dim = protein_embedding_dim
        self.ligand_language_embedding = ligand_language_embedding
        self.batch_groups = list(
            zip(frame["task_index"].tolist(), frame["has_ligand"].tolist(), frame["is_binary"].astype(bool).tolist())
        )

    def len(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int | tuple[int, int]):
        if isinstance(index, tuple):
            return self.get(index)
        return super().__getitem__(index)

    def get(self, index: int | tuple[int, int]) -> dict[str, MultiGraphData | None | bool]:
        index, ranking_role = index if isinstance(index, tuple) else (index, 0)
        row = self.frame.iloc[index]
        has_ligand = bool(row.has_ligand)
        molecule = self._molecule_graph(str(row.Ligand)) if has_ligand else None
        primary = self._graph(
            protein=self._protein_graph(str(row.Protein)),
            molecule=molecule,
            row=row,
            protein_key=str(row.Protein),
            ranking_role=ranking_role,
        )
        reference = None
        if int(row.task_index) == MUTATION:
            reference = self._graph(
                protein=self._protein_graph(str(row.WT_Protein)),
                molecule=molecule,
                row=row,
                protein_key=str(row.WT_Protein),
                ranking_role=ranking_role,
            )
        return {"primary": primary, "reference": reference, "protein_only": not has_ligand}

    def _protein_graph(self, key: str) -> dict:
        if key in self.prot_dict:
            return self.prot_dict[key]
        path = self.protein_folder / f"{key}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing protein cache: {path}")
        protein = torch.load(path, map_location="cpu")
        dimension = protein["token_representation"].shape[1]
        if self.protein_embedding_dim is None:
            self.protein_embedding_dim = dimension
        if dimension != self.protein_embedding_dim:
            raise ValueError(
                f"Protein cache {path} has dimension {dimension}, "
                f"expected {self.protein_embedding_dim}"
            )
        return protein

    def _molecule_graph(self, smiles: str) -> dict:
        if smiles in self.mol_dict:
            return self.mol_dict[smiles]
        path = self.molecule_folder / f"{hashlib.sha256(smiles.encode()).hexdigest()[:32]}.pt"
        if path.exists():
            return torch.load(path, map_location="cpu")
        normalizer = MoleculeNormalizer(timeout_seconds=self.timeout)
        result = normalizer.standardize_normalize(smiles) if self.standardize else normalizer.normalize(smiles)
        if result.smiles is None:
            raise ValueError(f"Invalid ligand SMILES: {smiles}")
        molecule = smiles_to_graph(result.smiles)
        if molecule is None:
            raise ValueError(f"Could not graph ligand SMILES: {smiles}")
        self.molecule_folder.mkdir(parents=True, exist_ok=True)
        torch.save(molecule, path)
        return molecule

    def _graph(
        self,
        protein: dict,
        molecule: dict | None,
        row: pd.Series,
        protein_key: str,
        ranking_role: int,
    ) -> MultiGraphData:
        fields = {
            "prot_node_aa": protein["seq_feat"].float(),
            "prot_node_evo": protein["token_representation"].float(),
            "prot_seq": protein["seq"],
            "prot_edge_index": protein["edge_index"].long(),
            "prot_edge_weight": protein["edge_weight"].float(),
            "prot_num_nodes": protein["num_nodes"],
            "id": str(row.ID),
            "prot_key": protein_key,
            "cluster_id": str(row.cluster_id),
            "task_index": torch.tensor([int(row.task_index)], dtype=torch.long),
            "label": torch.tensor([float(row.label)], dtype=torch.float),
            "is_binary": torch.tensor([bool(row.is_binary)], dtype=torch.bool),
            "sample_weight": torch.tensor([float(row.sample_weight)], dtype=torch.float),
            "protein_group_index": torch.tensor([int(row.protein_group_index)], dtype=torch.long),
            "ligand_group_index": torch.tensor([int(row.ligand_group_index)], dtype=torch.long),
            "ranking_role": torch.tensor([ranking_role], dtype=torch.long),
            "source": str(row.source),
        }
        if molecule is not None:
            fields.update(
                mol_x=molecule["atom_idx"],
                mol_x_feat=molecule["atom_feature"],
                mol_edge_index=molecule["atom_edge_index"],
                mol_edge_attr=molecule["atom_edge_attr"],
                mol_num_nodes=molecule["atom_num_nodes"],
                clique_x=molecule["x_clique"],
                clique_x_pe=molecule.get("clique_pe"),
                clique_edge_index=molecule["tree_edge_index"],
                atom2clique_index=molecule["atom2clique_index"],
                clique_num_nodes=molecule["clique_num_nodes"],
                mol_global_fingerprint=molecule["global_fingerprint"].float(),
                mol_key=str(row.Ligand),
            )
            if self.ligand_language_embedding:
                fields["mol_language_embedding"] = molecule["molt5_representation"].float()
        return MultiGraphData(**fields)


class MultiGraphData(Data):
    def __inc__(self, key, item, *args):
        if key in {"task_index", "protein_group_index", "ligand_group_index", "ranking_role"}:
            return 0
        if key == "mol_edge_index":
            return self.mol_x.size(0)
        if key == "clique_edge_index":
            return self.clique_x.size(0)
        if key == "atom2clique_index":
            return torch.tensor([[self.mol_x.size(0)], [self.clique_x.size(0)]])
        if key == "prot_edge_index":
            return self.prot_node_aa.size(0)
        return super().__inc__(key, item, *args)
