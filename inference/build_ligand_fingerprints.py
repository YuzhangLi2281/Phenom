#!/usr/bin/env python3
"""Add cached structural fingerprints and optional MolT5 embeddings to ligand graphs."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator

CODEBASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODEBASE_DIR))
from mol_utils import MoleculeNormalizer, smiles_to_graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, nargs="+", required=True)
    parser.add_argument("--ligand-cache-dir", type=Path, required=True)
    parser.add_argument("--molt5-model", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.ligand_cache_dir.mkdir(parents=True, exist_ok=True)
    ligands = pd.concat([pd.read_csv(path, usecols=["Ligand"], low_memory=False) for path in args.csv])
    smiles_values = sorted(ligands["Ligand"].dropna().astype(str).unique())
    morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    normalizer = MoleculeNormalizer(timeout_seconds=10)
    tokenizer = model = device = None
    if not args.molt5_model:
        smiles_values = [
            smiles for smiles in smiles_values
            if not (args.ligand_cache_dir / f"{hashlib.sha256(smiles.encode()).hexdigest()[:32]}.pt").exists()
        ]
    if args.molt5_model:
        from transformers import T5EncoderModel, T5Tokenizer

        tokenizer = T5Tokenizer.from_pretrained(args.molt5_model)
        model = T5EncoderModel.from_pretrained(args.molt5_model).eval()
        device = torch.device(args.device)
        model.to(device)
        pending = []
        for smiles in smiles_values:
            path = args.ligand_cache_dir / f"{hashlib.sha256(smiles.encode()).hexdigest()[:32]}.pt"
            if not path.exists() or "molt5_representation" not in torch.load(path, map_location="cpu"):
                pending.append(smiles)
        smiles_values = pending

    for start in range(0, len(smiles_values), args.batch_size):
        batch_smiles = smiles_values[start:start + args.batch_size]
        graphs = []
        paths = []
        normalized_smiles = []
        for smiles in batch_smiles:
            path = args.ligand_cache_dir / f"{hashlib.sha256(smiles.encode()).hexdigest()[:32]}.pt"
            graph = torch.load(path, map_location="cpu") if path.exists() else smiles_to_graph(
                normalizer.standardize_normalize(smiles).smiles
            )
            molecule = Chem.MolFromSmiles(graph["smiles"])
            maccs_array = np.zeros(167, dtype=np.float32)
            morgan_array = np.zeros(2048, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(molecule), maccs_array)
            DataStructs.ConvertToNumpyArray(morgan.GetFingerprint(molecule), morgan_array)
            graph["global_fingerprint"] = torch.from_numpy(
                np.concatenate([maccs_array, morgan_array])
            ).view(1, -1)
            graphs.append(graph)
            paths.append(path)
            normalized_smiles.append(graph["smiles"])

        if model is not None:
            tokens = tokenizer(
                normalized_smiles, return_tensors="pt", padding=True, truncation=True
            ).to(device)
            mixed_precision = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda" else nullcontext()
            )
            with torch.inference_mode(), mixed_precision:
                hidden = model(**tokens).last_hidden_state
            mask = tokens.attention_mask.bool()
            mask[torch.arange(mask.size(0), device=device), mask.sum(dim=1) - 1] = False
            embeddings = (
                (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
            ).float().cpu()
            for graph, embedding in zip(graphs, embeddings):
                graph["molt5_representation"] = embedding.view(1, -1)

        for graph, path in zip(graphs, paths):
            temporary = path.with_suffix(".pt.tmp")
            torch.save(graph, temporary)
            os.replace(temporary, path)
        print(f"ligands={min(start + args.batch_size, len(smiles_values))}/{len(smiles_values)}", flush=True)


if __name__ == "__main__":
    main()
