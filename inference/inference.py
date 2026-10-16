from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODEBASE_DIR))

from inference.phenome_normal_prediction import format_prediction, prepare_inference_frame
from models.net import Phenom
from utils.dataloader import DataLoader
from utils.dataset import ProteinMoleculeDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phenom enzyme predictions.")
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-folder", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("ckpts"))
    parser.add_argument("--protein-cache-dir", type=Path, required=True)
    parser.add_argument("--ligand-cache-dir", type=Path, required=True)
    parser.add_argument("--task", choices=["kcat", "km"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.input_file)
    model = load_model(args.model_dir, args.device)
    result = predict_frame(
        source,
        task=args.task,
        model=model,
        protein_cache_dir=args.protein_cache_dir,
        ligand_cache_dir=args.ligand_cache_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    args.output_folder.mkdir(parents=True, exist_ok=True)
    output = args.output_folder / "inference_results.csv"
    result.to_csv(output, index=False)
    print(f"Saved Phenom predictions to {output}")


def predict_frame(
    source: pd.DataFrame,
    *,
    task: str,
    model: Phenom,
    protein_cache_dir: Path,
    ligand_cache_dir: Path,
    device: str = "cpu",
    batch_size: int = 32,
    num_workers: int = 0,
) -> pd.DataFrame:
    """Predict kcat or Km for a frame with Protein and Ligand columns.

    The caller supplies an already loaded final model and ESMC/MolT5 caches.
    Returned columns use the original log10 units; nominal ranges are not
    calibrated confidence intervals.
    """
    frame = prepare_inference_frame(source, task)
    if frame.empty:
        raise ValueError("Inference input must contain at least one enzyme–ligand pair")
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    ligand_cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = ProteinMoleculeDataset(
        frame,
        molecule_folder=ligand_cache_dir,
        protein_folder=protein_cache_dir,
        protein_embedding_dim=model.enzgfm_adapter.in_features,
        ligand_language_embedding=True,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    columns: dict[str, list] = {}
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            primary = batch["primary"].to(device)
            mu, sigma, _, _, _ = model(
                primary,
                protein_only=batch["protein_only"],
            )
            mu, sigma = _restore_units(mu, sigma, model.target_scalers)
            batch_columns = format_prediction(task, mu, sigma)
            for name, values in batch_columns.items():
                columns.setdefault(name, []).extend(values.tolist())

    result = source.copy()
    for name, values in columns.items():
        result[name] = values
    return result


def load_model(model_dir: Path, device: str = "cpu") -> Phenom:
    """Load only the released Phenom checkpoint with strict state matching."""
    with (model_dir / "config.json").open() as handle:
        config = json.load(handle)
    if config.get("model_class") != "Phenom":
        raise ValueError(f"Unsupported PHENOM model class: {config.get('model_class')}")
    if config["architecture"] != "Phenom":
        raise ValueError(f"Unsupported PHENOM architecture: {config['architecture']}")
    if config.get("task_names") != ["kcat", "km"]:
        raise ValueError("The released Phenom checkpoint must expose only kcat and Km")
    degree = torch.load(model_dir / "degree.pt", map_location=device)
    state = torch.load(model_dir / "model.pt", map_location=device)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    params = dict(config["params"])
    params["prot_evo_channels"] = state["enzgfm_adapter.weight"].shape[1]
    model = Phenom(
        degree["ligand_deg"], degree["protein_deg"],
        **{key: params[key] for key in (
            "mol_in_channels", "mol_edge_channels", "clique_pe_walk_length", "prot_in_channels",
            "prot_evo_channels", "hidden_channels", "pre_layers", "post_layers", "aggregators", "scalers",
            "total_layer", "K", "heads", "dropout", "dropout_attn_score",
            "prediction_head_layout",
        )},
        device=device,
    ).to(device)
    model.prediction_head.eps = params["sigma_epsilon"]
    model.load_state_dict(state)
    model.target_scalers = config["target_scalers"]
    return model


def _restore_units(mu: torch.Tensor, sigma: torch.Tensor, scalers: dict):
    centers = mu.new_zeros(mu.size(1))
    scales = mu.new_ones(mu.size(1))
    for scaler in scalers.values():
        task = int(scaler["task_index"])
        centers[task] = float(scaler["median"])
        scales[task] = float(scaler["iqr"])
    return mu * scales + centers, sigma * scales


if __name__ == "__main__":
    main()
