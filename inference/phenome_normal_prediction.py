from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from models.enzyme_prediction import TASK_TO_INDEX, task_distribution

PRODUCTION_TASKS = frozenset({"kcat", "km"})


def prepare_inference_frame(frame: pd.DataFrame, task: str) -> pd.DataFrame:
    if task not in PRODUCTION_TASKS:
        raise ValueError(f"Phenom supports only {sorted(PRODUCTION_TASKS)}, got {task!r}")
    required = ["Protein", "Ligand"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Missing columns for {task}: {missing}")

    result = frame.copy()
    if "ID" not in result:
        result["ID"] = [f"Row_{index + 1}" for index in range(len(result))]
    result["task_index"] = TASK_TO_INDEX[task]
    result["label"] = 0.0
    result["is_binary"] = False
    result["sample_weight"] = 1.0
    result["source"] = "inference"
    result["cluster_id"] = result["Protein"].astype(str)
    return result


def format_prediction(
    task: str,
    mu: torch.Tensor,
    sigma: torch.Tensor,
) -> dict[str, np.ndarray]:
    if task not in PRODUCTION_TASKS:
        raise ValueError(f"Phenom supports only {sorted(PRODUCTION_TASKS)}, got {task!r}")
    selected_mu, selected_sigma = task_distribution(mu, sigma, TASK_TO_INDEX[task])
    return _kinetic_columns(selected_mu, selected_sigma)


def _kinetic_columns(mu: torch.Tensor, sigma: torch.Tensor) -> dict[str, np.ndarray]:
    nominal_factor = torch.pow(10.0, 1.96 * sigma)
    return {
        "mu_log10": _numpy(mu),
        "sigma_log10": _numpy(sigma),
        "prediction": _numpy(torch.pow(10.0, mu)),
        # These are Gaussian-assumption ranges, not calibrated confidence
        # intervals.  The explicit name prevents downstream users from
        # interpreting a large value as an activity prediction.
        "nominal_lower_95": _numpy(torch.pow(10.0, mu - 1.96 * sigma)),
        "nominal_upper_95": _numpy(torch.pow(10.0, mu + 1.96 * sigma)),
        "nominal_factor_95": _numpy(nominal_factor),
    }


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()
