"""Checkpoint-compatible 33-dimensional residue features for ESMC caches.

ESMC embeddings and contacts are built in ``train/build_esmc_cache.py``. This
module contains only the handcrafted residue features used alongside them.
"""

from __future__ import annotations

import numpy as np


pro_res_table = list("ACDEFGHIKLMNPQRSTVWYX")
_ALIPHATIC = set("AILMV")
_AROMATIC = set("FWY")
_POLAR_NEUTRAL = set("CNQST")
_ACIDIC = set("DE")
_BASIC = set("HKR")


def _normalize(table: dict[str, float]) -> dict[str, float]:
    """Preserve the historical cache feature scale, including its X value."""
    high, low = max(table.values()), min(table.values())
    normalized = {key: (value - low) / (high - low) for key, value in table.items()}
    normalized["X"] = (high + low) / 2.0
    return normalized


_WEIGHT = _normalize(dict(zip(
    "ACDEFGHIKLMNPQRSTVWY",
    [71.08, 103.15, 115.09, 129.12, 147.18, 57.05, 137.14, 113.16, 128.18,
     113.16, 131.20, 114.11, 97.12, 128.13, 156.19, 87.08, 101.11, 99.13,
     186.22, 163.18],
)))
_PKA = _normalize(dict(zip(
    "ACDEFGHIKLMNPQRSTVWY",
    [2.34, 1.96, 1.88, 2.19, 1.83, 2.34, 1.82, 2.36, 2.18, 2.36,
     2.28, 2.02, 1.99, 2.17, 2.17, 2.21, 2.09, 2.32, 2.83, 2.32],
)))
_PKB = _normalize(dict(zip(
    "ACDEFGHIKLMNPQRSTVWY",
    [9.69, 10.28, 9.60, 9.67, 9.13, 9.60, 9.17, 9.60, 8.95, 9.60,
     9.21, 8.80, 10.60, 9.13, 9.04, 9.15, 9.10, 9.62, 9.39, 9.62],
)))
_PKX = _normalize(dict(zip(
    "ACDEFGHIKLMNPQRSTVWY",
    [0.00, 8.18, 3.65, 4.25, 0.00, 0.00, 6.00, 0.00, 10.53, 0.00,
     0.00, 0.00, 0.00, 0.00, 12.48, 0.00, 0.00, 0.00, 0.00, 0.00],
)))
_PL = _normalize(dict(zip(
    "ACDEFGHIKLMNPQRSTVWY",
    [6.00, 5.07, 2.77, 3.22, 5.48, 5.97, 7.59, 6.02, 9.74, 5.98,
     5.74, 5.41, 6.30, 5.65, 10.76, 5.68, 5.60, 5.96, 5.89, 5.96],
)))
_HYDROPHOBIC_PH2 = _normalize(dict(zip(
    "ACDEFGHIKLMNPQRSTVWY",
    [47, 52, -18, 8, 92, 0, -42, 100, -37, 100,
     74, -41, -46, -18, -26, -7, 13, 79, 84, 49],
)))
_HYDROPHOBIC_PH7 = _normalize(dict(zip(
    "ACDEFGHIKLMNPQRSTVWY",
    [41, 49, -55, -31, 100, 0, 8, 99, -23, 97,
     74, -28, -46, -10, -14, -5, 13, 76, 97, 63],
)))
_PROPERTIES = (
    _WEIGHT, _PKA, _PKB, _PKX, _PL, _HYDROPHOBIC_PH2, _HYDROPHOBIC_PH7,
)


def replace_non_standard_residues(sequence: str, res_table=pro_res_table) -> str:
    allowed = set(res_table)
    return "".join(residue if residue in allowed else "X" for residue in sequence)


def seq_feature(sequence: str) -> np.ndarray:
    """Return the exact 21 one-hot + 12 physicochemical features per residue."""
    sequence = sequence.replace("U", "X").replace("B", "X")
    result = np.zeros((len(sequence), 33), dtype=np.float64)
    for index, residue in enumerate(sequence):
        result[index, pro_res_table.index(residue)] = 1.0
        result[index, 21:26] = (
            residue in _ALIPHATIC,
            residue in _AROMATIC,
            residue in _POLAR_NEUTRAL,
            residue in _ACIDIC,
            residue in _BASIC,
        )
        result[index, 26:] = [table[residue] for table in _PROPERTIES]
    return result
