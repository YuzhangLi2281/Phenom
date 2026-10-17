# PHENOM

![PHENOM enzyme–substrate prediction architecture](assets/PHENOM_Main_Architecture.svg)

**PH**ysicochemical knowledge enhanced network for **EN**zyme **O**pti**M**ization.

**Sequence-based enzyme–substrate kinetic prediction with interpretable interaction fingerprints and native Gaussian uncertainty heads.**

[Enzora workspace](https://enzora-agent.com/) · [Enzora software](https://github.com/YuzhangLi2281/Enzora)

Phenom combines a protein sequence with a substrate molecular graph to predict `log10(kcat)` or `log10(Km)`. Hierarchical graph networks construct residue–pocket and atom–motif representations. Bidirectional interactions connect these representations into an enzyme–substrate fingerprint. A Gaussian head returns a prediction mean and standard deviation for each released task.

## Architecture

| Component | Released configuration |
| --- | --- |
| Protein language representation | ESMC-600M, 1,152 features per residue |
| Protein graph | Final four ESMC attention layers; symmetric APC correction; top-16 neighbours |
| Residue features | 33 amino-acid physicochemical features |
| Substrate graph | Standardized SMILES atom/bond graph and motif graph |
| Molecular descriptors | MACCS-167, Morgan-2048 and MolT5-768 |
| Graph backbone | Three PNA layers; 200 hidden channels |
| Protein pooling | 5, 10 and 20 learned pockets |
| Interaction | Five-head pocket–motif attention and ligand-guided residue–atom interactions |
| Kinetic outputs | Separate Gaussian mean and sigma for kcat and Km |

Attention and pooling weights support inspection of residues and chemical fragments. They describe model behaviour; experimental studies establish catalytic mechanisms. Native sigma expresses the model's predictive spread and has not been calibrated as an experimental confidence interval.

The production interface accepts canonical protein sequences of 1–1,000 residues and valid substrate SMILES. Longer proteins require an explicit region choice before preprocessing. The checkpoint retains its internal tensor layout, while the public prediction interface exposes only kcat and Km.

## Release contents

```text
models/                 Phenom architecture and interaction layers
mol_utils/              Molecule normalization and graph construction
utils/                  Sequence features, dataset and batch assembly
inference/              CLI and Python prediction interfaces
inference/build_esmc_cache.py
inference/build_ligand_fingerprints.py
                        Inference feature-cache builders
ckpts/model.pt          Released Phenom parameters
ckpts/config.json       Architecture and target scalers
ckpts/degree.pt         PNA degree histograms
ckpts/selection.json    Checkpoint selection metadata
```

This release contains inference code and feature preprocessing. Model training scripts, training datasets and intermediate checkpoints are excluded. The included Phenom parameters contain 13,232,213 values; loading checks all 569 state-dictionary keys strictly.

ESMC and MolT5 weights are separate upstream assets. Download them on the inference server under their original terms. The graph builder requires a separately installed ESMC implementation with the matching attention-output contract.

## Installation

Create a dedicated Python environment with PyTorch and PyTorch Geometric matching your CUDA installation. Install `torch-scatter` for the same PyTorch/CUDA versions. The remaining inference dependencies include pandas, NumPy, SciPy, RDKit, Biopython, Transformers, SentencePiece, safetensors, einops, huggingface-hub and chembl_structure_pipeline.

```bash
git clone https://github.com/YuzhangLi2281/Phenom.git
cd Phenom
export PHENOM_ESM_SOURCE=/absolute/path/to/separately/installed/esm-source
```

The `environment.yml` file provides an initial environment specification. Check the installed PyTorch/PyG binaries against your GPU driver before building caches.

## Prepare input and feature caches

Create an enzyme–substrate CSV:

```csv
ID,Protein,Ligand
example_1,enzyme_1,O=C=O
```

`Protein` is a stable identifier, not the amino-acid sequence. Provide the corresponding sequence map:

```csv
Protein,Sequence
enzyme_1,ACDEFGHIKLMNPQRSTVWY
```

Build the protein cache using a local ESMC-600M model directory:

```bash
python inference/build_esmc_cache.py \
  --protein-sequence-map-csv sequences.csv \
  --protein-cache-dir cache/proteins \
  --esmc-model-dir /path/to/esmc-600m \
  --device cuda:0
```

Build molecular graphs, structural fingerprints and MolT5 embeddings:

```bash
python inference/build_ligand_fingerprints.py \
  --csv input.csv \
  --ligand-cache-dir cache/ligands \
  --molt5-model /path/to/molt5-base-smiles2caption \
  --device cuda:0
```

Protein cache filenames use the supplied protein identifier. Ligand cache filenames use a SHA-256-derived key from the input SMILES. Keep these caches consistent with the input table and preprocessing configuration.

## Predict

```bash
python inference/inference.py \
  --input-file input.csv \
  --output-folder results \
  --model-dir ckpts \
  --protein-cache-dir cache/proteins \
  --ligand-cache-dir cache/ligands \
  --task kcat \
  --device cuda:0 \
  --batch-size 32 \
  --num-workers 0
```

Use `--task km` for Km prediction. The command writes `results/inference_results.csv` and preserves the input columns.

| Output column | Meaning |
| --- | --- |
| `mu_log10` | Predicted kinetic value in log10 units |
| `sigma_log10` | Native Gaussian standard deviation in log10 units |
| `prediction` | `10 ** mu_log10` |
| `nominal_lower_95`, `nominal_upper_95` | Gaussian-assumption range transformed to linear units |
| `nominal_factor_95` | Multiplicative factor for that nominal range |

The checkpoint metadata defines target units and scaling. Use log10 predictions for comparisons; do not substitute sigma for activity.

## Python interface

Run from the repository root:

```python
from pathlib import Path
import pandas as pd
from inference.inference import load_model, predict_frame

model = load_model(Path("ckpts"), device="cpu")
result = predict_frame(
    pd.read_csv("input.csv"),
    task="kcat",
    model=model,
    protein_cache_dir=Path("cache/proteins"),
    ligand_cache_dir=Path("cache/ligands"),
    device="cpu",
    batch_size=8,
    num_workers=0,
)
result.to_csv("predictions.csv", index=False)
```

Enzora uses this release as its activity scorer and carries `mu_log10` and `sigma_log10` into reference-based ranking and evidence-informed score balancing.

## References and attribution

- [PSICHIC](https://github.com/huankoh/PSICHIC), Koh et al.: sequence-based hierarchical protein–ligand interaction fingerprints. Phenom adapts graph, pooling and interaction utilities from this work. Those portions retain Apache-2.0 terms; see `PSICHIC-Apache-2.0.txt` and `THIRD_PARTY_NOTICES.md`.
- [MolT5](https://github.com/blender-nlp/MolT5): molecular language embeddings; [model weights](https://huggingface.co/laituan245/molt5-base-smiles2caption).
- [Principal Neighbourhood Aggregation](https://github.com/lukecavabarrett/pna), [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) and [RDKit](https://github.com/rdkit/rdkit): graph modelling and molecular representations.
- [ChEMBL Structure Pipeline](https://github.com/chembl/ChEMBL_Structure_Pipeline): molecular standardization.

## License

Original Phenom additions and released Phenom parameters are provided under the MIT License. Adapted and vendored components retain their upstream licences. See `LICENSE`, `THIRD_PARTY_NOTICES.md` for the applicable notices.

ESMC source is an external runtime dependency and is not distributed here. Set `PHENOM_ESM_SOURCE` to the matching installation providing `esm.models.esmc.EsmcModel`. Keep its environment consistent with the pretrained ESMC files used for preprocessing.
