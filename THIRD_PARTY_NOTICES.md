# Third-party notices

Phenom adapts hierarchical graph, PNA, pooling, batching and molecule-processing utilities from [PSICHIC](https://github.com/huankoh/PSICHIC), by Koh and contributors. These adapted portions retain Apache-2.0 terms. The applicable text is in `LICENSES/PSICHIC-Apache-2.0.txt`.

The adapted source files include `models/layers.py`, `models/pna.py`, `models/scaler.py`, `models/pooling.py`, `models/interaction_layer.py`, `utils/dataset.py`, `utils/dataloader.py` and `mol_utils/`. Phenom changes the protein and molecular representations, catalytic interactions, prediction heads and production interface. Changes are identified by this notice dated 16 October 2026. Original additions and released Phenom parameters use the repository MIT licence.

`vendor/esm` contains the matching EvolutionaryScale ESM source implementation used for ESMC preprocessing. It retains upstream copyright and MIT terms, reproduced in `LICENSES/ESM-MIT.txt`. Upstream model weights have separate terms and are not included.

PyTorch Geometric and RDKit are external dependencies under their own licences. MolT5 source and pretrained assets are obtained from their original repositories. Consult the README references for sources.
