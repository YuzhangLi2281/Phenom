from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator, Sequence

import torch
from torch_geometric.data import Batch


FOLLOW_BATCH = ["mol_x", "clique_x", "prot_node_aa"]


class TaskBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Build homogeneous batches so protein-only and paired graphs never mix."""

    def __init__(self, groups: Sequence[tuple[int, bool, bool]], batch_size: int, shuffle: bool) -> None:
        self.groups = groups
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[list[int]]:
        grouped: dict[tuple[int, bool, bool], list[int]] = defaultdict(list)
        for index, group in enumerate(self.groups):
            grouped[group].append(index)
        batches = []
        for indices in grouped.values():
            if self.shuffle:
                random.shuffle(indices)
            batches.extend(indices[start : start + self.batch_size] for start in range(0, len(indices), self.batch_size))
        if self.shuffle:
            random.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        counts: dict[tuple[int, bool, bool], int] = defaultdict(int)
        for group in self.groups:
            counts[group] += 1
        return sum((count + self.batch_size - 1) // self.batch_size for count in counts.values())


def collate_enzyme_samples(samples: list[dict]) -> dict:
    protein_only = samples[0]["protein_only"]
    if any(sample["protein_only"] != protein_only for sample in samples):
        raise ValueError("A PHENOME batch must share one ligand mode")
    primary = Batch.from_data_list(
        [sample["primary"] for sample in samples],
        follow_batch=["prot_node_aa"] if protein_only else FOLLOW_BATCH,
    )
    references = [sample["reference"] for sample in samples]
    reference = None
    if references[0] is not None:
        reference = Batch.from_data_list(
            references,
            follow_batch=["prot_node_aa"] if protein_only else FOLLOW_BATCH,
        )
    return {"primary": primary, "reference": reference, "protein_only": protein_only}


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size: int = 1, shuffle: bool = False, **kwargs) -> None:
        kwargs.pop("follow_batch", None)
        kwargs.pop("sampler", None)
        batch_sampler = TaskBatchSampler(dataset.batch_groups, batch_size, shuffle)
        super().__init__(dataset, batch_sampler=batch_sampler, collate_fn=collate_enzyme_samples, **kwargs)
