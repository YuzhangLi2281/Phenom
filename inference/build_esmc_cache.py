from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.utils import add_self_loops, coalesce, degree, remove_self_loops, to_undirected
from transformers import PreTrainedTokenizerFast

SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_DIR = SCRIPT_DIR.parent
WORKSPACE_DIR = CODEBASE_DIR.parent
ESM_CODE_DIR = Path(os.environ.get("PHENOM_ESM_SOURCE", WORKSPACE_DIR / "esm"))
sys.path.insert(0, str(CODEBASE_DIR))
sys.path.insert(0, str(ESM_CODE_DIR))

from esm.models.esmc import EsmcModel
from utils.protein_init import pro_res_table, replace_non_standard_residues, seq_feature


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phenom protein graphs from ESMC embeddings and attention.")
    parser.add_argument("--protein-sequence-map-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--protein-map-key-column", default="Protein")
    parser.add_argument("--protein-map-sequence-column", default="Sequence")
    parser.add_argument("--protein-cache-dir", type=Path, required=True)
    parser.add_argument("--degree-output", type=Path)
    parser.add_argument("--esmc-model-dir", type=Path, required=True)
    parser.add_argument("--filter-csv", type=Path, action="append", default=[])
    parser.add_argument("--filter-key-column", default="Protein")
    parser.add_argument("--contact-layers", type=int, default=4)
    parser.add_argument("--contact-neighbors", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.protein_cache_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.concat(
        [
            pd.read_csv(path, usecols=[args.protein_map_key_column, args.protein_map_sequence_column])
            for path in args.protein_sequence_map_csv
        ],
        ignore_index=True,
    ).dropna().drop_duplicates(args.protein_map_key_column)
    frame[args.protein_map_key_column] = frame[args.protein_map_key_column].astype(str)
    frame[args.protein_map_sequence_column] = frame[args.protein_map_sequence_column].astype(str)

    if args.filter_csv:
        requested = set()
        for path in args.filter_csv:
            requested.update(pd.read_csv(path, usecols=[args.filter_key_column])[args.filter_key_column].astype(str))
        missing = sorted(requested - set(frame[args.protein_map_key_column]))
        if missing:
            raise ValueError(f"Protein sequence map is missing {len(missing)} IDs: {missing[:10]}")
        frame = frame[frame[args.protein_map_key_column].isin(requested)]

    rows = [
        (key, sequence)
        for key, sequence in frame[[args.protein_map_key_column, args.protein_map_sequence_column]].itertuples(index=False)
        if args.overwrite or not (args.protein_cache_dir / f"{key}.pt").exists()
    ]
    if args.limit is not None:
        rows = rows[: args.limit]
    for key, sequence in rows:
        if not sequence or len(sequence) > 1000 or set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"):
            raise ValueError(f"Protein {key} requires canonical amino acids and length <=1000; input is never truncated")
    rows.sort(key=lambda item: len(item[1]), reverse=True)
    print(f"protein_map_rows={len(frame)}")
    print(f"cache_files_to_build={len(rows)}")
    if not rows:
        if args.degree_output:
            protein_degree = torch.zeros(1, dtype=torch.long)
            for key in frame[args.protein_map_key_column]:
                graph = torch.load(args.protein_cache_dir / f"{key}.pt", map_location="cpu")
                graph_degree = degree(graph["edge_index"][1], num_nodes=graph["num_nodes"], dtype=torch.long)
                counts = torch.bincount(graph_degree)
                protein_degree = torch.nn.functional.pad(
                    protein_degree, (0, max(0, counts.numel() - protein_degree.numel()))
                )
                protein_degree[: counts.numel()] += counts
            args.degree_output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"protein_deg": protein_degree}, args.degree_output)
            print(f"degree_output={args.degree_output} nodes={int(protein_degree.sum())}")
        return

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(args.esmc_model_dir / "tokenizer.json"),
        cls_token="<cls>",
        eos_token="<eos>",
        pad_token="<pad>",
        mask_token="<mask>",
        unk_token="<unk>",
    )
    model = EsmcModel.from_pretrained(
        args.esmc_model_dir,
        device=args.device,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    encoder_name = args.esmc_model_dir.name
    attention_blocks = model.transformer.blocks[-args.contact_layers :]

    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start : start + args.batch_size]
            keys = [key for key, _ in batch_rows]
            sequences = [sequence for _, sequence in batch_rows]
            lengths = [len(sequence) for sequence in sequences]
            maximum_length = max(lengths)

            tokens = tokenizer(sequences, add_special_tokens=True, padding=True, return_tensors="pt")
            input_ids = tokens["input_ids"].to(args.device)
            attention_mask = tokens["attention_mask"].to(args.device)
            block_inputs: list[tuple[torch.Tensor, torch.Tensor | None]] = []
            handles = [
                block.attn.register_forward_pre_hook(
                    lambda _module, inputs: block_inputs.append((inputs[0].detach(), inputs[1]))
                )
                for block in attention_blocks
            ]
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            for handle in handles:
                handle.remove()

            contact_scores = torch.zeros(
                (len(sequences), maximum_length, maximum_length),
                device=args.device,
                dtype=torch.float32,
            )
            for block, (hidden, sequence_id) in zip(attention_blocks, block_inputs, strict=True):
                _, attention = block.attn(hidden, sequence_id, output_attentions=True)
                contact_scores += attention[:, :, 1 : maximum_length + 1, 1 : maximum_length + 1].float().mean(1)
            contact_scores /= len(attention_blocks)

            for offset, (key, sequence, length) in enumerate(zip(keys, sequences, lengths, strict=True)):
                token_representation = output.last_hidden_state[offset, 1 : length + 1].cpu()
                contact_score = contact_scores[offset, :length, :length]
                contact_score = 0.5 * (contact_score + contact_score.T)
                row_sum = contact_score.sum(1)
                contact_score = (contact_score - torch.outer(row_sum, row_sum) / contact_score.sum()).clamp_min(0)
                contact_score.fill_diagonal_(0)
                contact_score /= contact_score.max().clamp_min(torch.finfo(contact_score.dtype).eps)

                neighbors = min(args.contact_neighbors, max(1, length - 1))
                values, columns = contact_score.topk(neighbors, dim=1)
                rows_index = torch.arange(length, device=args.device).repeat_interleave(neighbors)
                edge_index = torch.stack((rows_index, columns.reshape(-1)))
                edge_weight = values.reshape(-1)
                if length > 1:
                    backbone = torch.arange(length - 1, device=args.device)
                    backbone_edges = torch.cat(
                        (torch.stack((backbone, backbone + 1)), torch.stack((backbone + 1, backbone))), dim=1
                    )
                    edge_index = torch.cat((edge_index, backbone_edges), dim=1)
                    edge_weight = torch.cat((edge_weight, edge_weight.new_full((backbone_edges.size(1),), 0.5)))
                edge_index, edge_weight = coalesce(edge_index, edge_weight, reduce="max")
                edge_index, edge_weight = to_undirected(edge_index, edge_weight, reduce="max")
                edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
                edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=1.0)

                graph = {
                    "seq": sequence,
                    "seq_feat": torch.from_numpy(seq_feature(sequence)),
                    "token_representation": token_representation.to(torch.float16),
                    "num_nodes": length,
                    "num_pos": torch.arange(length).reshape(-1, 1),
                    "edge_index": edge_index.cpu(),
                    "edge_weight": edge_weight.cpu(),
                    "encoder": encoder_name,
                    "encoder_dimension": model.config.hidden_size,
                    "contact_encoder": f"{encoder_name} last-{args.contact_layers}-layer attention APC top-{args.contact_neighbors}",
                }
                output_path = args.protein_cache_dir / f"{key}.pt"
                temporary = output_path.with_suffix(".pt.tmp")
                torch.save(graph, temporary)
                os.replace(temporary, output_path)
                print(f"[{start + offset + 1}/{len(rows)}] cached {key} ({length} residues)", flush=True)


if __name__ == "__main__":
    main()
