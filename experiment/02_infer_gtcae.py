"""Run inference from saved GTCAE checkpoints.

The graph datasets are rebuilt in memory from chunked CSV files. This avoids storing
large intermediate graph_dataset .pt files in the repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from wqri.constants import DEFAULT_WINDOW_LIST
from wqri.graph_builder import build_graph_splits_from_files
from wqri.inference import find_checkpoint, load_model_from_checkpoint, run_inference
from wqri.utils import clear_memory, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Extract latent representations from saved GTCAE checkpoints.")
    parser.add_argument("--version", type=str, default="904", help="Experiment version string used in checkpoint names.")
    parser.add_argument("--models_dir", type=str, default="experiment/models", help="Directory containing saved model checkpoints.")
    parser.add_argument("--result_dir", type=str, default="experiment/results_inference_only", help="Directory for inference results.")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing final_data01.csv to final_data20.csv.")
    parser.add_argument("--chunk_pattern", type=str, default="final_data*.csv", help="Glob pattern for chunked water-quality CSV files.")
    parser.add_argument("--expected_parts", type=int, default=20, help="Expected number of chunked CSV files.")
    parser.add_argument("--edge_csv", type=str, default="data/small_edge.csv", help="Path to the hydrological adjacency CSV.")
    parser.add_argument("--water_encoding", type=str, default="utf-8", help="Encoding used for the water-quality CSV chunks.")
    parser.add_argument("--edge_encoding", type=str, default="utf-8", help="Encoding used for the edge CSV.")
    parser.add_argument("--windows", nargs="+", type=int, default=DEFAULT_WINDOW_LIST, help="Window sizes to process.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for inference.")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(42)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    graph_splits = build_graph_splits_from_files(
        data_dir=args.data_dir,
        edge_csv=args.edge_csv,
        chunk_pattern=args.chunk_pattern,
        expected_parts=args.expected_parts,
        water_encoding=args.water_encoding,
        edge_encoding=args.edge_encoding,
        splits=("train", "val", "test"),
        verbose=False,
    )

    summary = []
    for window_size in args.windows:
        try:
            ckpt_path = find_checkpoint(args.models_dir, args.version, window_size)
            model, checkpoint = load_model_from_checkpoint(ckpt_path, device)
            out_prefix = result_dir / checkpoint["tag"]
            latent_dim = checkpoint["config"]["latent_dim"]

            for split_name in ["train", "val", "test"]:
                split_graphs = graph_splits[split_name]
                run_inference(
                    split_name,
                    split_graphs["main_graphs"],
                    split_graphs["risk_graphs"],
                    split_graphs["main_node_maps"],
                    model,
                    checkpoint["window_size"],
                    latent_dim,
                    device,
                    out_prefix,
                )

            summary.append({"tag": checkpoint["tag"], "window_size": checkpoint["window_size"], "ckpt_path": ckpt_path})
            del model, checkpoint
        except Exception as exc:
            print(f"[Skip] window={window_size} failed: {exc}")
        clear_memory()

    df_summary = pd.DataFrame(summary)
    summary_path = result_dir / f"v{args.version}_inference_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"[Done] Saved summary -> {summary_path}")


if __name__ == "__main__":
    main()
