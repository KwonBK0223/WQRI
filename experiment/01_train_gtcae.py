"""Train GTCAE over multiple temporal window sizes.

The graph datasets are built in memory from chunked CSV files. No train/validation/test
CSV files and no graph_dataset .pt files are written to disk.
"""

from __future__ import annotations

import argparse

from wqri.graph_builder import build_graph_splits_from_files
from wqri.train import ExperimentConfig, train_window_sweep
from wqri.utils import load_yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Train GTCAE for WQRI estimation.")
    parser.add_argument("--config", type=str, default="experiment/config.yaml", help="YAML configuration file.")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing final_data01.csv to final_data20.csv.")
    parser.add_argument("--chunk_pattern", type=str, default="final_data*.csv", help="Glob pattern for chunked water-quality CSV files.")
    parser.add_argument("--expected_parts", type=int, default=20, help="Expected number of chunked CSV files.")
    parser.add_argument("--edge_csv", type=str, default="data/small_edge.csv", help="Path to the hydrological adjacency CSV.")
    parser.add_argument("--water_encoding", type=str, default="utf-8", help="Encoding used for the water-quality CSV chunks.")
    parser.add_argument("--edge_encoding", type=str, default="utf-8", help="Encoding used for the edge CSV.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg_dict = load_yaml(args.config)
    cfg = ExperimentConfig(**cfg_dict)

    graph_splits = build_graph_splits_from_files(
        data_dir=args.data_dir,
        edge_csv=args.edge_csv,
        chunk_pattern=args.chunk_pattern,
        expected_parts=args.expected_parts,
        water_encoding=args.water_encoding,
        edge_encoding=args.edge_encoding,
        splits=("train", "val", "test"),
        verbose=True,
    )
    train_window_sweep(cfg, graph_splits=graph_splits)


if __name__ == "__main__":
    main()
