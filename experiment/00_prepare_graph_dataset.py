"""Check chunked input data and build in-memory graph splits once.

This script does not save train/validation/test CSV files or graph_dataset .pt files.
It is intended as a lightweight sanity check before training.
"""

from __future__ import annotations

import argparse

from wqri.graph_builder import build_graph_splits_from_files


def parse_args():
    parser = argparse.ArgumentParser(description="Check chunked CSV files and build in-memory STWG graph splits.")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing final_data01.csv to final_data20.csv.")
    parser.add_argument("--chunk_pattern", type=str, default="final_data*.csv", help="Glob pattern for chunked water-quality CSV files.")
    parser.add_argument("--expected_parts", type=int, default=20, help="Expected number of chunked CSV files.")
    parser.add_argument("--edge_csv", type=str, default="data/edge.csv", help="Path to the hydrological adjacency CSV.")
    parser.add_argument("--water_encoding", type=str, default="utf-8", help="Encoding used for the water-quality CSV chunks.")
    parser.add_argument("--edge_encoding", type=str, default="utf-8", help="Encoding used for the edge CSV.")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed graph summaries.")
    return parser.parse_args()


def main():
    args = parse_args()
    build_graph_splits_from_files(
        data_dir=args.data_dir,
        edge_csv=args.edge_csv,
        chunk_pattern=args.chunk_pattern,
        expected_parts=args.expected_parts,
        water_encoding=args.water_encoding,
        edge_encoding=args.edge_encoding,
        verbose=not args.quiet,
    )
    print("[Done] Input chunks and in-memory graph construction are valid.")


if __name__ == "__main__":
    main()
