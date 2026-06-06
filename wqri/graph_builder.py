"""Graph construction utilities for the Spatio-Temporal Watershed Graph."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import pandas as pd
import torch
from torch_geometric.data import Data

from .constants import ENV_FEATURE_COLUMNS, RISK_FEATURE_COLUMNS
from .data import (
    filter_edges_by_locations,
    load_edges,
    load_water_quality_source,
    prepare_split_dataframe_from_raw,
)


def build_station_graph(df: pd.DataFrame, edge_list: pd.DataFrame, feature_columns: List[str]) -> nx.Graph:
    """Build a full spatial graph over station-time nodes for one feature view."""
    graph = nx.Graph()
    df_indexed = df.set_index("node_id")

    for node_id in df_indexed.index:
        graph.add_node(node_id, attr=df_indexed.loc[node_id, feature_columns].values)

    loc_to_nodes = {
        loc: group.sort_values("seq")["node_id"].tolist()
        for loc, group in df.groupby("Location")
    }

    for _, row in edge_list.iterrows():
        loc_u, loc_v = row["node1"], row["node2"]
        nodes_u, nodes_v = loc_to_nodes[loc_u], loc_to_nodes[loc_v]
        for node_u, node_v in zip(nodes_u, nodes_v):
            graph.add_edge(node_u, node_v)

    return graph


def summarize_graph(df: pd.DataFrame, edge_list: pd.DataFrame, graph: nx.Graph) -> pd.DataFrame:
    """Return a basic consistency summary for a constructed graph."""
    timepoints = int(df.groupby("Location").size().mode()[0])
    expected_nodes = df["node_id"].nunique()
    expected_edges = len(edge_list) * timepoints

    summary = pd.DataFrame(
        [
            ("Nodes", expected_nodes, graph.number_of_nodes()),
            ("Spatial unique undirected", expected_edges, graph.number_of_edges()),
        ],
        columns=["Type", "Expected", "Actual"],
    )
    summary["Check"] = summary["Expected"] == summary["Actual"]
    return summary


def build_sequence_graphs(
    df: pd.DataFrame,
    edge_list: pd.DataFrame,
    feature_columns: List[str],
) -> Tuple[List[Data], Dict[int, Dict[str, int]]]:
    """Build one PyTorch Geometric graph snapshot for each sequence index."""
    graphs = []
    mappings = {}
    seq_values = sorted(df["seq"].unique())

    for seq in seq_values:
        df_seq = df[df["seq"] == seq].sort_values("node_id")
        node_ids = df_seq["node_id"].tolist()
        x = torch.tensor(df_seq[feature_columns].values, dtype=torch.float)

        loc_to_node = {
            loc: group["node_id"].values[0]
            for loc, group in df_seq.groupby("Location")
        }
        node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
        mappings[int(seq)] = node_to_idx

        edges = []
        for _, row in edge_list.iterrows():
            loc_u, loc_v = row["node1"], row["node2"]
            node_u = loc_to_node.get(loc_u)
            node_v = loc_to_node.get(loc_v)
            if node_u in node_to_idx and node_v in node_to_idx:
                edges.append([node_to_idx[node_u], node_to_idx[node_v]])

        edge_index = (
            torch.tensor(edges, dtype=torch.long).t().contiguous()
            if edges
            else torch.empty((2, 0), dtype=torch.long)
        )
        graphs.append(Data(x=x, edge_index=edge_index, timestamp=torch.tensor([int(seq)])))

    return graphs, mappings


def graph_to_pyg_data(graph: nx.Graph, node_mapping: Dict[str, int]) -> Data:
    """Convert a NetworkX graph into a PyTorch Geometric Data object."""
    edges = [(node_mapping[u], node_mapping[v]) for u, v in graph.edges()]
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    x = torch.tensor([graph.nodes[n]["attr"] for n in graph.nodes()], dtype=torch.float)
    return Data(x=x, edge_index=edge_index)


def build_graph_dataset_from_dataframe(
    df: pd.DataFrame,
    edges: pd.DataFrame,
    split_name: str,
    verbose: bool = True,
) -> dict:
    """Build one split of the STWG graph dataset in memory."""
    edges_split = filter_edges_by_locations(edges, df["Location"].unique().tolist())

    env_graph = build_station_graph(df, edges_split, ENV_FEATURE_COLUMNS)
    risk_graph = build_station_graph(df, edges_split, RISK_FEATURE_COLUMNS)

    if verbose:
        print(f"\n=== {split_name}: Environment-view graph summary ===")
        print(summarize_graph(df, edges_split, env_graph).to_string(index=False))
        print(f"\n=== {split_name}: Risk-view graph summary ===")
        print(summarize_graph(df, edges_split, risk_graph).to_string(index=False))

    env_graphs, env_node_maps = build_sequence_graphs(df, edges_split, ENV_FEATURE_COLUMNS)
    risk_graphs, risk_node_maps = build_sequence_graphs(df, edges_split, RISK_FEATURE_COLUMNS)
    print(
        f"[Build] {split_name}: created {len(env_graphs):,} environment-view graphs "
        f"and {len(risk_graphs):,} risk-view graphs."
    )

    node_mapping = {node: i for i, node in enumerate(env_graph.nodes())}
    data_env = graph_to_pyg_data(env_graph, node_mapping)
    data_risk = graph_to_pyg_data(risk_graph, node_mapping)

    return {
        "node_mapping": node_mapping,
        "data_main": data_env,
        "data_risk": data_risk,
        "main_graphs": env_graphs,
        "risk_graphs": risk_graphs,
        "main_node_maps": env_node_maps,
        "risk_node_maps": risk_node_maps,
    }


def build_graph_splits_from_raw(
    raw_df: pd.DataFrame,
    edges: pd.DataFrame,
    splits: tuple[str, ...] = ("train", "val", "test"),
    verbose: bool = True,
) -> dict[str, dict]:
    """Create train/validation/test graph datasets in memory from one raw table."""
    graph_splits = {}
    for split in splits:
        prepared = prepare_split_dataframe_from_raw(raw_df, split=split)
        graph_splits[split] = build_graph_dataset_from_dataframe(
            prepared,
            edges,
            split_name=split,
            verbose=verbose,
        )
    return graph_splits


def build_graph_splits_from_files(
    data_dir: str | Path = "data",
    edge_csv: str | Path = "data/small_edge.csv",
    chunk_pattern: str = "final_data*.csv",
    expected_parts: int | None = 20,
    water_encoding: str | None = "utf-8",
    edge_encoding: str = "utf-8",
    splits: tuple[str, ...] = ("train", "val", "test"),
    verbose: bool = True,
) -> dict[str, dict]:
    """Load chunked CSV files and build graph datasets in memory."""
    raw_df = load_water_quality_source(
        data_dir=data_dir,
        input_csv=None,
        chunk_pattern=chunk_pattern,
        expected_parts=expected_parts,
        encoding=water_encoding,
    )
    edges = load_edges(edge_csv, encoding=edge_encoding)
    return build_graph_splits_from_raw(raw_df, edges, splits=splits, verbose=verbose)


def save_graph_dataset(graph_dataset: dict, output_pt: str | Path) -> None:
    """Optionally save an in-memory graph dataset for local debugging."""
    output_pt = Path(output_pt)
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph_dataset, output_pt)
    print(f"[Save] graph dataset -> {output_pt}")
