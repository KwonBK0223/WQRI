"""Inference utilities for extracting GTCAE latent representations."""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from tqdm.auto import tqdm

from .constants import ENV_FEATURE_COLUMNS, RISK_FEATURE_COLUMNS
from .dataset import TemporalWindowDataset
from .model import GraphTCNAE, build_tcn_channels
from .utils import clear_memory


def find_checkpoint(models_dir: str | os.PathLike[str], version: str, window_size: int) -> str:
    """Find the checkpoint corresponding to a specific window size."""
    pattern = os.path.join(str(models_dir), f"v{version}.graph_tcn_ae_*w{window_size}_*.pt")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found for window={window_size}; pattern={pattern}")
    if len(candidates) > 1:
        print(f"[Warning] Multiple checkpoints found for window={window_size}; using {candidates[-1]}")
    return candidates[-1]


def run_inference(
    split_name: str,
    main_graphs,
    risk_graphs,
    node_maps,
    model: GraphTCNAE,
    window_size: int,
    latent_dim: int,
    device,
    out_prefix: str | os.PathLike[str],
    return_df: bool = False,
) -> Optional[pd.DataFrame]:
    """Run window-based inference and save node-level latent features."""
    dataset = TemporalWindowDataset(main_graphs, risk_graphs, window_size)
    model.eval()
    records = []

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc=f"[Infer {split_name}]"):
            main_window, risk_window = dataset[idx]
            main_window = [g.to(device) for g in main_window]
            risk_window = [g.to(device) for g in risk_window]

            _, latent = model(main_window)
            latent_np = latent.detach().cpu().numpy()
            num_nodes = latent_np.shape[0]

            for t in range(window_size):
                env_graph = main_window[t]
                risk_graph = risk_window[t]
                seq_t = int(env_graph.timestamp.item())

                env_np = env_graph.x.detach().cpu().numpy()
                risk_np = risk_graph.x.detach().cpu().numpy()
                inverse_map = {idx_: node_id for node_id, idx_ in node_maps[seq_t].items()}

                for local_idx in range(num_nodes):
                    row = {
                        "node_idx": local_idx,
                        "node_id": inverse_map[local_idx],
                        "seq": seq_t,
                    }
                    for j, value in enumerate(env_np[local_idx]):
                        row[ENV_FEATURE_COLUMNS[j]] = value
                    for j, value in enumerate(risk_np[local_idx]):
                        row[RISK_FEATURE_COLUMNS[j]] = value
                    for j, value in enumerate(latent_np[local_idx]):
                        row[f"latent_{j}"] = value
                    records.append(row)

            del main_window, risk_window, latent, latent_np
            clear_memory()

    df_all = pd.DataFrame(records)
    columns = ["node_idx", "node_id", "seq"] + ENV_FEATURE_COLUMNS + RISK_FEATURE_COLUMNS + [f"latent_{i}" for i in range(latent_dim)]
    df_all = df_all[columns]

    out_path = f"{out_prefix}_result_{split_name}.csv"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(out_path, index=False)
    print(f"[Save] inference result -> {out_path}")

    if return_df:
        return df_all
    return None


def load_model_from_checkpoint(ckpt_path: str | os.PathLike[str], device) -> tuple[GraphTCNAE, dict]:
    """Load a GTCAE model and metadata from a saved checkpoint."""
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    cfg = checkpoint["config"]
    tcn_channels = build_tcn_channels(cfg["hid_feats"], checkpoint["num_tcn_layers"])

    model = GraphTCNAE(
        in_feats=cfg["in_feats"],
        hid_feats=cfg["hid_feats"],
        latent_dim=cfg["latent_dim"],
        window_size=checkpoint["window_size"],
        conv_type=cfg["conv_type"],
        heads=cfg["heads"],
        num_spatial_layers=cfg["num_spatial_layers"],
        tcn_channels=tcn_channels,
        kernel_size=cfg["kernel_size"],
        tcn_dropout=cfg["tcn_dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint
