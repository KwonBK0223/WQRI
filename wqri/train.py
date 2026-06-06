"""Training routine for GTCAE."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .constants import DEFAULT_WINDOW_LIST
from .dataset import TemporalWindowDataset
from .inference import run_inference
from .model import GraphTCNAE, build_tcn_channels, compute_receptive_field, tcn_layers_by_window
from .scheduler import WarmupCosineScheduler
from .utils import clear_memory, ensure_dir, seed_everything


@dataclass
class ExperimentConfig:
    """Configuration for one GTCAE experiment sweep."""

    seed: int = 42
    device: str = "cuda:0"
    version: str = "904"
    conv_type: str = "gcn"
    num_spatial_layers: int = 1
    heads: int = 4
    in_feats: int = 10
    hid_feats: int = 16
    latent_dim: int = 5
    kernel_size: int = 3
    tcn_dropout: float = 0.0
    num_epochs: int = 500
    patience: int = 5
    base_lr: float = 1e-4
    final_lr: float = 1e-5
    warmup_epochs: int = 10
    do_inference: bool = True
    save_dir: str = "experiment/models"
    log_dir: str = "experiment/logs"
    result_dir: str = "experiment/results"
    window_list: List[int] = field(default_factory=lambda: DEFAULT_WINDOW_LIST.copy())


def train_one_setting(
    window_size: int,
    cfg: ExperimentConfig,
    train_graphs: dict,
    val_graphs: dict,
    test_graphs: dict,
):
    """Train GTCAE for one temporal window size."""
    seed_everything(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    num_tcn_layers = tcn_layers_by_window(window_size)
    tcn_channels = build_tcn_channels(cfg.hid_feats, num_tcn_layers)
    receptive_field = compute_receptive_field(cfg.kernel_size, num_tcn_layers, dilation_base=2)

    tag = (
        f"v{cfg.version}.w{window_size}_L{num_tcn_layers}_RF{receptive_field}_"
        f"{cfg.conv_type}_sp{cfg.num_spatial_layers}_h{cfg.hid_feats}_z{cfg.latent_dim}"
    )
    print(f"[Config] {tag}")

    train_dataset = TemporalWindowDataset(train_graphs["main_graphs"], train_graphs["risk_graphs"], window_size)
    val_dataset = TemporalWindowDataset(val_graphs["main_graphs"], val_graphs["risk_graphs"], window_size)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=lambda batch: batch[0])
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda batch: batch[0])

    model = GraphTCNAE(
        in_feats=cfg.in_feats,
        hid_feats=cfg.hid_feats,
        latent_dim=cfg.latent_dim,
        window_size=window_size,
        conv_type=cfg.conv_type,
        heads=cfg.heads,
        num_spatial_layers=cfg.num_spatial_layers,
        tcn_channels=tcn_channels,
        kernel_size=cfg.kernel_size,
        tcn_dropout=cfg.tcn_dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.base_lr)
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=cfg.warmup_epochs,
        total_steps=cfg.num_epochs,
        base_lr=cfg.base_lr,
        final_lr=cfg.final_lr,
    )

    logs = {
        "epoch": [],
        "train_total": [], "train_l1": [], "train_l2": [],
        "val_total": [], "val_l1": [], "val_l2": [],
        "lr": [],
    }
    best_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        train_total = train_l1 = train_l2 = 0.0

        pbar = tqdm(train_loader, desc=f"[Train {tag}] Epoch {epoch}", leave=False)
        for main_window, risk_window in pbar:
            main_window = [g.to(device) for g in main_window]
            risk_window = [g.to(device) for g in risk_window]

            reconstruction, _ = model(main_window)
            x_env = torch.stack([g.x for g in main_window], dim=0)
            x_risk = torch.stack([g.x for g in risk_window], dim=0)

            loss_env = F.mse_loss(reconstruction, x_env)
            loss_risk = F.mse_loss(reconstruction, x_risk)
            loss = loss_env + loss_risk

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_total += loss.item()
            train_l1 += loss_env.item()
            train_l2 += loss_risk.item()
            pbar.set_postfix({"total": f"{loss.item():.4f}", "env": f"{loss_env.item():.4f}", "risk": f"{loss_risk.item():.4f}"})

        avg_train_total = train_total / max(1, len(train_loader))
        avg_train_l1 = train_l1 / max(1, len(train_loader))
        avg_train_l2 = train_l2 / max(1, len(train_loader))

        model.eval()
        val_total = val_l1 = val_l2 = 0.0
        with torch.no_grad():
            for main_window, risk_window in val_loader:
                main_window = [g.to(device) for g in main_window]
                risk_window = [g.to(device) for g in risk_window]
                reconstruction, _ = model(main_window)
                x_env = torch.stack([g.x for g in main_window], dim=0)
                x_risk = torch.stack([g.x for g in risk_window], dim=0)
                loss_env = F.mse_loss(reconstruction, x_env)
                loss_risk = F.mse_loss(reconstruction, x_risk)
                loss = loss_env + loss_risk
                val_total += loss.item()
                val_l1 += loss_env.item()
                val_l2 += loss_risk.item()

        avg_val_total = val_total / max(1, len(val_loader))
        avg_val_l1 = val_l1 / max(1, len(val_loader))
        avg_val_l2 = val_l2 / max(1, len(val_loader))

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        for key, value in [
            ("epoch", epoch),
            ("train_total", avg_train_total),
            ("train_l1", avg_train_l1),
            ("train_l2", avg_train_l2),
            ("val_total", avg_val_total),
            ("val_l1", avg_val_l1),
            ("val_l2", avg_val_l2),
            ("lr", current_lr),
        ]:
            logs[key].append(value)

        print(
            f"[Epoch {epoch:03d}] train_total={avg_train_total:.4f}, train_env={avg_train_l1:.4f}, "
            f"train_risk={avg_train_l2:.4f} | val_total={avg_val_total:.4f}, "
            f"val_env={avg_val_l1:.4f}, val_risk={avg_val_l2:.4f} | lr={current_lr:.6f} | patience={patience_counter}"
        )

        if avg_val_total < best_loss:
            best_loss = avg_val_total
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= cfg.patience:
            print(f"[Early stopping] No improvement for {cfg.patience} epochs. Stopped at epoch {epoch}.")
            break

    trained_epochs = len(logs["epoch"])
    if best_state is not None:
        model.load_state_dict(best_state)

    ensure_dir(cfg.log_dir)
    log_path = Path(cfg.log_dir) / f"v{cfg.version}.train_log_{tag}.csv"
    pd.DataFrame(logs).to_csv(log_path, index=False)
    print(f"[Save] log -> {log_path}")

    if cfg.do_inference:
        ensure_dir(cfg.result_dir)
        prefix = Path(cfg.result_dir) / tag
        run_inference(
            "test",
            test_graphs["main_graphs"],
            test_graphs["risk_graphs"],
            test_graphs["main_node_maps"],
            model,
            window_size,
            cfg.latent_dim,
            device,
            prefix,
        )

    ensure_dir(cfg.save_dir)
    model_path = Path(cfg.save_dir) / f"v{cfg.version}.graph_tcn_ae_{tag}.pt"
    torch.save(
        {
            "tag": tag,
            "window_size": window_size,
            "num_tcn_layers": num_tcn_layers,
            "kernel_size": cfg.kernel_size,
            "receptive_field": receptive_field,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": trained_epochs,
            "best_loss": best_loss,
            "config": asdict(cfg),
        },
        model_path,
    )
    print(f"[Save] model -> {model_path}")
    clear_memory()
    return tag, best_loss, trained_epochs, receptive_field


def train_window_sweep(cfg: ExperimentConfig, graph_splits: dict[str, dict]) -> pd.DataFrame:
    """Train GTCAE over all configured window sizes and save a summary."""
    summary = []
    for window_size in cfg.window_list:
        tag, best_loss, trained_epochs, receptive_field = train_one_setting(
            window_size,
            cfg,
            graph_splits["train"],
            graph_splits["val"],
            graph_splits["test"],
        )
        summary.append(
            {
                "tag": tag,
                "window": window_size,
                "RF": receptive_field,
                "best_val_loss": best_loss,
                "trained_epochs": trained_epochs,
            }
        )

    df_summary = pd.DataFrame(summary).sort_values("best_val_loss")
    ensure_dir(cfg.log_dir)
    out_path = Path(cfg.log_dir) / f"v{cfg.version}.exp_summary_graph_tcn_ae.csv"
    df_summary.to_csv(out_path, index=False)
    print(f"[Done] Saved summary -> {out_path}")
    return df_summary
