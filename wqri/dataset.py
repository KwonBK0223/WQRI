"""Dataset definitions for temporal graph-window learning."""

from __future__ import annotations

from torch.utils.data import Dataset


class TemporalWindowDataset(Dataset):
    """Return paired environment-view and risk-view graph windows."""

    def __init__(self, main_graphs, risk_graphs, window_size: int = 30):
        if len(main_graphs) != len(risk_graphs):
            raise ValueError("main_graphs and risk_graphs must have the same length.")
        self.main = main_graphs
        self.risk = risk_graphs
        self.window_size = window_size
        self.num_timepoints = len(main_graphs)

    def __len__(self) -> int:
        return self.num_timepoints - self.window_size + 1

    def __getitem__(self, idx):
        return (
            self.main[idx : idx + self.window_size],
            self.risk[idx : idx + self.window_size],
        )
