"""Graph Temporal Convolutional AutoEncoder for WQRI estimation."""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, TransformerConv


class Chomp1d(nn.Module):
    """Remove extra right-side padding to preserve causal convolution length."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size]


class TemporalBlock(nn.Module):
    """Residual causal temporal convolution block."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(out + residual)


class TemporalConvNet(nn.Module):
    """Stacked temporal convolution network with exponentially increasing dilation."""

    def __init__(self, in_ch: int, channels: List[int], kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        layers = []
        prev = in_ch
        for i, ch in enumerate(channels):
            dilation = 2 ** i
            layers.append(TemporalBlock(prev, ch, kernel_size=kernel_size, dilation=dilation, dropout=dropout))
            prev = ch
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def compute_receptive_field(kernel_size: int, num_layers: int, dilation_base: int = 2) -> int:
    """Compute the temporal receptive field for exponentially dilated TCN blocks."""
    dilation_sum = sum(dilation_base ** i for i in range(num_layers))
    return 1 + (kernel_size - 1) * dilation_sum


def build_tcn_channels(hidden_dim: int, num_layers: int) -> List[int]:
    """Build a constant-width TCN channel list."""
    return [hidden_dim] * num_layers


def tcn_layers_by_window(window_size: int) -> int:
    """Return the TCN depth used in the reported GTCAE setting."""
    return 3


class GraphTCNAE(nn.Module):
    """Graph Temporal Convolutional AutoEncoder."""

    def __init__(
        self,
        in_feats: int,
        hid_feats: int = 32,
        latent_dim: int = 5,
        window_size: int = 20,
        conv_type: str = "gcn",
        heads: int = 4,
        num_spatial_layers: int = 2,
        tcn_channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        tcn_dropout: float = 0.0,
    ):
        super().__init__()
        self.window_size = window_size
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.latent_dim = latent_dim
        self.conv_type = conv_type.lower()
        self.heads = heads
        self.num_spatial_layers = num_spatial_layers

        self.spatial_convs = nn.ModuleList()
        for i in range(num_spatial_layers):
            in_dim = in_feats if i == 0 else hid_feats
            out_dim = hid_feats
            if self.conv_type == "gcn":
                self.spatial_convs.append(GCNConv(in_dim, out_dim))
            elif self.conv_type == "gat":
                if out_dim % heads != 0:
                    raise ValueError("hid_feats must be divisible by heads for GATConv.")
                self.spatial_convs.append(GATConv(in_dim, out_dim // heads, heads=heads, concat=True))
            elif self.conv_type == "transformer":
                if out_dim % heads != 0:
                    raise ValueError("hid_feats must be divisible by heads for TransformerConv.")
                self.spatial_convs.append(TransformerConv(in_dim, out_dim // heads, heads=heads, concat=True))
            else:
                raise ValueError(f"Unknown conv_type: {conv_type}")

        if tcn_channels is None:
            tcn_channels = [hid_feats, hid_feats, hid_feats]

        self.temporal_enc = TemporalConvNet(hid_feats, tcn_channels, kernel_size=kernel_size, dropout=tcn_dropout)
        self.proj_down = nn.Linear(tcn_channels[-1], latent_dim)
        self.proj_up = nn.Linear(latent_dim, hid_feats)
        self.temporal_dec = TemporalConvNet(hid_feats, tcn_channels, kernel_size=kernel_size, dropout=tcn_dropout)
        self.proj_out = nn.Linear(tcn_channels[-1], in_feats)

    def _check_constant_num_nodes(self, graph_window) -> int:
        num_nodes = graph_window[0].num_nodes
        for t in range(1, self.window_size):
            if graph_window[t].num_nodes != num_nodes:
                raise RuntimeError(
                    "All graph snapshots within a window must have the same number of nodes. "
                    f"Found {num_nodes} and {graph_window[t].num_nodes}."
                )
        return num_nodes

    def forward(self, main_graph_window):
        if len(main_graph_window) != self.window_size:
            raise ValueError(f"Expected a window of length {self.window_size}, got {len(main_graph_window)}.")
        self._check_constant_num_nodes(main_graph_window)

        spatial_sequence = []
        for graph in main_graph_window:
            x = graph.x
            edge_index = graph.edge_index
            for conv in self.spatial_convs:
                x = F.relu(conv(x, edge_index))
            spatial_sequence.append(x)

        hidden = torch.stack(spatial_sequence, dim=0)
        hidden_node_channel_time = hidden.permute(1, 2, 0).contiguous()

        memory = self.temporal_enc(hidden_node_channel_time)
        last_state = memory[:, :, -1]
        latent = self.proj_down(last_state)

        decoder_seed = self.proj_up(latent)
        decoder_input = decoder_seed.unsqueeze(-1).repeat(1, 1, self.window_size)
        decoded = self.temporal_dec(decoder_input)
        decoded_time_major = decoded.permute(2, 0, 1).contiguous()
        reconstruction = self.proj_out(decoded_time_major)
        return reconstruction, latent
