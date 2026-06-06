"""Data loading, chronological splitting, and feature preparation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from .constants import (
    COLUMN_ALIASES,
    ENV_FEATURE_COLUMNS,
    INTERNAL_LOCATION_COLUMN,
    NODE_ID_COLUMN,
    PREPARED_COLUMNS,
    RAW_DATE_COLUMN,
    RAW_LOCATION_COLUMN,
    REQUIRED_INPUT_COLUMNS,
    RISK_FEATURE_COLUMNS,
    SEQUENCE_COLUMN,
)

SplitName = Literal["train", "val", "test"]


def sigmoid(x):
    """Apply the logistic sigmoid transformation."""
    return 1.0 / (1.0 + np.exp(-x))


def remove_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove index columns accidentally saved by pandas."""
    return df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")]


def normalize_public_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize known spelling variants in the public CSV schema."""
    out = df.copy()
    rename_map = {src: dst for src, dst in COLUMN_ALIASES.items() if src in out.columns and dst not in out.columns}
    if rename_map:
        out = out.rename(columns=rename_map)
    return out


def validate_input_schema(df: pd.DataFrame) -> None:
    """Validate that the required public CSV columns are available."""
    missing = [col for col in REQUIRED_INPUT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing required input columns after alias normalization: "
            f"{missing}. Check data/README.md for the expected CSV schema."
        )


def find_chunk_files(
    data_dir: str | Path = "data",
    pattern: str = "final_data*.csv",
    expected_parts: int | None = 20,
) -> list[Path]:
    """Find chunked water-quality CSV files in deterministic order."""
    data_dir = Path(data_dir)
    paths = sorted(data_dir.glob(pattern))

    if not paths:
        raise FileNotFoundError(
            f"No input chunks were found. Expected files like {data_dir / 'final_data01.csv'} "
            f"using pattern '{pattern}'."
        )

    if expected_parts is not None and len(paths) != expected_parts:
        raise FileNotFoundError(
            f"Expected {expected_parts} chunk files using pattern '{pattern}', but found {len(paths)}: "
            f"{[p.name for p in paths]}"
        )

    return paths


def load_water_quality_chunks(
    data_dir: str | Path = "data",
    pattern: str = "final_data*.csv",
    expected_parts: int | None = 20,
    encoding: str | None = "utf-8",
) -> pd.DataFrame:
    """Load and concatenate chunked water-quality tables.

    The public repository assumes that the original large table is split into
    final_data01.csv, final_data02.csv, ..., final_data20.csv under data/.
    """
    paths = find_chunk_files(data_dir=data_dir, pattern=pattern, expected_parts=expected_parts)
    frames = []
    for path in paths:
        print(f"[Load] {path}")
        frame = pd.read_csv(path, encoding=encoding)
        frame = remove_unnamed_columns(frame)
        frame = normalize_public_column_names(frame)
        validate_input_schema(frame)
        frames.append(frame)

    df = pd.concat(frames, axis=0, ignore_index=True)
    print(f"[Load] concatenated water-quality table: {df.shape[0]:,} rows, {df.shape[1]:,} columns")
    return df


def load_water_quality_table(path: str | Path, encoding: str | None = "utf-8") -> pd.DataFrame:
    """Load a single water-quality CSV file for local development."""
    df = pd.read_csv(path, encoding=encoding)
    df = remove_unnamed_columns(df)
    df = normalize_public_column_names(df)
    validate_input_schema(df)
    return df


def load_water_quality_source(
    data_dir: str | Path = "data",
    input_csv: str | Path | None = None,
    chunk_pattern: str = "final_data*.csv",
    expected_parts: int | None = 20,
    encoding: str | None = "utf-8",
) -> pd.DataFrame:
    """Load either one CSV file or chunked CSV files."""
    if input_csv is not None:
        return load_water_quality_table(input_csv, encoding=encoding)
    return load_water_quality_chunks(
        data_dir=data_dir,
        pattern=chunk_pattern,
        expected_parts=expected_parts,
        encoding=encoding,
    )


def split_by_year(df: pd.DataFrame, split: SplitName, date_col: str = RAW_DATE_COLUMN) -> pd.DataFrame:
    """Split water-quality records into train, validation, or test periods."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out["year"] = out[date_col].dt.year

    if split == "train":
        out = out.loc[~out["year"].isin([2023, 2024, 2025])].reset_index(drop=True)
    elif split == "val":
        out = out.loc[out["year"] == 2023].reset_index(drop=True)
    elif split == "test":
        out = out.loc[out["year"] == 2024].reset_index(drop=True)
    else:
        raise ValueError(f"Unknown split: {split}")

    return out.drop(columns=["year"])


def standardize_column_layout(df: pd.DataFrame, date_col: str = RAW_DATE_COLUMN) -> pd.DataFrame:
    """Create sequence and node identifiers and align internal column names."""
    out = normalize_public_column_names(df)
    validate_input_schema(out)

    out = out.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out[RAW_LOCATION_COLUMN] = out[RAW_LOCATION_COLUMN].astype(str)
    out = out.sort_values([RAW_LOCATION_COLUMN, date_col]).reset_index(drop=True)

    out[SEQUENCE_COLUMN] = out.groupby(RAW_LOCATION_COLUMN).cumcount()
    out[NODE_ID_COLUMN] = out[RAW_LOCATION_COLUMN] + "_" + out[SEQUENCE_COLUMN].astype(str).str.zfill(4)
    out[INTERNAL_LOCATION_COLUMN] = out[RAW_LOCATION_COLUMN]

    prepared = out[PREPARED_COLUMNS].copy()
    return prepared


def align_risk_directions(df: pd.DataFrame) -> pd.DataFrame:
    """Align environment-view standardized features with the feature-level risk direction."""
    out = df.copy()
    out["DO_scaled"] = -out["DO_scaled"]
    out["pH_scaled"] = np.abs(out["pH_scaled"])
    out[ENV_FEATURE_COLUMNS] = out[ENV_FEATURE_COLUMNS].apply(sigmoid)
    return out


def prepare_split_dataframe_from_raw(
    raw_df: pd.DataFrame,
    split: SplitName,
) -> pd.DataFrame:
    """Split, standardize, and risk-align an already-loaded water-quality table."""
    split_df = split_by_year(raw_df, split=split)
    prepared = standardize_column_layout(split_df)
    prepared = align_risk_directions(prepared)
    print(f"[Prepare] {split}: {prepared.shape[0]:,} rows, {prepared['Location'].nunique():,} locations")
    return prepared


def load_edges(path: str | Path, encoding: str = "utf-8") -> pd.DataFrame:
    """Load and deduplicate hydrological adjacency edges."""
    edges = pd.read_csv(path, encoding=encoding)
    edges = remove_unnamed_columns(edges)
    required = {"node1", "node2"}
    missing = required.difference(edges.columns)
    if missing:
        raise ValueError(f"Missing required edge columns: {sorted(missing)}")

    edges = edges.copy()
    edges["node1"] = edges["node1"].astype(str)
    edges["node2"] = edges["node2"].astype(str)
    edges["pair"] = edges.apply(lambda r: frozenset([r["node1"], r["node2"]]), axis=1)
    edges = edges.loc[~edges["pair"].duplicated(), ["node1", "node2"]].reset_index(drop=True)
    print(f"[Load] hydrological edges: {len(edges):,} unique undirected pairs")
    return edges


def filter_edges_by_locations(edges: pd.DataFrame, locations: Iterable[str]) -> pd.DataFrame:
    """Keep only edges whose endpoints exist in the current split."""
    locations = set(map(str, locations))
    return edges[
        edges["node1"].isin(locations) & edges["node2"].isin(locations)
    ].reset_index(drop=True)


__all__ = [
    "ENV_FEATURE_COLUMNS",
    "RISK_FEATURE_COLUMNS",
    "find_chunk_files",
    "load_water_quality_chunks",
    "load_water_quality_source",
    "prepare_split_dataframe_from_raw",
    "load_edges",
    "filter_edges_by_locations",
]
