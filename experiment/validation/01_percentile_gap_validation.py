"""Percentile-gap validation for GTCAE-based WQRI results.

This script computes WQRI from latent features, derives an independently computed
reference risk probability from risk-view columns, and evaluates the upper-lower
percentile gap used for quantitative validation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RISK_COLUMNS = [
    "Water_temperature_type1", "DO_type2", "BOD_type1", "COD_type1", "SS_type1",
    "TN_type1", "TP_type1", "TOC_type1", "pH_type3", "EC_type1",
]


def latent_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("latent_")]
    if not cols:
        raise ValueError("No latent_* columns were found.")
    return sorted(cols, key=lambda x: int(x.split("_")[-1]))


def compute_wqri(train_df: pd.DataFrame, eval_df: pd.DataFrame, score_name: str = "WQRI") -> pd.DataFrame:
    cols = latent_columns(train_df)
    train_latent = train_df[cols].astype(float)
    eval_latent = eval_df[cols].astype(float)

    mean = train_latent.mean(axis=0)
    std = train_latent.std(axis=0).replace(0, np.nan)

    train_raw = ((train_latent - mean) / std).mean(axis=1)
    eval_raw = ((eval_latent - mean) / std).mean(axis=1)

    lo, hi = train_raw.min(), train_raw.max()
    if np.isclose(hi, lo):
        eval_score = pd.Series(np.zeros(len(eval_df)), index=eval_df.index)
    else:
        eval_score = (eval_raw - lo) / (hi - lo) * 100.0

    out = eval_df.copy()
    out[score_name] = eval_score.clip(lower=0.0, upper=100.0)
    return out


def compute_reference_risk(df: pd.DataFrame, risk_columns: list[str]) -> pd.Series:
    missing = [c for c in risk_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing risk columns: {missing}")
    return df[risk_columns].astype(float).mean(axis=1)


def percentile_gap(df: pd.DataFrame, index_col: str, risk_col: str, percentiles: list[float]) -> pd.DataFrame:
    rows = []
    sorted_df = df.sort_values(index_col, ascending=False).reset_index(drop=True)
    n = len(sorted_df)

    for p in percentiles:
        k = max(1, int(np.ceil(n * (p / 100.0))))
        upper = sorted_df.head(k)[risk_col].mean()
        lower = sorted_df.tail(k)[risk_col].mean()
        rows.append({
            "percentile": p,
            "n_upper": k,
            "upper_mean_risk": upper,
            "lower_mean_risk": lower,
            "gap": upper - lower,
        })
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Run percentile-gap validation for WQRI.")
    parser.add_argument("--train_result", type=str, required=True, help="Inference CSV from the training split.")
    parser.add_argument("--eval_result", type=str, required=True, help="Inference CSV to evaluate, usually validation or test.")
    parser.add_argument("--output_csv", type=str, default="experiment/validation/percentile_gap.csv", help="Output CSV path.")
    parser.add_argument("--percentiles", nargs="+", type=float, default=[1, 5, 10, 20, 30, 40, 50], help="Percentile levels.")
    parser.add_argument("--index_col", type=str, default="WQRI", help="Index column used for sorting.")
    parser.add_argument("--risk_cols", nargs="+", default=DEFAULT_RISK_COLUMNS, help="Risk-view columns used as reference risk probability.")
    return parser.parse_args()


def main():
    args = parse_args()
    train_df = pd.read_csv(args.train_result)
    eval_df = pd.read_csv(args.eval_result)

    scored = compute_wqri(train_df, eval_df, score_name=args.index_col)
    scored["reference_risk"] = compute_reference_risk(scored, args.risk_cols)
    gap_df = percentile_gap(scored, index_col=args.index_col, risk_col="reference_risk", percentiles=args.percentiles)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    gap_df.to_csv(output_csv, index=False)
    print(f"[Save] percentile-gap validation -> {output_csv}")


if __name__ == "__main__":
    main()
