"""Ecological validation template using sub-basin fixed effects.

Expected inputs:
- Monthly WQRI table with columns: sub_basin, month, WQRI
- Biodiversity table with columns: sub_basin, survey_month, biodiversity

The script creates cumulative recent-exposure variables and estimates fixed-effect
regressions with cluster-robust standard errors at the sub-basin level.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def build_cumulative_exposure(monthly_wqri: pd.DataFrame, biodiversity: pd.DataFrame, max_lag: int = 6) -> pd.DataFrame:
    monthly = monthly_wqri.copy()
    bio = biodiversity.copy()
    monthly["month"] = pd.to_datetime(monthly["month"])
    bio["survey_month"] = pd.to_datetime(bio["survey_month"])

    rows = []
    for _, target in bio.iterrows():
        sub_basin = target["sub_basin"]
        survey_month = target["survey_month"]
        sub_monthly = monthly[monthly["sub_basin"] == sub_basin]
        row = target.to_dict()

        for lag in range(1, max_lag + 1):
            start = survey_month - pd.DateOffset(months=lag)
            end = survey_month - pd.DateOffset(months=1)
            mask = (sub_monthly["month"] >= start) & (sub_monthly["month"] <= end)
            row[f"recent_{lag}m_mean_wqri"] = sub_monthly.loc[mask, "WQRI"].mean()

            single_month = survey_month - pd.DateOffset(months=lag)
            single_mask = sub_monthly["month"].dt.to_period("M") == single_month.to_period("M")
            row[f"lag_{lag}m_wqri"] = sub_monthly.loc[single_mask, "WQRI"].mean()

        rows.append(row)
    return pd.DataFrame(rows)


def run_fixed_effects(df: pd.DataFrame, exposure_col: str) -> dict:
    model_df = df.dropna(subset=["biodiversity", exposure_col, "sub_basin"]).copy()
    formula = f"biodiversity ~ {exposure_col} + C(sub_basin)"
    result = smf.ols(formula, data=model_df).fit(cov_type="cluster", cov_kwds={"groups": model_df["sub_basin"]})
    coef = result.params[exposure_col]
    pvalue = result.pvalues[exposure_col]
    ci_low, ci_high = result.conf_int().loc[exposure_col].tolist()
    return {
        "exposure": exposure_col,
        "n_obs": int(result.nobs),
        "coef": coef,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": pvalue,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run ecological validation with sub-basin fixed effects.")
    parser.add_argument("--monthly_wqri", type=str, required=True, help="Monthly sub-basin WQRI CSV.")
    parser.add_argument("--biodiversity", type=str, required=True, help="Biodiversity observation CSV.")
    parser.add_argument("--output_csv", type=str, default="experiment/extra_validation/ecological_fixed_effects.csv", help="Output CSV path.")
    parser.add_argument("--max_lag", type=int, default=6, help="Maximum lag/exposure window in months.")
    return parser.parse_args()


def main():
    args = parse_args()
    monthly_wqri = pd.read_csv(args.monthly_wqri)
    biodiversity = pd.read_csv(args.biodiversity)
    panel = build_cumulative_exposure(monthly_wqri, biodiversity, max_lag=args.max_lag)

    rows = []
    for lag in range(1, args.max_lag + 1):
        rows.append(run_fixed_effects(panel, f"recent_{lag}m_mean_wqri"))
        rows.append(run_fixed_effects(panel, f"lag_{lag}m_wqri"))

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"[Save] ecological validation -> {output_csv}")


if __name__ == "__main__":
    main()
