# Extra validation

This folder is intended for analyses that are not required for training GTCAE but are used to support the paper-level validation.

## Ecological validation

```bash
python experiment/extra_validation/01_ecological_fixed_effects.py \
  --monthly_wqri data/monthly_subbasin_wqri.csv \
  --biodiversity data/biodiversity.csv \
  --output_csv experiment/extra_validation/ecological_fixed_effects.csv
```

Expected columns:

- `monthly_subbasin_wqri.csv`: `sub_basin`, `month`, `WQRI`
- `biodiversity.csv`: `sub_basin`, `survey_month`, `biodiversity`

This script estimates sub-basin fixed-effect regressions for cumulative recent WQRI exposure and single-month lagged WQRI exposure.
