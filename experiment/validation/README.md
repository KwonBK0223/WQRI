# Validation scripts

This folder contains scripts for quantitative validation after GTCAE inference.

## Percentile-gap validation

```bash
python experiment/validation/01_percentile_gap_validation.py \
  --train_result experiment/results_inference_only/<tag>_result_train.csv \
  --eval_result experiment/results_inference_only/<tag>_result_test.csv \
  --output_csv experiment/validation/percentile_gap_<tag>.csv
```

The script computes WQRI from latent features using training-set latent statistics, derives the reference risk probability from risk-view columns, and calculates the upper-lower percentile gap.
