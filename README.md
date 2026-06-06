# Water Quality Risk Index Estimation using Graph Neural Networks for Spatio-temporal Watershed Systems

This repository provides the source code for the study:

> **Water Quality Risk Index Estimation using Graph Neural Networks for Spatio-temporal Watershed Systems**  
> Byeongkeun Kwon, Dasom Seong, Jiyun Park, Hyeonjun Hwang, and Suhyeon Kim

## Overview

This study proposes a graph-based framework for estimating a **Water Quality Risk Index (WQRI)** from nationwide river water quality monitoring data. The framework represents monitoring sites and temporal observations as a **Spatio-Temporal Watershed Graph (STWG)** and trains a **Graph Temporal Convolutional AutoEncoder (GTCAE)** to learn risk-aware latent water quality representations.

The learned latent features are standardized and aggregated into a scalar WQRI, where higher values indicate higher water quality risk.

## Repository structure

```text
WQRI/
├── wqri/
│   ├── __init__.py
│   ├── constants.py
│   ├── data.py
│   ├── dataset.py
│   ├── graph_builder.py
│   ├── inference.py
│   ├── model.py
│   ├── scheduler.py
│   ├── train.py
│   └── utils.py
├── data/
│   └── README.md
├── experiment/
│   ├── config.yaml
│   ├── 00_prepare_graph_dataset.py
│   ├── 01_train_gtcae.py
│   ├── 02_infer_gtcae.py
│   ├── validation/
│   │   ├── README.md
│   │   └── 01_percentile_gap_validation.py
│   └── extra_validation/
│       ├── README.md
│       └── 01_ecological_fixed_effects.py
├── .gitignore
├── LICENSE
├── README.md
└── wqri.yaml
```

Only three top-level folders are used: `wqri`, `data`, and `experiment`. Runtime output folders such as `experiment/logs`, `experiment/models`, `experiment/results`, and `experiment/results_inference_only` are created automatically when the scripts are executed.

## Environment

```bash
conda env create -f wqri.yaml
conda activate wqri
```

The author's setup used Python 3.10, PyTorch 2.5.x, PyTorch Geometric 2.5.x/2.6.x, and CUDA-enabled GPU acceleration.

## Data

Because the original preprocessed water-quality table can be large, this repository assumes that it is split into twenty UTF-8 encoded CSV files:

```text
data/final_data01.csv
data/final_data02.csv
data/final_data03.csv
...
data/final_data20.csv
```

The hydrological adjacency file should also be placed under `data/`:

```text
data/small_edge.csv
```

`small_edge.csv` should be UTF-8 encoded and include the following columns:

```text
node1,node2
```

The water-quality chunks should use English column names. The required public schema is:

```text
Observation_point, Observation_date,
Water_temperature, DO, BOD, COD, SS, TN, TP, TOC, pH, EC,
Water_temperature_scaled, DO_scaled, BOD_scaled, COD_scaled, SS_scaled,
TN_scaled, TP_scaled, TOC_scaled, pH_scaled, EC_scaled,
BOD_type1, COD_type1, SS_type1, TN_type1, TP_type1, TOC_type1,
Water_temperature_type1, EC_type1, DO_type2, pH_type3
```

For compatibility with previously generated files, the loader also accepts the spelling variants `Water_temparature_scaled` and `Water_temparature_type1`, and maps them internally to `Water_temperature_scaled` and `Water_temperature_type1`.

The train/validation/test split is not stored as separate files. It is implemented inside the code:

```text
Training:   2014-2022
Validation: 2023
Test:       2024
```

Graph datasets are also not stored as `.pt` files. They are constructed in memory during training and inference.

## Optional input check

This step loads the twenty CSV chunks, loads `small_edge.csv`, performs the chronological split, and builds graph snapshots in memory once. It does not save train/validation/test CSV files or graph `.pt` files.

```bash
python experiment/00_prepare_graph_dataset.py \
  --data_dir data \
  --chunk_pattern "final_data*.csv" \
  --expected_parts 20 \
  --edge_csv data/small_edge.csv \
  --water_encoding utf-8 \
  --edge_encoding utf-8
```

## Train GTCAE

```bash
python experiment/01_train_gtcae.py \
  --config experiment/config.yaml \
  --data_dir data \
  --chunk_pattern "final_data*.csv" \
  --expected_parts 20 \
  --edge_csv data/small_edge.csv \
  --water_encoding utf-8 \
  --edge_encoding utf-8
```

Outputs are created automatically when the script runs:

```text
experiment/models/v904.graph_tcn_ae_*.pt
experiment/logs/v904.train_log_*.csv
experiment/logs/v904.exp_summary_graph_tcn_ae.csv
experiment/results/*_result_test.csv
```

The default configuration evaluates temporal window sizes:

```text
[3, 5, 7, 10, 14, 20, 30]
```

## Run inference from saved checkpoints

```bash
python experiment/02_infer_gtcae.py \
  --version 904 \
  --models_dir experiment/models \
  --result_dir experiment/results_inference_only \
  --data_dir data \
  --chunk_pattern "final_data*.csv" \
  --expected_parts 20 \
  --edge_csv data/small_edge.csv \
  --water_encoding utf-8 \
  --edge_encoding utf-8
```

Outputs are created automatically when the script runs:

```text
experiment/results_inference_only/*_result_train.csv
experiment/results_inference_only/*_result_val.csv
experiment/results_inference_only/*_result_test.csv
experiment/results_inference_only/v904_inference_summary.csv
```

## Quantitative validation

```bash
python experiment/validation/01_percentile_gap_validation.py \
  --train_result experiment/results_inference_only/<tag>_result_train.csv \
  --eval_result experiment/results_inference_only/<tag>_result_test.csv \
  --output_csv experiment/validation/percentile_gap_<tag>.csv
```

This script computes WQRI from latent features using training-set latent statistics, derives the reference risk probability from risk-view columns, and calculates the upper-lower percentile gap.

## Extra ecological validation

```bash
python experiment/extra_validation/01_ecological_fixed_effects.py \
  --monthly_wqri data/monthly_subbasin_wqri.csv \
  --biodiversity data/biodiversity.csv \
  --output_csv experiment/extra_validation/ecological_fixed_effects.csv
```

This script estimates sub-basin fixed-effect regressions for cumulative recent WQRI exposure and single-month lagged WQRI exposure.

## Model overview

GTCAE consists of three main components:

1. **Spatial graph encoder**: applies graph convolution to each daily watershed graph snapshot to capture hydrological connectivity among monitoring sites.
2. **Temporal convolutional encoder-decoder**: applies causal dilated temporal convolution to model recent water quality dynamics within a sliding time window.
3. **Risk-aware dual objective**: optimizes both environment-view reconstruction loss and risk-view alignment loss.

The total loss is:

```text
Loss = MSE(reconstructed environment-view, environment-view target)
     + MSE(reconstructed environment-view, risk-view target)
```

## Reproducibility notes

- Random seed is fixed to `42` by default.
- The default hidden dimension is `16`.
- The default latent dimension is `5`.
- The default TCN kernel size is `3`.
- The default number of TCN layers is `3`.
- The default optimizer is AdamW.
- The default learning-rate schedule uses 10 warm-up epochs followed by cosine decay from `1e-4` to `1e-5`.
- Early stopping is applied with patience `5` based on validation loss.

## Citation

If you find this repository useful, please cite:

```bibtex
@article{kwon2026wqri,
  title   = {Water Quality Risk Index Estimation using Graph Neural Networks for Spatio-temporal Watershed Systems},
  author  = {Kwon, Byeongkeun and Seong, Dasom and Park, Jiyun and Hwang, Hyeonjun and Kim, Suhyeon},
  journal = {Ecological Informatics},
  year    = {2026},
  note    = {Equal contribution: Byeongkeun Kwon and Dasom Seong. Co-corresponding authors: Hyeonjun Hwang and Suhyeon Kim.}
}
```

## License

Released under the MIT License. See `LICENSE` for details.

## Contact

For questions and collaborations, please open an issue or contact the authors.

- house9895@knu.ac.kr
