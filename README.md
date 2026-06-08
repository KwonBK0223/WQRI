# Water Quality Risk Index Estimation using Graph Neural Networks for Spatio-temporal Watershed Systems

Status: Submitted manuscript  
Goal: This repository provides the implementation of a graph-based Water Quality Risk Index (WQRI) framework for spatio-temporal watershed systems.

---
 
## Repository Structure

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
│   ├── final_data01.csv
│   ├── final_data02.csv
│   ├── ...
│   ├── final_data20.csv
│   └── edge.csv
├── experiment/
│   ├── config.yaml
│   ├── 00_prepare_graph_dataset.py
│   ├── 01_train_gtcae.py
│   ├── 02_infer_gtcae.py
│   ├── validation/
│   └── extra_validation/
├── .gitignore
├── LICENSE
├── README.md
└── wqri.yaml
```

---

## Environment

The conda environment can be created using:

```bash
conda env create -f wqri.yaml
conda activate wqri
```

Main dependencies include Python, PyTorch, PyTorch Geometric, NumPy, pandas, scikit-learn, NetworkX, and tqdm.

---

## Data

The water-quality dataset is expected to be split into 20 CSV files to avoid large single-file uploads:

```text
data/final_data01.csv
data/final_data02.csv
...
data/final_data20.csv
data/small_edge.csv
```

The `final_data*.csv` files contain preprocessed water-quality observations, and `small_edge.csv` contains hydrological adjacency information between observation points.

The train, validation, and test splits are generated inside the code. No separate `train.csv`, `val.csv`, `test.csv`, or graph dataset files are required.

---

## Training

First, check whether the data files can be loaded and graph snapshots can be constructed:

```bash
python experiment/00_prepare_graph_dataset.py
```

Then train the proposed Graph Temporal Convolutional AutoEncoder (GTCAE):

```bash
python experiment/01_train_gtcae.py
```

The chronological split is:

```text
Training:   2014-2022
Validation: 2023
Test:       2024
```

Output directories such as `experiment/models/`, `experiment/logs/`, and `experiment/results/` are created automatically during execution.

---

## Inference

After training, run inference with the saved checkpoint:

```bash
python experiment/02_infer_gtcae.py
```

Inference outputs are saved automatically under the experiment output directory.

---

## Model Overview

The proposed framework consists of three main phases:

1. Spatio-Temporal Watershed Graph (STWG) construction from observation points, hydrological adjacency, and temporal observations.
2. Risk-aware representation learning using GTCAE, which combines spatial graph convolution and temporal convolutional encoding-decoding.
3. WQRI calculation by standardizing and aggregating learned latent representations into a scalar water-quality risk index.

---

## Reproducibility Notes

- The random seed is fixed to `42` for NumPy, PyTorch, and CUDA.
- The train, validation, and test periods are generated chronologically in the code.
- Large intermediate files such as checkpoints, logs, and inference results are not tracked by Git.
- Data files are expected to be encoded in UTF-8.

---

## Citation

If you use this repository or refer to this work, please cite it as:

Byeongkeun Kwon*, Dasom Seong*, Jiyun Park, Hyeonjun Hwang†, and Suhyeon Kim†.
**Water Quality Risk Index Estimation using Graph Neural Networks for Spatio-temporal Watershed Systems**.
Submitted manuscript.

* Equal contribution.
† Co-corresponding authors.

```bibtex
@article{wqri,
  title   = {Water Quality Risk Index Estimation using Graph Neural Networks for Spatio-temporal Watershed Systems},
  author  = {Byeongkeun Kwon and Dasom Seong and Jiyun Park and Hyeonjun Hwang and Suhyeon Kim},
  journal = {},
  note    = {Submitted manuscript. Byeongkeun Kwon and Dasom Seong contributed equally. Hyeonjun Hwang and Suhyeon Kim are co-corresponding authors.},
  year    = {}
}
```

---

## License

Released under the MIT License. See `LICENSE` for details.

---

## Contact

For questions and collaborations, please open an issue or contact the authors.

- Byeongkeun Kwon: house9895@knu.ac.kr 
