# Data

Place the input files required for WQRI experiments in this folder.

## Required files

The public workflow assumes that the large preprocessed water-quality table is split into twenty UTF-8 encoded CSV chunks:

```text
data/final_data01.csv
data/final_data02.csv
data/final_data03.csv
...
data/final_data20.csv
data/small_edge.csv
```

The chunk files are loaded in sorted order and concatenated in memory by the experiment scripts. No train/validation/test CSV files are stored separately.

`small_edge.csv` is the hydrological adjacency table. It should be UTF-8 encoded and include the following columns:

```text
node1,node2
```

## Required water-quality schema

The water-quality chunks should use the following English column names:

```text
Observation_point
Observation_date
Water_temperature
DO
BOD
COD
SS
TN
TP
TOC
pH
EC
Water_temperature_scaled
DO_scaled
BOD_scaled
COD_scaled
SS_scaled
TN_scaled
TP_scaled
TOC_scaled
pH_scaled
EC_scaled
BOD_type1
COD_type1
SS_type1
TN_type1
TP_type1
TOC_type1
Water_temperature_type1
EC_type1
DO_type2
pH_type3
```

For compatibility with previously generated files, the loader also accepts `Water_temparature_scaled` and `Water_temparature_type1` and maps them to `Water_temperature_scaled` and `Water_temperature_type1` internally.

## Split policy

The chronological split is implemented inside the code:

```text
train: 2014-2022
val:   2023
test:  2024
```

## Intermediate graph datasets

The graph datasets are constructed in memory during training and inference. The repository does not save or require the following files:

```text
data/graph_dataset/train_graph.pt
data/graph_dataset/val_graph.pt
data/graph_dataset/test_graph.pt
```
