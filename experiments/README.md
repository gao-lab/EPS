# Experiments

## Environment

You should install `expression_copilot` with `dev` dependency to run experiments:

```sh
pip install "expression_copilot[dev, torch]"
```

All versions of packages used in our experiments are recorded in [`conda.yaml`](./conda.yaml). You could create conda environment based on it.

## Data

All datasets used in experiments are publicly available, please see our manuscript for download links.

## Code

Each subdir contains a single experiment:
- [`10x_slices`](./10x_slices): EPS on 10x slices (7 slices)
- [`hest_dataset`](./hest_dataset): EPS on HEST dataset (>500 slices)
- [`sc_multi_omics`](./sc_multi_omics): EPS on single-cell multi-omics datasets (CITE-seq and 10x Multiome)

Inside every subdir, there are series python files. Users could run them one by one to reproduce our results.