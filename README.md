# GPS Benchmark Settings

Find the smallest number of inducing points **M** that stays within a given percentage of the full-model performance.

- **Regression** (`regression_find_m_for_threshold.py`): SGPR vs Exact GPR, metrics are RMSE and NLPD.
- **Classification** (`classification_find_m_for_threshold.py`): sparse SVGP vs full SVGP (M=N), metrics are ERRP and NLPD.

## Setup

```bash
conda create -n gpsenv python=3.10 -y
conda activate gpsenv
pip install -r requirements.txt
pip install datasets
```

## Regression

```bash
# Default: uses seed and threshold_pct from configs/datasets.yaml
python regression_find_m_for_threshold.py --dataset concrete

# Override threshold and seed
python regression_find_m_for_threshold.py --dataset bike --threshold_pct 2.5 --seed 42

# Greedy inducing point selection (frozen Z, no optimisation of locations)
python regression_find_m_for_threshold.py --dataset concrete --method greedy
```

## Classification

```bash
# Diabetes (binary, loaded from data/diabetes.csv)
python classification_find_m_for_threshold.py --dataset diabetes

# MNIST (10-class, loaded via tf.keras.datasets)
python classification_find_m_for_threshold.py --dataset MNIST

# Greedy inducing point selection
python classification_find_m_for_threshold.py --dataset diabetes --method greedy
```

### Arguments (both scripts)

| Argument | Description | Default |
|---|---|---|
| `--dataset` | Dataset name | *required* |
| `--threshold_pct` | Allowed degradation as % of the trivial–full gap | from config (5.0) |
| `--seed` | Random seed for train/test split | from config (0) |
| `--method` | `train` (optimise Z) or `greedy` (conditional variance, frozen Z) | `train` |

If `--seed` or `--threshold_pct` are omitted, values are read from `configs/datasets.yaml`.

## Config Files

- **`configs/datasets.yaml`** — lists each dataset with its default `seed` and `threshold_pct`.
- **`configs/grids.yaml`** — integer M candidate grids (`start`/`stop`/`increment`) and LR candidate grids per dataset.

If a dataset has no M grid entry, the script falls back to `[10, 20, 50, 100, 200, 500, 1000, 2000, 5000]`.

## Search Strategy

1. **Coarse pass** — evaluates at each M in the grid until both metric thresholds are met.
2. **Refinement pass** — searches with step 1 between the last failing M and the first passing M to find the exact smallest M.

## Output

Results are saved to `optimal_settings/<dataset>_seed<N>_<method>.yaml`.
