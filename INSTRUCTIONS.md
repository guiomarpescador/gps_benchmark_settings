# Instructions

## Setup

```bash
conda create -n gpsenv python=3.10 -y
conda activate gpsenv
pip install -r requirements.txt
pip install datasets
```

## 1. Finding Optimal M

### Regression

```bash
# Default: uses seed, thresholds, and n_folds from configs/datasets.yaml
python regression_find_m_for_threshold.py --dataset concrete

# Override thresholds and seed
python regression_find_m_for_threshold.py --dataset bike --threshold_pct_rmse 2.5 --threshold_pct_nlpd 5.0 --seed 42

# Greedy inducing point selection (frozen Z, no optimisation of locations)
python regression_find_m_for_threshold.py --dataset concrete --method greedy

# Single split (no CV)
python regression_find_m_for_threshold.py --dataset concrete --n_folds 1
```

### Classification

```bash
# Diabetes (binary, loaded from data/diabetes.csv)
python classification_find_m_for_threshold.py --dataset diabetes

# MNIST (10-class, loaded via tf.keras.datasets)
python classification_find_m_for_threshold.py --dataset MNIST

# Greedy inducing point selection
python classification_find_m_for_threshold.py --dataset diabetes --method greedy
```

### Arguments

**Regression** (`regression_find_m_for_threshold.py`):

| Argument | Description | Default |
|---|---|---|
| `--dataset` | Dataset name | *required* |
| `--threshold_pct_rmse` | Allowed RMSE degradation as % of the trivial–full gap | from config (5.0) |
| `--threshold_pct_nlpd` | Allowed NLPD degradation as % of the trivial–full gap | from config (10.0) |
| `--seed` | Random seed | from config (0) |
| `--n_folds` | Number of CV folds (1 = single split) | from config (5) |
| `--method` | `train` (optimise Z) or `greedy` (conditional variance, frozen Z) | `train` |

**Classification** (`classification_find_m_for_threshold.py`):

| Argument | Description | Default |
|---|---|---|
| `--dataset` | Dataset name | *required* |
| `--threshold_pct_errp` | Allowed ERRP degradation as % of the trivial–full gap | from config (5.0) |
| `--threshold_pct_nlpd` | Allowed NLPD degradation as % of the trivial–full gap | from config (10.0) |
| `--seed` | Random seed | from config (0) |
| `--n_folds` | Number of CV folds (1 = single split) | from config (5) |
| `--method` | `train` (optimise Z) or `greedy` (conditional variance, frozen Z) | `train` |

If `--seed`, threshold, or `--n_folds` arguments are omitted, values are read from `configs/datasets.yaml`.

### Search Strategy

1. **Coarse pass** — evaluates at each M in the grid until both metric thresholds are met.
2. **Refinement pass** — searches with step 1 between the last failing M and the first passing M to find the exact smallest M.
3. **Cross-validation** — when `n_folds > 1`, steps 1–2 run independently on each fold. The final M is the **max** across folds (conservative: guarantees threshold met on every fold).

## 2. Learning Rate Search

Once optimal M is found, find the best learning rate for minibatch SVGP:

```bash
# Uses batch_size from config and LR grid from grids.yaml
python find_optimal_lr.py --dataset concrete --method train

# Override batch size or epochs
python find_optimal_lr.py --dataset concrete --batch_size 32 --epochs 200
```

| Argument | Description | Default |
|---|---|---|
| `--dataset` | Dataset name | *required* |
| `--method` | `train` or `greedy` (must match threshold run) | `train` |
| `--seed` | Random seed | from config (0) |
| `--n_folds` | Number of CV folds (1 = single split) | from config (5) |
| `--batch_size` | Minibatch size for Adam | from config |
| `--epochs` | Number of epochs (passes over the dataset) | 100 |
| `--metric` | Metric to optimise (`rmse`, `nlpd`, `errp`) | `nlpd` (regression) / `errp` (classification) |

## Config Files

- **`configs/datasets.yaml`** — lists each dataset with its default `seed`, `n_folds`, `batch_size`, and per-metric thresholds (`threshold_pct_rmse`/`threshold_pct_nlpd` for regression, `threshold_pct_errp`/`threshold_pct_nlpd` for classification).
- **`configs/grids.yaml`** — integer M candidate grids (`start`/`stop`/`increment`) and LR candidate grids per dataset.

If a dataset has no M grid entry, the script falls back to `[10, 20, 50, 100, 200, 500, 1000, 2000, 5000]`.

## Output

- Threshold results: `optimal_settings/<dataset>_seed<N>_<method>.yaml`
- LR search results: `optimal_settings/<dataset>_seed<N>_<method>_lr.yaml`

## Full Pipeline

To run everything (M search + LR search) for one or more datasets in one command:

```bash
# Single dataset
python run_pipeline.py --datasets concrete

# Multiple datasets
python run_pipeline.py --datasets concrete bike kin8nm

# All datasets in configs/datasets.yaml
python run_pipeline.py --all

# Greedy method, skip LR search
python run_pipeline.py --datasets concrete --method greedy --skip_lr
```

Results are saved to `results/{dataset}/summary_seed{N}_{method}.yaml` and include:

- `noise_model_baseline` — trivial predictor (training mean / majority class), averaged across folds
- `full_model_baseline` — exact GPR / full SVGP (M=N), averaged across folds
- `threshold_settings` — the percentage thresholds used per metric
- `thresholds_avg` — the actual threshold values, averaged across folds
- `optimal_m` — smallest M per metric + `recommended` (max across metrics)
- `optimal_lr` — best learning rate for minibatch SVGP
- `best_metrics_at_optimal_lr` — fold-averaged metrics at the optimal LR

The LR step is skipped automatically if no LR grid is configured for the dataset in `configs/grids.yaml`.

| Argument | Description | Default |
|---|---|---|
| `--datasets` | One or more dataset names | *required (or --all)* |
| `--all` | Run all datasets in config | — |
| `--method` | `train` or `greedy` | `train` |
| `--skip_lr` | Skip the LR search step | off |

---