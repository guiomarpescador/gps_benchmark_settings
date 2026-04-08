# Gaussian Processes Benchmark Settings

## Motivation

Sparse Gaussian Processes approximate the full GP by using a set of **M** inducing points. Choosing M is important: 

- **M too small** — the approximation is poor and we sacrifice performance.
- **M too large** — computation is wasted without meaningful improvement.

This repository provides tools to find the **smallest M** that stays within a given percentage of full-model performance, following the methodology of [Pescador-Barrios et al. (2025)](https://arxiv.org/abs/2408.07588).

### How it works

For each dataset, the scripts compare a sparse model against a full (exact) reference:

- **Regression**: SGPR vs Exact GPR — metrics are RMSE and NLPD.
- **Classification**: sparse SVGP (M < N) vs full SVGP (M = N) — metrics are ERRP and NLPD.

A *noise model baseline* (predicting the training mean for regression, or the majority class for classification) represents the worst-case reference — a model that has learned nothing. The threshold for each metric is defined as a percentage of the gap between this baseline and the full model:

$$M_\text{threshold} = \text{Exact} + \frac{p}{100} \times |\text{Trivial} - \text{Exact}|$$

where $p$ is the threshold percentage (default: 5% for RMSE/ERRP, 10% for NLPD). The search finds the smallest M where the sparse model meets both thresholds. It also works with cross-validation folds, averaging metrics across folds to determine the optimal M (max across folds).

## Stochastic SVGP — Learning Rate Search

Once the optimal M is determined, the next step is to configure **minibatch SVGP** for scalable training. The `find_optimal_lr.py` script sweeps a grid of learning rates using Adam, training for a fixed number of epochs, and selects the LR that minimises the chosen metric (averaged across CV folds).

This is useful when moving from the deterministic (full-batch) setting used for M selection to the stochastic (minibatch) setting used in practice.

## Citation

If you use this code, please cite:

```bibtex
@software{pescador2025gpsbenchmark,
  author    = {Pescador-Barrios, Guiomar},
  title     = {Gaussian Processes Benchmark Settings},
  year      = {2026},
  url       = {https://github.com/guiomarpescador/gps_benchmark_settings}
}
```

This work builds on ideas from:

> G. Pescador-Barrios, S. Filippi, M. van der Wilk. *Adjusting Model Size in Continual Gaussian Processes: How Big is Big Enough?* ICML 2025. [arXiv:2408.07588](https://arxiv.org/abs/2408.07588)

## Instructions

See [INSTRUCTIONS.md](INSTRUCTIONS.md) for setup, usage, arguments, and configuration details.
