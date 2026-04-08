"""
Full pipeline: find optimal M (threshold search) then optimal LR (minibatch SVGP)
for one or more datasets. Saves a consolidated summary to results/{dataset}/.

Usage:
  python run_pipeline.py --datasets concrete bike
  python run_pipeline.py --all
  python run_pipeline.py --datasets concrete --method greedy --skip_lr
"""

import argparse
import os
import sys
import subprocess
import yaml
import numpy as np


def load_datasets_config():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'datasets.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f) or {}


def get_dataset_entry(dataset_name, datasets_config):
    for task in ('regression', 'classification'):
        for entry in datasets_config.get(task, []):
            if entry.get('name') == dataset_name:
                return task, entry
    raise ValueError(f"Dataset '{dataset_name}' not found in configs/datasets.yaml")


def get_all_datasets(datasets_config):
    result = []
    for task in ('regression', 'classification'):
        for entry in datasets_config.get(task, []):
            result.append((entry['name'], task, entry))
    return result


def run_script(script_path, extra_args):
    cmd = [sys.executable, script_path] + extra_args
    print(f"\n  $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    if proc.returncode != 0:
        raise RuntimeError(f"Script exited with return code {proc.returncode}")


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def avg_over_folds(fold_dicts, keys):
    """Return {key: mean} averaged over a list of dicts."""
    result = {}
    for key in keys:
        values = [d[key] for d in fold_dicts if d is not None and d.get(key) is not None]
        result[key] = round(float(np.mean(values)), 6) if values else None
    return result


def build_summary(threshold_data, lr_data, task):
    per_fold = threshold_data.get('per_fold', [])

    if task == 'regression':
        metric_keys = ['rmse', 'nlpd']
        trivial_avg = avg_over_folds([f['trivial'] for f in per_fold], metric_keys)
        full_model_avg = avg_over_folds([f['exact_gpr'] for f in per_fold], metric_keys)
        threshold_avg = avg_over_folds([f['thresholds'] for f in per_fold], metric_keys)
        threshold_pct_keys = ['threshold_pct_rmse', 'threshold_pct_nlpd']
    else:
        metric_keys = ['errp', 'nlpd']
        trivial_avg = avg_over_folds([f['trivial'] for f in per_fold], metric_keys)
        full_model_avg = avg_over_folds([f['full_svgp'] for f in per_fold], metric_keys)
        threshold_avg = avg_over_folds([f['thresholds'] for f in per_fold], metric_keys)
        threshold_pct_keys = ['threshold_pct_errp', 'threshold_pct_nlpd']

    optimal_m = threshold_data.get('optimal_m', {})
    m_values = [v for v in optimal_m.values() if v is not None]
    recommended_m = int(max(m_values)) if m_values else None

    summary = {
        'dataset': threshold_data['dataset'],
        'task': task,
        'seed': threshold_data['seed'],
        'n_folds': threshold_data.get('n_folds', 1),
        'method': threshold_data['method'],
        'noise_model_baseline': trivial_avg,
        'full_model_baseline': full_model_avg,
        'threshold_settings': {k: threshold_data[k] for k in threshold_pct_keys if k in threshold_data},
        'thresholds_avg': threshold_avg,
        'optimal_m': {**{k: v for k, v in optimal_m.items()}, 'recommended': recommended_m},
    }

    if lr_data is not None:
        summary['optimal_lr'] = lr_data.get('best_lr')
        summary['best_metrics_at_optimal_lr'] = lr_data.get('best_metrics')
    else:
        summary['optimal_lr'] = None
        summary['best_metrics_at_optimal_lr'] = None

    return summary


def run_dataset(dataset_name, task, entry, args):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    seed = entry.get('seed', 0)
    n_folds = entry.get('n_folds', 5)
    method = args.method

    common_args = [
        '--dataset', dataset_name,
        '--method', method,
        '--seed', str(seed),
        '--n_folds', str(n_folds),
    ]

    # Step 1: Find optimal M
    threshold_script = os.path.join(
        script_dir,
        'regression_find_m_for_threshold.py' if task == 'regression'
        else 'classification_find_m_for_threshold.py'
    )
    print(f"\n[1/2] Finding optimal M for '{dataset_name}'...")
    run_script(threshold_script, common_args)

    # Step 2: Find optimal LR (optional)
    lr_data = None
    if not args.skip_lr:
        lr_script = os.path.join(script_dir, 'find_optimal_lr.py')
        # Only run if LR grid exists for this dataset
        grids_config_path = os.path.join(script_dir, 'configs', 'grids.yaml')
        with open(grids_config_path) as f:
            grids = yaml.safe_load(f) or {}
        if dataset_name in grids.get('LR_candidates', {}):
            print(f"\n[2/2] Finding optimal LR for '{dataset_name}'...")
            run_script(lr_script, common_args)
        else:
            print(f"\n[2/2] Skipping LR search — no LR grid configured for '{dataset_name}'.")

    # Step 3: Load results
    settings_dir = os.path.join(script_dir, 'optimal_settings')
    threshold_path = os.path.join(settings_dir, f'{dataset_name}_seed{seed}_{method}.yaml')
    lr_path = os.path.join(settings_dir, f'{dataset_name}_seed{seed}_{method}_lr.yaml')

    threshold_data = load_yaml(threshold_path)
    if os.path.exists(lr_path):
        lr_data = load_yaml(lr_path)

    # Step 4: Build and save consolidated summary
    summary = build_summary(threshold_data, lr_data, task)

    out_dir = os.path.join(script_dir, 'results', dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'summary_seed{seed}_{method}.yaml')
    with open(out_path, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)

    print(f"\n  Summary saved to {out_path}")
    _print_summary(summary)
    return summary


def _print_summary(s):
    print(f"\n  {'─'*40}")
    print(f"  Dataset : {s['dataset']} ({s['task']})")
    print(f"  Folds   : {s['n_folds']}")
    print(f"  Method  : {s['method']}")
    print(f"\n  Noise model baseline : {s['noise_model_baseline']}")
    print(f"  Full model baseline  : {s['full_model_baseline']}")
    print(f"  Thresholds (avg)     : {s['thresholds_avg']}")
    print(f"\n  Optimal M            : {s['optimal_m']}")
    if s['optimal_lr'] is not None:
        print(f"  Optimal LR           : {s['optimal_lr']}")
        print(f"  Best metrics @ opt LR: {s['best_metrics_at_optimal_lr']}")
    print(f"  {'─'*40}")


def main():
    parser = argparse.ArgumentParser(
        description='Run the full GP benchmark pipeline for one or more datasets.'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--datasets', nargs='+', metavar='DATASET',
                       help='One or more dataset names')
    group.add_argument('--all', action='store_true',
                       help='Run all datasets listed in configs/datasets.yaml')
    parser.add_argument('--method', type=str, default='train',
                        choices=['train', 'greedy'],
                        help='Inducing point method (default: train)')
    parser.add_argument('--skip_lr', action='store_true',
                        help='Skip the LR search step')
    args = parser.parse_args()

    datasets_config = load_datasets_config()

    if args.all:
        datasets_to_run = get_all_datasets(datasets_config)
    else:
        datasets_to_run = []
        for name in args.datasets:
            task, entry = get_dataset_entry(name, datasets_config)
            datasets_to_run.append((name, task, entry))

    print(f"Pipeline: {[d[0] for d in datasets_to_run]}")
    print(f"Method: {args.method}, Skip LR: {args.skip_lr}")

    failed = []
    for dataset_name, task, entry in datasets_to_run:
        print(f"\n{'='*60}")
        print(f"  DATASET: {dataset_name}  ({task})")
        print(f"{'='*60}")
        try:
            run_dataset(dataset_name, task, entry, args)
        except Exception as e:
            print(f"\n  ERROR: {dataset_name} failed — {e}")
            failed.append(dataset_name)

    n_total = len(datasets_to_run)
    n_ok = n_total - len(failed)
    print(f"\n{'='*60}")
    print(f"Done. {n_ok}/{n_total} datasets succeeded.")
    if failed:
        print(f"Failed: {failed}")


if __name__ == '__main__':
    main()
