import numpy as np
import tensorflow as tf
import gpflow
from gpflow.utilities import print_summary
import argparse
import yaml
import os
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DEFAULT_M_CANDIDATES = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]

HF_REPO = "OccaMLab/bayesian-benchmarks"


def load_datasets_config():
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'datasets.yaml')
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f) or {}


def get_dataset_defaults(dataset_name, datasets_config):
    for entry in datasets_config.get('regression', []):
        if entry.get('name') == dataset_name:
            return entry
    return {}


def load_regression_data(dataset_name, seed=0, train_fraction=0.8):
    ds = load_dataset(HF_REPO, dataset_name, split="train")
    X = np.array(ds["features"], dtype=np.float64)
    y = np.array(ds["target"], dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    if y.shape[1] > 1:
        raise ValueError(f"Dataset {dataset_name} has {y.shape[1]} targets — not a scalar regression task.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_fraction, random_state=seed
    )
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)
    X_train = x_scaler.transform(X_train)
    X_test = x_scaler.transform(X_test)
    y_train = y_scaler.transform(y_train)
    y_test = y_scaler.transform(y_test)
    return X_train, y_train, X_test, y_test

def load_grids_config():
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'grids.yaml')
    if not os.path.exists(config_path):
        print("Warning: configs/grids.yaml not found. Using default M candidates.")
        return {}

    with open(config_path, 'r') as file:
        return yaml.safe_load(file) or {}


def build_grid_candidates(range_cfg):
    if not range_cfg:
        return []

    start = int(range_cfg['start'])
    stop = int(range_cfg['stop'])
    increment = int(range_cfg['increment'])
    return list(range(start, stop + 1, increment))


def resolve_m_candidates(dataset_name, n_train, grids_config):
    dataset_cfg = grids_config.get('M_candidates', {}).get(dataset_name, {})
    m_candidates = build_grid_candidates(dataset_cfg)

    if not m_candidates:
        m_candidates = list(DEFAULT_M_CANDIDATES)
        print(f"No M grid configured for {dataset_name}. Using defaults.")
    else:
        print(f"Loaded M search grid for {dataset_name} from configs/grids.yaml")

    m_candidates = sorted({m for m in m_candidates if 0 < m <= n_train})
    return m_candidates

def run_exact_gpr(X_train, y_train, X_test, y_test):
    print(f"Training Exact GPR on N={X_train.shape[0]} points...")
    
    D = X_train.shape[1]
    kernel = gpflow.kernels.SquaredExponential(lengthscales=np.ones(D))
    model = gpflow.models.GPR(data=(X_train, y_train), kernel=kernel, mean_function=None)
    
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=1000))
    
    # Evaluate
    f_mean, f_var = model.predict_f(X_test)
    rmse = np.sqrt(np.mean((f_mean - y_test) ** 2))
    nlpd = -np.mean(model.likelihood.predict_log_density(X_test, f_mean, f_var, y_test))
    
    return rmse, nlpd

def run_sgpr(X_train, y_train, X_test, y_test, M):
    print(f"Training SGPR with M={M}...")
    
    # Initialize inducing points (random subset)
    if M >= X_train.shape[0]:
        Z = X_train.copy()
    else:
        indices = np.random.choice(X_train.shape[0], M, replace=False)
        Z = X_train[indices].copy()
        
    D = X_train.shape[1]
    kernel = gpflow.kernels.SquaredExponential(lengthscales=np.ones(D))
    model = gpflow.models.SGPR(data=(X_train, y_train), kernel=kernel, inducing_variable=Z)
    
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=1000))
    
    # Evaluate
    f_mean, f_var = model.predict_f(X_test)
    rmse = np.sqrt(np.mean((f_mean - y_test) ** 2))
    nlpd = -np.mean(model.likelihood.predict_log_density(X_test, f_mean, f_var, y_test))
    
    return rmse, nlpd

def select_inducing_points_greedy(X, num_inducing, kernel):
    Kff_diag = kernel(X, full_cov=False)
    index = [int(np.argmax(Kff_diag))]

    for _ in range(1, num_inducing):
        Z = X[index, :]
        M = Z.shape[0]
        Kzz = kernel(Z) + 1e-6 * tf.eye(M, dtype=gpflow.default_float())
        Kzx = kernel(Z, X)
        Lu = tf.linalg.cholesky(Kzz)
        Luinv_Kuf = tf.linalg.triangular_solve(Lu, Kzx, lower=True)
        Qff = tf.linalg.matmul(Luinv_Kuf, Luinv_Kuf, transpose_a=True)
        var = Kff_diag - tf.linalg.diag_part(Qff)
        index.append(int(np.argmax(var)))

    return X[index, :]

def run_sgpr_greedy(X_train, y_train, X_test, y_test, M):
    print(f"Training SGPR (greedy Z, frozen) with M={M}...")

    if M >= X_train.shape[0]:
        Z = X_train.copy()
    else:
        D = X_train.shape[1]
        kernel_init = gpflow.kernels.SquaredExponential(lengthscales=np.ones(D))
        Z = select_inducing_points_greedy(X_train, M, kernel_init)

    D = X_train.shape[1]
    kernel = gpflow.kernels.SquaredExponential(lengthscales=np.ones(D))
    model = gpflow.models.SGPR(data=(X_train, y_train), kernel=kernel, inducing_variable=Z)
    gpflow.set_trainable(model.inducing_variable, False)

    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=1000))

    f_mean, f_var = model.predict_f(X_test)
    rmse = np.sqrt(np.mean((f_mean - y_test) ** 2))
    nlpd = -np.mean(model.likelihood.predict_log_density(X_test, f_mean, f_var, y_test))

    return rmse, nlpd

def main():
    parser = argparse.ArgumentParser(description='Find M for SGPR based on RMSE threshold.')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--threshold_pct_rmse', type=float, default=None, help='RMSE threshold pct (overrides config)')
    parser.add_argument('--threshold_pct_nlpd', type=float, default=None, help='NLPD threshold pct (overrides config)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (overrides config)')
    parser.add_argument('--method', type=str, default='train', choices=['train', 'greedy'],
                        help='train: optimise Z locations | greedy: freeze Z from conditional variance')
    
    args = parser.parse_args()
    
    datasets_config = load_datasets_config()
    grids_config = load_grids_config()
    defaults = get_dataset_defaults(args.dataset, datasets_config)

    seed = args.seed if args.seed is not None else defaults.get('seed', 0)
    threshold_pct_rmse = args.threshold_pct_rmse if args.threshold_pct_rmse is not None else defaults.get('threshold_pct_rmse', 5.0)
    threshold_pct_nlpd = args.threshold_pct_nlpd if args.threshold_pct_nlpd is not None else defaults.get('threshold_pct_nlpd', 10.0)

    np.random.seed(seed)
    tf.random.set_seed(seed)

    print(f"Using seed={seed}, threshold_pct_rmse={threshold_pct_rmse}, threshold_pct_nlpd={threshold_pct_nlpd}, method={args.method}")
    X_train, y_train, X_test, y_test = load_regression_data(args.dataset, seed=seed)
    print(f"Data loaded. Train N={X_train.shape[0]}, Test N={X_test.shape[0]}")

    # 1. Compute Trivial Metrics
    y_pred_trivial = np.full(y_test.shape, np.mean(y_train))
    y_var_trivial = np.var(y_train)
    rmse_trivial = np.sqrt(np.mean((y_test - y_pred_trivial) ** 2))
    nlpd_trivial = 0.5 * np.log(2 * np.pi * y_var_trivial) + 0.5 * np.mean((y_test - y_pred_trivial) ** 2) / y_var_trivial
    
    print(f"Trivial RMSE: {rmse_trivial:.4f}, NLPD: {nlpd_trivial:.4f}")
    
    # 2. Compute Exact GPR Metrics
    try:
        rmse_exact, nlpd_exact = run_exact_gpr(X_train, y_train, X_test, y_test)
        print(f"Exact GPR RMSE: {rmse_exact:.4f}, NLPD: {nlpd_exact:.4f}")
    except Exception as e:
        print(f"Exact GPR failed (likely OOM): {e}")
        print("Attempting SVGP with M=N (if N is not too huge)...")
        return

    # 3. Calculate Threshold
    # Threshold = Exact + pct * |Trivial - Exact|
    rmse_threshold = rmse_exact + (threshold_pct_rmse / 100.0) * np.abs(rmse_trivial - rmse_exact)
    nlpd_threshold = nlpd_exact + (threshold_pct_nlpd / 100.0) * np.abs(nlpd_trivial - nlpd_exact)
    
    print(f"RMSE Threshold ({threshold_pct_rmse}%): {rmse_threshold:.4f}")
    print(f"NLPD Threshold ({threshold_pct_nlpd}%): {nlpd_threshold:.4f}")
    
    # 4. Tune M for SGPR
    print("\nTuning M for SGPR...")
    
    N = X_train.shape[0]
    m_candidates = resolve_m_candidates(args.dataset, N, grids_config)
    
    run_fn = run_sgpr_greedy if args.method == 'greedy' else run_sgpr

    found_m_rmse = None
    found_m_nlpd = None
    prev_m = None
    
    for M in m_candidates:
        rmse_sgpr, nlpd_sgpr = run_fn(X_train, y_train, X_test, y_test, M)
        print(f"M={M}: RMSE={rmse_sgpr:.4f}, NLPD={nlpd_sgpr:.4f}")
        
        if found_m_rmse is None and rmse_sgpr <= rmse_threshold:
            found_m_rmse = M
            print(f"   -> RMSE threshold met at M={M}")
            
        if found_m_nlpd is None and nlpd_sgpr <= nlpd_threshold:
            found_m_nlpd = M
            print(f"   -> NLPD threshold met at M={M}")
            
        if found_m_rmse is not None and found_m_nlpd is not None:
            break
        prev_m = M

    # 5. Refinement: search between prev_m and found_m with step 1
    coarse_m = max(found_m_rmse or 0, found_m_nlpd or 0)
    lo = prev_m + 1 if prev_m is not None and prev_m < coarse_m else 1
    if coarse_m > lo:
        print(f"\nRefining search in [{lo}, {coarse_m})...")
        found_m_rmse_r = None
        found_m_nlpd_r = None
        for M in range(lo, coarse_m):
            rmse_sgpr, nlpd_sgpr = run_fn(X_train, y_train, X_test, y_test, M)
            print(f"M={M}: RMSE={rmse_sgpr:.4f}, NLPD={nlpd_sgpr:.4f}")
            if found_m_rmse_r is None and rmse_sgpr <= rmse_threshold:
                found_m_rmse_r = M
                print(f"   -> RMSE refined at M={M}")
            if found_m_nlpd_r is None and nlpd_sgpr <= nlpd_threshold:
                found_m_nlpd_r = M
                print(f"   -> NLPD refined at M={M}")
            if found_m_rmse_r is not None and found_m_nlpd_r is not None:
                break
        if found_m_rmse_r is not None:
            found_m_rmse = found_m_rmse_r
        if found_m_nlpd_r is not None:
            found_m_nlpd = found_m_nlpd_r

    # 6. Save results
    results = {
        'dataset': args.dataset,
        'seed': seed,
        'threshold_pct_rmse': threshold_pct_rmse,
        'threshold_pct_nlpd': threshold_pct_nlpd,
        'method': args.method,
        'n_train': int(X_train.shape[0]),
        'n_test': int(X_test.shape[0]),
        'trivial': {'rmse': float(rmse_trivial), 'nlpd': float(nlpd_trivial)},
        'exact_gpr': {'rmse': float(rmse_exact), 'nlpd': float(nlpd_exact)},
        'thresholds': {'rmse': float(rmse_threshold), 'nlpd': float(nlpd_threshold)},
        'optimal_m': {
            'rmse': found_m_rmse,
            'nlpd': found_m_nlpd,
        },
    }

    out_dir = os.path.join(os.path.dirname(__file__), 'optimal_settings')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{args.dataset}_seed{seed}_{args.method}.yaml')
    with open(out_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)

    print("\nResults:")
    if found_m_rmse:
        print(f"Smallest M for RMSE threshold: {found_m_rmse}")
    else:
        print("Could not find M for RMSE threshold.")
    if found_m_nlpd:
        print(f"Smallest M for NLPD threshold: {found_m_nlpd}")
    else:
        print("Could not find M for NLPD threshold.")
    print(f"Saved to {out_path}")

if __name__ == '__main__':
    main()
