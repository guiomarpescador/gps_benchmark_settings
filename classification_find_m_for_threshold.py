import numpy as np
import pandas as pd
import tensorflow as tf
import gpflow
import argparse
import yaml
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DEFAULT_M_CANDIDATES = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
DEFAULT_LENGTHSCALE = 1.0
DEFAULT_VARIANCE = 1.0
DEFAULT_LEARNING_RATE = 0.01
ADAM_STEPS = 1000


def load_datasets_config():
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'datasets.yaml')
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f) or {}


def get_dataset_defaults(dataset_name, datasets_config):
    for entry in datasets_config.get('classification', []):
        if entry.get('name') == dataset_name:
            return entry
    return {}


def load_classification_data(dataset_name, seed=0, train_fraction=0.8):
    data_path = os.path.join(os.path.dirname(__file__), 'data')

    if dataset_name == "diabetes":
        df = pd.read_csv(os.path.join(data_path, "diabetes.csv"))
        X = df[
            [
                "Pregnancies",
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "BMI",
                "DiabetesPedigreeFunction",
                "Age",
            ]
        ].to_numpy(dtype=np.float64)
        y = df["Outcome"].to_numpy(dtype=np.float64).reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_fraction, random_state=seed
        )
        x_scaler = StandardScaler().fit(X_train)
        X_train = x_scaler.transform(X_train)
        X_test = x_scaler.transform(X_test)
    elif dataset_name == "MNIST":
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        X_train = X_train.reshape(-1, 784).astype(np.float64) / 255.0
        X_test = X_test.reshape(-1, 784).astype(np.float64) / 255.0
        y_train = y_train.reshape(-1, 1).astype(np.float64)
        y_test = y_test.reshape(-1, 1).astype(np.float64)
    else:
        raise ValueError(
            f"Unknown classification dataset: {dataset_name}. "
            "Supported: diabetes, MNIST"
        )

    return X_train, y_train, X_test, y_test


def get_num_classes(y):
    return len(np.unique(y))


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


# -------------------------
# Trivial baseline
# -------------------------
def calculate_trivial_metrics(y_train, y_test, num_classes):
    if num_classes == 2:
        p = np.clip(np.mean(y_train), 1e-10, 1 - 1e-10)
        y_pred = float(p > 0.5)
        errp = np.mean(y_test.flatten() != y_pred)
        nlpd = -np.mean(
            y_test.flatten() * np.log(p)
            + (1 - y_test.flatten()) * np.log(1 - p)
        )
    else:
        classes, counts = np.unique(y_train, return_counts=True)
        freqs = np.clip(counts / len(y_train), 1e-10, 1.0)
        class_prob = {int(c): f for c, f in zip(classes, freqs)}
        majority_class = classes[np.argmax(counts)]
        errp = np.mean(y_test.flatten() != majority_class)
        log_probs = [np.log(class_prob.get(int(y), 1e-10)) for y in y_test.flatten()]
        nlpd = -np.mean(log_probs)

    return errp, nlpd


# -------------------------
# SVGP evaluation helper
# -------------------------
def evaluate_svgp(model, X_test, Y_test, num_classes):
    if num_classes == 2:
        f_mean, _ = model.predict_f(X_test)
        p = model.likelihood.invlink(f_mean).numpy()
        y_pred = (p > 0.5).astype(np.float64)
        errp = np.mean(Y_test.flatten() != y_pred.flatten())
    else:
        f_mean, _ = model.predict_f(X_test)
        p = tf.nn.softmax(f_mean).numpy()
        y_pred = np.argmax(p, axis=1)
        errp = np.mean(Y_test.flatten() != y_pred)

    nlpd = -tf.reduce_mean(model.predict_log_density((X_test, Y_test))).numpy()
    return errp, nlpd


# -------------------------
# Full-batch SVGP (Adam -> Scipy)
# -------------------------
def run_svgp(X_train, Y_train, X_test, Y_test, M, num_classes,
             learning_rate=DEFAULT_LEARNING_RATE):
    print(f"Training SVGP with M={M}...")

    if num_classes == 2:
        likelihood = gpflow.likelihoods.Bernoulli()
        num_latent = 1
    else:
        likelihood = gpflow.likelihoods.Softmax(num_classes)
        num_latent = num_classes

    if M >= X_train.shape[0]:
        Z = X_train.copy()
        print("  Using full inducing set (M = N)")
    else:
        idx = np.random.choice(X_train.shape[0], M, replace=False)
        Z = X_train[idx]

    D = X_train.shape[1]
    kernel = gpflow.kernels.SquaredExponential(
        lengthscales=np.ones(D),
        variance=DEFAULT_VARIANCE,
    )

    model = gpflow.models.SVGP(
        kernel=kernel,
        likelihood=likelihood,
        inducing_variable=Z,
        num_latent_gps=num_latent,
        whiten=True,
    )

    # Stage 1: Adam
    adam = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)

    @tf.function
    def adam_step():
        with tf.GradientTape() as tape:
            loss = model.training_loss((X_train, Y_train))
        grads = tape.gradient(loss, model.trainable_variables)
        adam.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    for step in range(ADAM_STEPS):
        loss = adam_step()
        if step % 500 == 0:
            tf.print(f"  [Adam] step {step}, loss:", loss)

    # Stage 2: Scipy
    scipy_opt = gpflow.optimizers.Scipy()
    scipy_opt.minimize(
        lambda: model.training_loss((X_train, Y_train)),
        model.trainable_variables,
    )

    errp, nlpd = evaluate_svgp(model, X_test, Y_test, num_classes)
    return errp, nlpd


# -------------------------
# Greedy inducing-point selection
# -------------------------
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


def run_svgp_greedy(X_train, Y_train, X_test, Y_test, M, num_classes,
                    learning_rate=DEFAULT_LEARNING_RATE):
    print(f"Training SVGP (greedy Z, frozen) with M={M}...")

    if num_classes == 2:
        likelihood = gpflow.likelihoods.Bernoulli()
        num_latent = 1
    else:
        likelihood = gpflow.likelihoods.Softmax(num_classes)
        num_latent = num_classes

    if M >= X_train.shape[0]:
        Z = X_train.copy()
    else:
        D = X_train.shape[1]
        kernel_init = gpflow.kernels.SquaredExponential(
            lengthscales=np.ones(D),
            variance=DEFAULT_VARIANCE,
        )
        Z = select_inducing_points_greedy(X_train, M, kernel_init)

    D = X_train.shape[1]
    kernel = gpflow.kernels.SquaredExponential(
        lengthscales=np.ones(D),
        variance=DEFAULT_VARIANCE,
    )

    model = gpflow.models.SVGP(
        kernel=kernel,
        likelihood=likelihood,
        inducing_variable=Z,
        num_latent_gps=num_latent,
        whiten=True,
    )
    gpflow.set_trainable(model.inducing_variable, False)

    # Stage 1: Adam
    adam = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)

    @tf.function
    def adam_step():
        with tf.GradientTape() as tape:
            loss = model.training_loss((X_train, Y_train))
        grads = tape.gradient(loss, model.trainable_variables)
        adam.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    for step in range(ADAM_STEPS):
        loss = adam_step()
        if step % 500 == 0:
            tf.print(f"  [Adam] step {step}, loss:", loss)

    # Stage 2: Scipy
    scipy_opt = gpflow.optimizers.Scipy()
    scipy_opt.minimize(
        lambda: model.training_loss((X_train, Y_train)),
        model.trainable_variables,
    )

    errp, nlpd = evaluate_svgp(model, X_test, Y_test, num_classes)
    return errp, nlpd


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Find M for SVGP classification based on ERRP/NLPD threshold.'
    )
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--threshold_pct', type=float, default=None,
                        help='Percentage threshold (overrides config)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config)')
    parser.add_argument('--method', type=str, default='train',
                        choices=['train', 'greedy'],
                        help='train: optimise Z | greedy: freeze Z from conditional variance')
    args = parser.parse_args()

    datasets_config = load_datasets_config()
    grids_config = load_grids_config()
    defaults = get_dataset_defaults(args.dataset, datasets_config)

    seed = args.seed if args.seed is not None else defaults.get('seed', 0)
    threshold_pct = (args.threshold_pct if args.threshold_pct is not None
                     else defaults.get('threshold_pct', 5.0))

    np.random.seed(seed)
    tf.random.set_seed(seed)

    print(f"Using seed={seed}, threshold_pct={threshold_pct}, method={args.method}")

    X_train, y_train, X_test, y_test = load_classification_data(
        args.dataset, seed=seed
    )
    num_classes = get_num_classes(np.concatenate([y_train, y_test]))
    N = X_train.shape[0]

    print(f"Data loaded. Train N={N}, Test N={X_test.shape[0]}, "
          f"num_classes={num_classes}")

    # 1. Trivial baseline
    errp_trivial, nlpd_trivial = calculate_trivial_metrics(
        y_train, y_test, num_classes
    )
    print(f"Trivial ERRP: {errp_trivial:.4f}, NLPD: {nlpd_trivial:.4f}")

    # 2. Full SVGP (M = N) as "exact" reference
    try:
        errp_full, nlpd_full = run_svgp(
            X_train, y_train, X_test, y_test, N, num_classes
        )
        print(f"Full SVGP ERRP: {errp_full:.4f}, NLPD: {nlpd_full:.4f}")
    except Exception as e:
        print(f"Full SVGP failed (likely OOM): {e}")
        return

    # 3. Thresholds
    errp_threshold = errp_full + (threshold_pct / 100.0) * abs(errp_trivial - errp_full)
    nlpd_threshold = nlpd_full + (threshold_pct / 100.0) * abs(nlpd_trivial - nlpd_full)

    print(f"ERRP Threshold ({threshold_pct}%): {errp_threshold:.4f}")
    print(f"NLPD Threshold ({threshold_pct}%): {nlpd_threshold:.4f}")

    # 4. Tune M
    print("\nTuning M for SVGP...")

    m_candidates = resolve_m_candidates(args.dataset, N, grids_config)

    run_fn = run_svgp_greedy if args.method == 'greedy' else run_svgp

    found_m_errp = None
    found_m_nlpd = None
    prev_m = None

    for M in m_candidates:
        errp, nlpd = run_fn(
            X_train, y_train, X_test, y_test, M, num_classes
        )
        print(f"M={M}: ERRP={errp:.4f}, NLPD={nlpd:.4f}")

        if found_m_errp is None and errp <= errp_threshold:
            found_m_errp = M
            print(f"   -> ERRP threshold met at M={M}")

        if found_m_nlpd is None and nlpd <= nlpd_threshold:
            found_m_nlpd = M
            print(f"   -> NLPD threshold met at M={M}")

        if found_m_errp is not None and found_m_nlpd is not None:
            break
        prev_m = M

    # 5. Refinement search
    coarse_m = max(found_m_errp or 0, found_m_nlpd or 0)
    lo = prev_m + 1 if prev_m is not None and prev_m < coarse_m else 1
    if coarse_m > lo:
        print(f"\nRefining search in [{lo}, {coarse_m})...")
        found_m_errp_r = None
        found_m_nlpd_r = None
        for M in range(lo, coarse_m):
            errp, nlpd = run_fn(
                X_train, y_train, X_test, y_test, M, num_classes
            )
            print(f"M={M}: ERRP={errp:.4f}, NLPD={nlpd:.4f}")
            if found_m_errp_r is None and errp <= errp_threshold:
                found_m_errp_r = M
                print(f"   -> ERRP refined at M={M}")
            if found_m_nlpd_r is None and nlpd <= nlpd_threshold:
                found_m_nlpd_r = M
                print(f"   -> NLPD refined at M={M}")
            if found_m_errp_r is not None and found_m_nlpd_r is not None:
                break
        if found_m_errp_r is not None:
            found_m_errp = found_m_errp_r
        if found_m_nlpd_r is not None:
            found_m_nlpd = found_m_nlpd_r

    # 6. Save results
    results = {
        'dataset': args.dataset,
        'seed': seed,
        'threshold_pct': threshold_pct,
        'method': args.method,
        'num_classes': num_classes,
        'n_train': int(N),
        'n_test': int(X_test.shape[0]),
        'trivial': {'errp': float(errp_trivial), 'nlpd': float(nlpd_trivial)},
        'full_svgp': {'errp': float(errp_full), 'nlpd': float(nlpd_full)},
        'thresholds': {'errp': float(errp_threshold), 'nlpd': float(nlpd_threshold)},
        'optimal_m': {
            'errp': found_m_errp,
            'nlpd': found_m_nlpd,
        },
    }

    out_dir = os.path.join(os.path.dirname(__file__), 'optimal_settings')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{args.dataset}_seed{seed}_{args.method}.yaml')
    with open(out_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)

    print("\nResults:")
    if found_m_errp:
        print(f"Smallest M for ERRP threshold: {found_m_errp}")
    else:
        print("Could not find M for ERRP threshold.")
    if found_m_nlpd:
        print(f"Smallest M for NLPD threshold: {found_m_nlpd}")
    else:
        print("Could not find M for NLPD threshold.")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
