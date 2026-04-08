import numpy as np
import tensorflow as tf
import gpflow
import argparse
import yaml
import os
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


HF_REPO = "OccaMLab/bayesian-benchmarks"
DEFAULT_EPOCHS = 100


# -------------------------
# Config loading
# -------------------------
def load_datasets_config():
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'datasets.yaml')
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f) or {}


def load_grids_config():
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'grids.yaml')
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f) or {}


def get_dataset_entry(dataset_name, datasets_config):
    for section in ('regression', 'classification'):
        for entry in datasets_config.get(section, []):
            if entry.get('name') == dataset_name:
                return entry, section
    return {}, None


def load_optimal_m(dataset_name, seed, method):
    path = os.path.join(
        os.path.dirname(__file__), 'optimal_settings',
        f'{dataset_name}_seed{seed}_{method}.yaml'
    )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No optimal settings found at {path}. "
            f"Run the threshold script first."
        )
    with open(path, 'r') as f:
        results = yaml.safe_load(f)
    m_rmse = results.get('optimal_m', {}).get('rmse')
    m_nlpd = results.get('optimal_m', {}).get('nlpd')
    m_errp = results.get('optimal_m', {}).get('errp')
    candidates = [m for m in [m_rmse, m_nlpd, m_errp] if m is not None]
    if not candidates:
        raise ValueError(f"No optimal M found in {path}")
    return max(candidates)


def resolve_lr_candidates(dataset_name, grids_config):
    lr_cfg = grids_config.get('LR_candidates', {}).get(dataset_name)
    if not lr_cfg:
        raise ValueError(f"No LR grid configured for {dataset_name} in grids.yaml")
    start = lr_cfg['start']
    stop = lr_cfg['stop']
    increment = lr_cfg['increment']
    candidates = []
    lr = start
    while lr <= stop + 1e-12:
        candidates.append(round(lr, 10))
        lr += increment
    print(f"LR candidates for {dataset_name}: {candidates}")
    return candidates


# -------------------------
# Data loading
# -------------------------
def load_regression_data(dataset_name, seed=0, train_fraction=0.8):
    ds = load_dataset(HF_REPO, dataset_name, split="train")
    X = np.array(ds["features"], dtype=np.float64)
    y = np.array(ds["target"], dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
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


def load_classification_data(dataset_name, seed=0, train_fraction=0.8):
    data_path = os.path.join(os.path.dirname(__file__), 'data')
    if dataset_name == "diabetes":
        df = pd.read_csv(os.path.join(data_path, "diabetes.csv"))
        X = df.iloc[:, :-1].to_numpy(dtype=np.float64)
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
        raise ValueError(f"Unknown classification dataset: {dataset_name}")
    return X_train, y_train, X_test, y_test


# -------------------------
# Minibatch SVGP training
# -------------------------
def make_svgp(X_train, M, task, num_classes=2):
    if task == 'regression':
        likelihood = gpflow.likelihoods.Gaussian()
        num_latent = 1
    elif num_classes == 2:
        likelihood = gpflow.likelihoods.Bernoulli()
        num_latent = 1
    else:
        likelihood = gpflow.likelihoods.Softmax(num_classes)
        num_latent = num_classes

    if M >= X_train.shape[0]:
        Z = X_train.copy()
    else:
        idx = np.random.choice(X_train.shape[0], M, replace=False)
        Z = X_train[idx]

    D = X_train.shape[1]
    kernel = gpflow.kernels.SquaredExponential(lengthscales=np.ones(D))
    model = gpflow.models.SVGP(
        kernel=kernel,
        likelihood=likelihood,
        inducing_variable=Z,
        num_latent_gps=num_latent,
        num_data=X_train.shape[0],
        whiten=True,
    )
    return model


def train_svgp_minibatch(model, X_train, y_train, batch_size, lr, epochs):
    N = X_train.shape[0]
    steps_per_epoch = max(1, N // batch_size)
    total_steps = epochs * steps_per_epoch
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.shuffle(N).repeat().batch(batch_size)
    iterator = iter(dataset)

    adam = tf.keras.optimizers.legacy.Adam(learning_rate=lr)

    @tf.function
    def step(batch_x, batch_y):
        with tf.GradientTape() as tape:
            loss = model.training_loss((batch_x, batch_y))
        grads = tape.gradient(loss, model.trainable_variables)
        adam.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    print(f"  N={N}, batch_size={batch_size}, steps_per_epoch={steps_per_epoch}, total_steps={total_steps}")
    for i in range(total_steps):
        batch_x, batch_y = next(iterator)
        loss = step(batch_x, batch_y)
        if i % steps_per_epoch == 0:
            epoch = i // steps_per_epoch
            elbo = -model.elbo((X_train, y_train)).numpy()
            tf.print(f"  [Adam lr={lr}] epoch {epoch}, ELBO:", -elbo)


def evaluate_model(model, X_test, y_test, task, num_classes=2):
    if task == 'regression':
        f_mean, _ = model.predict_f(X_test)
        rmse = float(np.sqrt(np.mean((f_mean - y_test) ** 2)))
        nlpd = float(-tf.reduce_mean(model.predict_log_density((X_test, y_test))).numpy())
        return {'rmse': rmse, 'nlpd': nlpd}
    else:
        if num_classes == 2:
            f_mean, _ = model.predict_f(X_test)
            p = model.likelihood.invlink(f_mean).numpy()
            y_pred = (p > 0.5).astype(np.float64)
            errp = float(np.mean(y_test.flatten() != y_pred.flatten()))
        else:
            f_mean, _ = model.predict_f(X_test)
            p = tf.nn.softmax(f_mean).numpy()
            y_pred = np.argmax(p, axis=1)
            errp = float(np.mean(y_test.flatten() != y_pred))
        nlpd = float(-tf.reduce_mean(model.predict_log_density((X_test, y_test))).numpy())
        return {'errp': errp, 'nlpd': nlpd}


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Find optimal LR for minibatch SVGP given optimal M.'
    )
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--method', type=str, default='train',
                        choices=['train', 'greedy'])
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Minibatch size (overrides config)')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help='Number of epochs (passes over the dataset)')
    parser.add_argument('--metric', type=str, default=None,
                        choices=['rmse', 'nlpd', 'errp'],
                        help='Metric to optimise (default: nlpd for regression, errp for classification)')
    args = parser.parse_args()

    datasets_config = load_datasets_config()
    grids_config = load_grids_config()
    entry, task = get_dataset_entry(args.dataset, datasets_config)

    if task is None:
        raise ValueError(f"Dataset {args.dataset} not found in datasets.yaml")

    seed = args.seed if args.seed is not None else entry.get('seed', 0)
    batch_size = args.batch_size if args.batch_size is not None else entry.get('batch_size')
    if batch_size is None:
        raise ValueError(
            f"No batch_size configured for {args.dataset} in datasets.yaml "
            f"and --batch_size not provided."
        )

    np.random.seed(seed)
    tf.random.set_seed(seed)

    M = load_optimal_m(args.dataset, seed, args.method)
    print(f"Dataset: {args.dataset}, task: {task}, seed: {seed}")
    print(f"Optimal M: {M}, batch_size: {batch_size}")

    # Load data
    if task == 'regression':
        X_train, y_train, X_test, y_test = load_regression_data(args.dataset, seed=seed)
        num_classes = None
        default_metric = 'nlpd'
    else:
        X_train, y_train, X_test, y_test = load_classification_data(args.dataset, seed=seed)
        num_classes = len(np.unique(np.concatenate([y_train, y_test])))
        default_metric = 'errp'

    metric = args.metric or default_metric
    print(f"Train N={X_train.shape[0]}, Test N={X_test.shape[0]}, metric={metric}")

    # Sweep LR
    lr_candidates = resolve_lr_candidates(args.dataset, grids_config)
    all_results = []

    for lr in lr_candidates:
        print(f"\n--- LR={lr} ---")
        np.random.seed(seed)
        tf.random.set_seed(seed)

        model = make_svgp(X_train, M, task, num_classes or 2)
        train_svgp_minibatch(model, X_train, y_train, batch_size, lr, args.epochs)
        metrics = evaluate_model(model, X_test, y_test, task, num_classes or 2)
        print(f"  Result: {metrics}")
        all_results.append({'lr': lr, **metrics})

    # Find best
    best = min(all_results, key=lambda r: r[metric])
    print(f"\nBest LR by {metric}: {best['lr']} -> {best}")

    # Save
    output = {
        'dataset': args.dataset,
        'task': task,
        'seed': seed,
        'method': args.method,
        'optimal_m': M,
        'batch_size': batch_size,
        'epochs': args.epochs,
        'metric_optimised': metric,
        'best_lr': best['lr'],
        'best_metrics': {k: v for k, v in best.items() if k != 'lr'},
        'all_results': all_results,
    }

    out_dir = os.path.join(os.path.dirname(__file__), 'optimal_settings')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{args.dataset}_seed{seed}_{args.method}_lr.yaml')
    with open(out_path, 'w') as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)

    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
