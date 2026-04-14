"""
ode_utils.py
════════════
System-agnostic helpers shared by the SINDy / WSINDy / GS-SINDy notebooks.

Everything here works for any ODESystem defined in ode_systems.py.
Nothing is hard-coded to Lorenz or any other specific system.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  (keeps 3-D projection available)


# ─────────────────────────────────────────────────────────────────────────────
# Data generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_data(system, x0, t_span, dt,
                  noise_level=0.0, seed=42,
                  integrator_kw=None):
    """Integrate *any* ODESystem and optionally add Gaussian noise.

    Parameters
    ----------
    system      : ODESystem  (callable  f(t, state) → list)
    x0          : array-like, initial condition
    t_span      : (t0, tf)
    dt          : timestep
    noise_level : standard deviation of additive noise relative to signal RMS
                  (0 = clean data)
    seed        : random seed for reproducibility
    integrator_kw : extra kwargs forwarded to solve_ivp (e.g. rtol, atol)

    Returns
    -------
    t : 1-D array  (N,)
    X : 2-D array  (N, n_dim)
    """
    kw = dict(method="RK45", rtol=1e-10, atol=1e-12)
    if integrator_kw:
        kw.update(integrator_kw)

    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(system, t_span, x0, t_eval=t_eval, **kw)

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    t = sol.t
    X = sol.y.T   # (N, n_dim)

    if noise_level > 0.0:
        rng = np.random.default_rng(seed)
        # Noise amplitude = noise_level × RMS of each column
        rms = np.sqrt(np.mean(X**2, axis=0))
        X = X + noise_level * rms * rng.standard_normal(X.shape)

    return t, X


def generate_multiple_trajectories(system, x0_list, t_span, dt,
                                    noise_level=0.0, seed=42,
                                    integrator_kw=None):
    """Generate one trajectory per initial condition in x0_list.

    Returns
    -------
    t_list : list of 1-D arrays
    X_list : list of 2-D arrays  (N_i, n_dim)
    """
    t_list, X_list = [], []
    for i, x0 in enumerate(x0_list):
        t_i, X_i = generate_data(system, x0, t_span, dt,
                                  noise_level=noise_level,
                                  seed=seed + i,        # different seed per trajectory
                                  integrator_kw=integrator_kw)
        t_list.append(t_i)
        X_list.append(X_i)
    return t_list, X_list


# ─────────────────────────────────────────────────────────────────────────────
# Simulation from a recovered model
# ─────────────────────────────────────────────────────────────────────────────

def simulate_from_coefficients(coef_matrix, feature_fn, x0, t_eval,
                                integrator_kw=None):
    """Forward-simulate a model defined by a coefficient matrix.

    Parameters
    ----------
    coef_matrix : (n_eqs, n_features) array — one row per equation
    feature_fn  : callable  x (1-D array) → 1-D feature vector
    x0          : initial condition
    t_eval      : 1-D time array
    """
    kw = dict(method="RK45", rtol=1e-8, atol=1e-10)
    if integrator_kw:
        kw.update(integrator_kw)

    def rhs(t, x):
        theta = feature_fn(x)
        return coef_matrix @ theta

    sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), x0, t_eval=t_eval, **kw)
    if not sol.success:
        raise RuntimeError(f"Simulation diverged: {sol.message}")
    return sol.y.T   # (N, n_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_phase_portrait(t, X, var_names, title="Phase portrait", figsize=None):
    """
    For 2-D systems: phase plot (x vs y).
    For 3-D systems: 3-D attractor.
    For 4-D+ systems: pairwise phase plots in a grid.
    """
    n = X.shape[1]

    if n == 2:
        fig, ax = plt.subplots(figsize=figsize or (5, 5))
        ax.plot(X[:, 0], X[:, 1], lw=0.5, alpha=0.8)
        ax.set_xlabel(var_names[0]); ax.set_ylabel(var_names[1])
        ax.set_title(title)
        return fig

    if n == 3:
        fig = plt.figure(figsize=figsize or (7, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(X[:, 0], X[:, 1], X[:, 2], lw=0.4, alpha=0.8)
        ax.set_xlabel(var_names[0]); ax.set_ylabel(var_names[1]); ax.set_zlabel(var_names[2])
        ax.set_title(title)
        return fig

    # ≥4 dims: pairwise grid of the first 4 dimensions
    dims = min(n, 4)
    fig, axes = plt.subplots(dims, dims, figsize=figsize or (3*dims, 3*dims))
    for i in range(dims):
        for j in range(dims):
            ax = axes[i, j]
            if i == j:
                ax.plot(t, X[:, i], lw=0.6)
                ax.set_ylabel(var_names[i])
            else:
                ax.plot(X[:, j], X[:, i], lw=0.3, alpha=0.7)
                ax.set_xlabel(var_names[j]); ax.set_ylabel(var_names[i])
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def plot_time_series(t, X, var_names, title="Time series", figsize=None,
                     X_pred=None, pred_label="Predicted"):
    """Plot each state variable vs time; optionally overlay a prediction."""
    n = X.shape[1]
    fig, axes = plt.subplots(n, 1, figsize=figsize or (10, 2*n), sharex=True)
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(t, X[:, i], "k-", lw=1.5, label="True")
        if X_pred is not None:
            T_pred = t[:len(X_pred)]
            ax.plot(T_pred, X_pred[:, i], "r--", lw=1.5, label=pred_label)
        ax.set_ylabel(var_names[i])
        ax.legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("time")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def plot_trajectories(t, X_true, X_pred, var_names,
                      labels=("True", "Predicted"),
                      title="Trajectory comparison", figsize=None):
    """Side-by-side true vs predicted time series for every dimension."""
    n = X_true.shape[1]
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=figsize or (4*ncols, 3*nrows),
                              sharex=True)
    axes = np.array(axes).flatten()
    for i in range(n):
        ax = axes[i]
        ax.plot(t, X_true[:, i], "k-", lw=1.5, label=labels[0])
        ax.plot(t[:len(X_pred)], X_pred[:, i], "r--", lw=1.5, label=labels[1])
        ax.set_ylabel(var_names[i])
        ax.legend(fontsize=7)
    # hide unused panels
    for i in range(n, len(axes)):
        axes[i].set_visible(False)
    axes[-1].set_xlabel("time")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def plot_coefficient_comparison(true_coeffs, pred_coeffs, feature_names,
                                  eq_names=None, title="Coefficient comparison",
                                  figsize=None):
    """Bar-chart of true vs predicted library coefficients.

    Parameters
    ----------
    true_coeffs  : (n_eqs, n_features)  — pass None to skip true bars
    pred_coeffs  : (n_eqs, n_features)
    feature_names: list of strings, length n_features
    eq_names     : list of strings, length n_eqs  (auto-generated if None)
    """
    n_eq   = pred_coeffs.shape[0]
    n_feat = len(feature_names)

    if eq_names is None:
        eq_names = [f"eq{i}" for i in range(n_eq)]

    fig, axes = plt.subplots(1, n_eq,
                              figsize=figsize or (max(5, n_feat//2)*n_eq, 4))
    if n_eq == 1:
        axes = [axes]

    x = np.arange(n_feat)
    w = 0.38

    for i, ax in enumerate(axes):
        if true_coeffs is not None:
            ax.bar(x - w/2, true_coeffs[i], w,
                   label="True", color="steelblue", alpha=0.9)
        ax.bar(x + (w/2 if true_coeffs is not None else 0),
               pred_coeffs[i], w,
               label="Predicted", color="tomato", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
        ax.set_title(eq_names[i])
        ax.axhline(0, color="k", lw=0.5)
        ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def plot_noise_sweep(noise_levels, results_dict, metric="traj",
                     title=None, figsize=(6, 3)):
    """
    Plot a noise-sweep comparison curve for each method.

    Parameters
    ----------
    noise_levels  : list of floats
    results_dict  : {method_name: list_of_metric_values}
    metric        : 'traj' (trajectory L2) or 'coef' (coefficient L2)
    """
    ylabel = ("Relative L2 trajectory error" if metric == "traj"
              else "Relative L2 coefficient error")
    if title is None:
        title = f"{'Trajectory' if metric=='traj' else 'Coefficient'} Error vs Noise"

    colors  = ["mediumturquoise", "orangered", "orchid"]
    markers = ["o", "s", "^"]

    fig, ax = plt.subplots(figsize=figsize)
    for i, (name, values) in enumerate(results_dict.items()):
        ax.semilogy(noise_levels, values,
                    f"{markers[i % len(markers)]}-",
                    color=colors[i % len(colors)],
                    label=name, lw=1, ms=4,markeredgecolor="white")
    ax.set_xlabel("Noise level", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def relative_l2_error(X_true, X_pred):
    """Relative L2 error per dimension and overall.

    Handles mismatched lengths by truncating to the shorter array.
    """
    n = min(len(X_true), len(X_pred))
    Xt = X_true[:n]; Xp = X_pred[:n]
    per_dim = (np.linalg.norm(Xt - Xp, axis=0)
               / (np.linalg.norm(Xt, axis=0) + 1e-15))
    total   = (np.linalg.norm(Xt - Xp)
               / (np.linalg.norm(Xt) + 1e-15))
    return per_dim, total


def coefficient_error(true_flat, pred_flat):
    """Relative L2 error on the (flattened) coefficient vector."""
    norm_true = np.linalg.norm(true_flat)
    return np.linalg.norm(true_flat - pred_flat) / (norm_true + 1e-15)


def precision_recall(true_flat, pred_flat, tol=1e-2):
    """Support-recovery precision and recall."""
    ts = np.abs(true_flat) > tol
    ps = np.abs(pred_flat) > tol
    tp = np.sum(ts & ps)
    fp = np.sum(~ts & ps)
    fn = np.sum(ts & ~ps)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(precision), float(recall)


def print_metrics(label, per_dim_err, total_err, coeff_err, prec, rec,
                  var_names=None):
    """Pretty-print a standard metrics block."""
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    for i, e in enumerate(per_dim_err):
        name = var_names[i] if var_names else str(i)
        print(f"  Traj. L2 error [{name}]  : {e:.4f}")
    print(f"  Total traj. L2 error    : {total_err:.4f}")
    if coeff_err is not None:
        print(f"  Coefficient L2 error    : {coeff_err:.4f}")
    if prec is not None:
        print(f"  Precision               : {prec:.3f}")
        print(f"  Recall                  : {rec:.3f}")


def metrics_dict(label, per_dim_err, total_err, coeff_err, prec, rec, fit_time):
    """Return a dict of metrics suitable for saving / comparing."""
    return dict(
        method=label,
        per_dim_error=per_dim_err,
        total_error=total_err,
        coef_error=coeff_err,
        precision=prec,
        recall=rec,
        fit_time=fit_time,
    )


# ─────────────────────────────────────────────────────────────────────────────
# True-coefficient helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_true_coef_matrix(system, feature_names):
    """
    Build the ground-truth coefficient matrix from an ODESystem's
    `true_coeffs` field, aligned to the given feature_names list.

    Returns (n_eqs, n_features) array, or None if true_coeffs is None.
    """
    if system.true_coeffs is None:
        return None

    n_eq   = len(system.true_coeffs)
    n_feat = len(feature_names)
    matrix = np.zeros((n_eq, n_feat))

    # Normalise feature names for fuzzy matching
    def normalise(s):
        return str(s).lower().replace(" ", "").replace("·", "").replace("*", "")

    feat_norm = [normalise(f) for f in feature_names]

    for eq_i, coeff_dict in enumerate(system.true_coeffs):
        for feat_str, value in coeff_dict.items():
            target = normalise(feat_str)
            for j, fn in enumerate(feat_norm):
                if fn == target:
                    matrix[eq_i, j] = value
                    break
            else:
                # Warn but do not crash — user may be using a different library
                pass  # silently skip unmatched features

    return matrix


def print_discovered_equations(coef_matrix, feature_names, var_names,
                                 threshold=1e-4):
    """Print the discovered equations in human-readable form."""
    n_eq = coef_matrix.shape[0]
    print("\nDiscovered equations:")
    for i in range(n_eq):
        vname = var_names[i] if i < len(var_names) else f"x{i}"
        terms = []
        for j, name in enumerate(feature_names):
            c = coef_matrix[i, j]
            if abs(c) > threshold:
                terms.append(f"({c:+.4f})·{name}")
        rhs = " + ".join(terms) if terms else "0"
        print(f"  {vname}' = {rhs}")
