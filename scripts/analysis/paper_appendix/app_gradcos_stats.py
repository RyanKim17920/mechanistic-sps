"""Small stats helper for app_gradcos.py (kept separate so the plotting file stays about
plotting).  Every function here operates on plain 1-D numpy arrays of per-tensor
cos(g_state, g_pred) values -- no knowledge of the a16 JSON schema.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def bootstrap_mean_ci(values, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap 95% CI (default) for the mean of `values`.  Returns
    (mean, lo, hi).  NaN triple if `values` is empty."""
    v = np.asarray(values, float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boot_means = v[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi)


def frac_positive(values):
    v = np.asarray(values, float)
    if v.size == 0:
        return float("nan")
    return float(np.mean(v > 0))


def paired_wilcoxon(trained, init):
    """Wilcoxon signed-rank test on paired (trained_i - init_i) differences, one pair per
    matrix.  Returns (n_pairs, statistic, p_value); (n, nan, nan) if too few non-zero
    differences for the test to run."""
    t = np.asarray(trained, float)
    i = np.asarray(init, float)
    assert t.shape == i.shape, "paired_wilcoxon needs matched trained/init arrays"
    n = t.size
    if n < 2 or np.allclose(t, i):
        return n, float("nan"), float("nan")
    try:
        stat, p = wilcoxon(t, i)
    except ValueError:
        return n, float("nan"), float("nan")
    return n, float(stat), float(p)


def pair_by_tensor_name(trained_by_name: dict, init_by_name: dict):
    """(names, t_vals, i_vals) for the name intersection of two {tensor_name: cos} dicts,
    name-sorted for determinism."""
    names = sorted(set(trained_by_name) & set(init_by_name))
    t = np.array([trained_by_name[n] for n in names], float)
    i = np.array([init_by_name[n] for n in names], float)
    return names, t, i
