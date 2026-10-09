from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import stats


def finite_pairs(x: Iterable[float], y: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    xa = np.asarray(list(x), dtype=float)
    ya = np.asarray(list(y), dtype=float)
    keep = np.isfinite(xa) & np.isfinite(ya)
    return xa[keep], ya[keep]


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    out = np.full_like(p, np.nan)
    finite_idx = np.flatnonzero(np.isfinite(p))
    if finite_idx.size == 0:
        return out
    vals = p[finite_idx]
    order = np.argsort(vals)
    sorted_vals = vals[order]
    m = len(sorted_vals)
    adjusted = np.maximum.accumulate((m - np.arange(m)) * sorted_vals)
    adjusted = np.clip(adjusted, 0.0, 1.0)
    inv = np.empty_like(order)
    inv[order] = np.arange(m)
    out[finite_idx] = adjusted[inv]
    return out


def spearman_with_bootstrap(
    x: Iterable[float],
    y: Iterable[float],
    iterations: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    x, y = finite_pairs(x, y)
    n = len(x)
    if n < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return {"n": n, "rho": np.nan, "p": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    rho, p = stats.spearmanr(x, y)
    boot = []
    for _ in range(iterations):
        idx = rng.integers(0, n, n)
        xb, yb = x[idx], y[idx]
        if np.nanstd(xb) == 0 or np.nanstd(yb) == 0:
            continue
        r, _ = stats.spearmanr(xb, yb)
        if np.isfinite(r):
            boot.append(float(r))
    if boot:
        lo, hi = np.percentile(boot, [2.5, 97.5])
    else:
        lo = hi = np.nan
    return {"n": n, "rho": float(rho), "p": float(p), "ci_low": float(lo), "ci_high": float(hi)}


def cliffs_delta(x: Iterable[float], y: Iterable[float]) -> float:
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    # Number of favorable minus unfavorable pairs divided by all pairs.
    diff = x[:, None] - y[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / diff.size)


def bootstrap_median_ci(values: Iterable[float], iterations: int, rng: np.random.Generator) -> tuple[float, float, float]:
    a = np.asarray(list(values), dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan, np.nan, np.nan
    med = float(np.median(a))
    if len(a) == 1:
        return med, med, med
    boot = np.median(a[rng.integers(0, len(a), size=(iterations, len(a)))], axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return med, float(lo), float(hi)


def within_video_rhos(df, feature_col: str, benefit_col: str, min_points: int) -> np.ndarray:
    values: list[float] = []
    for _, g in df.groupby("sample_uid", sort=False):
        x, y = finite_pairs(g[feature_col], g[benefit_col])
        if len(x) < min_points or np.std(x) == 0 or np.std(y) == 0:
            continue
        rho, _ = stats.spearmanr(x, y)
        if np.isfinite(rho):
            values.append(float(rho))
    return np.asarray(values, dtype=float)


def wilcoxon_against_zero(values: Iterable[float]) -> float:
    a = np.asarray(list(values), dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 3 or np.allclose(a, 0):
        return np.nan
    try:
        return float(stats.wilcoxon(a, zero_method="zsplit", alternative="two-sided").pvalue)
    except ValueError:
        return np.nan


def holm_adjust_list(p_values: Iterable[float]) -> list[float]:
    """List-returning compatibility API for experiment statistics."""
    return holm_adjust(p_values).tolist()
