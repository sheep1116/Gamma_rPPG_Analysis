from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESPONSE_ORDER = ["positive_sensitive", "stable", "non_monotonic", "negative_sensitive", "invalid"]


def _finish(fig, path: Path, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _short_feature(name: str) -> str:
    return name.replace("delta_", "Δ ").replace("_", " ")


def plot_forest(df: pd.DataFrame, title: str, path: Path, dpi: int) -> None:
    d = df.dropna(subset=["spearman_rho"]).copy().sort_values("spearman_rho")
    if d.empty:
        return
    y = np.arange(len(d))
    lower = d["spearman_rho"] - d["rho_ci95_low"]
    upper = d["rho_ci95_high"] - d["spearman_rho"]
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.4 * len(d) + 1.8)))
    sig = d.get("p_holm", pd.Series(np.nan, index=d.index)).to_numpy() < 0.05
    for i in range(len(d)):
        marker = "o" if sig[i] else "s"
        alpha = 1.0 if sig[i] else 0.55
        ax.errorbar(
            d.iloc[i]["spearman_rho"], y[i],
            xerr=np.array([[lower.iloc[i]], [upper.iloc[i]]]),
            fmt=marker, capsize=3, alpha=alpha,
        )
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([_short_feature(v) for v in d["feature"]])
    ax.set_xlabel("Spearman rho: baseline ROI feature vs best MAE benefit")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    _finish(fig, path, dpi)


def plot_scatter(
    df: pd.DataFrame,
    feature: str,
    title: str,
    annotation: str,
    path: Path,
    dpi: int,
) -> None:
    d = df[[feature, "absolute_mae_improvement", "response_type", "sample_uid"]].dropna().copy()
    if len(d) < 3:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    for response in RESPONSE_ORDER:
        g = d[d["response_type"] == response]
        if g.empty:
            continue
        ax.scatter(g[feature], g["absolute_mae_improvement"], alpha=0.72, label=f"{response} (n={len(g)})")
    x = d[feature].to_numpy(float)
    y = d["absolute_mae_improvement"].to_numpy(float)
    if np.std(x) > 0:
        coef = np.polyfit(x, y, 1)
        xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ax.plot(xx, coef[0] * xx + coef[1], linewidth=1.5, linestyle="--")
    ax.axhline(0, linewidth=1)
    ax.set_xlabel(_short_feature(feature))
    ax.set_ylabel("Best MAE benefit (BPM)")
    ax.set_title(title)
    ax.text(0.02, 0.98, annotation, transform=ax.transAxes, va="top", ha="left")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.22)
    _finish(fig, path, dpi)


def plot_group_distribution(
    df: pd.DataFrame,
    feature: str,
    positive_label: str,
    stable_label: str,
    title: str,
    annotation: str,
    path: Path,
    dpi: int,
) -> None:
    groups = []
    labels = []
    for label in [positive_label, stable_label]:
        vals = pd.to_numeric(df.loc[df["response_type"] == label, feature], errors="coerce").dropna().to_numpy()
        if len(vals):
            groups.append(vals)
            labels.append(f"{label}\n(n={len(vals)})")
    if len(groups) < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.boxplot(groups, labels=labels, showfliers=False)
    rng = np.random.default_rng(12345)
    for i, vals in enumerate(groups, 1):
        jitter = rng.normal(i, 0.045, len(vals))
        ax.scatter(jitter, vals, alpha=0.65, s=26)
    ax.set_ylabel(_short_feature(feature))
    ax.set_title(title)
    ax.text(0.02, 0.98, annotation, transform=ax.transAxes, va="top", ha="left")
    ax.grid(axis="y", alpha=0.22)
    _finish(fig, path, dpi)


def plot_mechanism_curves(
    summary: pd.DataFrame,
    feature: str,
    title: str,
    path: Path,
    dpi: int,
) -> None:
    d = summary[summary["feature"] == feature].copy()
    if d.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 8.2), sharex=True)
    for response_type, g in d.groupby("response_type"):
        g = g.sort_values("gamma")
        axes[0].plot(g["gamma"], g["benefit_median"], marker="o", label=response_type)
        axes[0].fill_between(g["gamma"], g["benefit_ci_low"], g["benefit_ci_high"], alpha=0.15)
        axes[1].plot(g["gamma"], g["feature_delta_median"], marker="o", label=response_type)
        axes[1].fill_between(g["gamma"], g["feature_delta_ci_low"], g["feature_delta_ci_high"], alpha=0.15)
    axes[0].axhline(0, linewidth=1)
    axes[1].axhline(0, linewidth=1)
    axes[0].set_ylabel("MAE benefit = MAE(1.0) - MAE(gamma), BPM")
    axes[1].set_ylabel(f"Change in {_short_feature(feature)}")
    axes[1].set_xlabel("Gamma")
    axes[0].set_title(title)
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.22)
    _finish(fig, path, dpi)


def _matrix_from_long(df: pd.DataFrame, value: str, sample_order: list[str], gammas: list[float]) -> np.ndarray:
    pivot = df.pivot_table(index="sample_uid", columns="gamma", values=value, aggfunc="first")
    return pivot.reindex(index=sample_order, columns=gammas).to_numpy(float)


def plot_paired_heatmaps(
    df: pd.DataFrame,
    feature_delta: str,
    title: str,
    path: Path,
    dpi: int,
) -> None:
    d = df[df["gamma"] != 1.0].copy()
    if d.empty:
        return
    order_df = (
        d.groupby("sample_uid", as_index=False)
        .agg(response_type=("response_type", "first"), max_benefit=("MAE_benefit", "max"))
    )
    rank = {name: i for i, name in enumerate(RESPONSE_ORDER)}
    order_df["response_rank"] = order_df["response_type"].map(rank).fillna(99)
    order_df = order_df.sort_values(["response_rank", "max_benefit"], ascending=[True, False])
    samples = order_df["sample_uid"].tolist()
    gammas = sorted(d["gamma"].dropna().unique().tolist())
    a = _matrix_from_long(d, "MAE_benefit", samples, gammas)
    b = _matrix_from_long(d, feature_delta, samples, gammas)
    fig, axes = plt.subplots(1, 2, figsize=(13, max(5.5, 0.16 * len(samples) + 2.2)), sharey=True)
    im0 = axes[0].imshow(a, aspect="auto", interpolation="nearest")
    im1 = axes[1].imshow(b, aspect="auto", interpolation="nearest")
    axes[0].set_title("MAE benefit")
    axes[1].set_title(_short_feature(feature_delta))
    for ax in axes:
        ax.set_xticks(np.arange(len(gammas)))
        ax.set_xticklabels([f"{g:.1f}" for g in gammas], rotation=45)
        ax.set_xlabel("Gamma")
    axes[0].set_yticks(np.arange(len(samples)))
    axes[0].set_yticklabels([s.split("/")[-1] for s in samples], fontsize=6)
    axes[0].set_ylabel("Samples sorted by response type and benefit")
    fig.colorbar(im0, ax=axes[0], fraction=0.025, pad=0.02)
    fig.colorbar(im1, ax=axes[1], fraction=0.025, pad=0.02)
    fig.suptitle(title)
    _finish(fig, path, dpi)


def plot_pixel_composition(summary: pd.DataFrame, title: str, path: Path, dpi: int) -> None:
    if summary.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for ax, response in zip(axes, ["positive_sensitive", "stable"]):
        d = summary[summary["response_type"] == response].sort_values("gamma")
        if d.empty:
            ax.set_visible(False)
            continue
        for col, label in [
            ("valid_pixel_ratio_median", "valid"),
            ("rejected_dark_pixel_ratio_median", "rejected dark"),
            ("rejected_bright_pixel_ratio_median", "rejected bright"),
        ]:
            ax.plot(d["gamma"], d[col], marker="o", label=label)
        ax.set_title(f"{response} (sample-level medians)")
        ax.set_xlabel("Gamma")
        ax.grid(alpha=0.22)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Pixel proportion")
    fig.suptitle(title)
    _finish(fig, path, dpi)


def plot_failure_heatmap(raw: pd.DataFrame, title: str, path: Path, dpi: int) -> None:
    d = raw.copy()
    if d.empty:
        return
    d["success"] = (d["status"] == "ok").astype(float)
    pivot = d.pivot_table(index="sample_uid", columns="gamma", values="success", aggfunc="min")
    if pivot.empty:
        return
    fail_counts = (1 - pivot).sum(axis=1)
    pivot = pivot.loc[fail_counts.sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(9, max(5, 0.13 * len(pivot) + 2)))
    im = ax.imshow(pivot.to_numpy(float), aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{float(v):.1f}" for v in pivot.columns], rotation=45)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([s.split("/")[-1] for s in pivot.index], fontsize=6)
    ax.set_xlabel("Gamma")
    ax.set_ylabel("Samples sorted by failure count")
    ax.set_title(title + " (0=failed, 1=ok)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    _finish(fig, path, dpi)


def plot_failure_rate(raw: pd.DataFrame, title: str, path: Path, dpi: int) -> None:
    d = raw.groupby("gamma", as_index=False).agg(success_rate=("status", lambda x: float(np.mean(x == "ok"))))
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(d["gamma"], d["success_rate"], marker="o")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Gamma")
    ax.set_ylabel("Task success rate")
    ax.set_title(title)
    ax.grid(alpha=0.22)
    _finish(fig, path, dpi)


def plot_consensus_distribution(df: pd.DataFrame, title: str, path: Path, dpi: int) -> None:
    counts = df["cross_method_consensus"].value_counts().sort_values(ascending=False)
    if counts.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.bar(counts.index, counts.values)
    ax.set_ylabel("Video count")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.22)
    _finish(fig, path, dpi)


def plot_consensus_feature(df: pd.DataFrame, feature: str, title: str, path: Path, dpi: int) -> None:
    groups = []
    labels = []
    for label in ["multi_method_positive", "single_method_positive", "mostly_stable", "mixed_response", "mostly_negative"]:
        vals = pd.to_numeric(df.loc[df["cross_method_consensus"] == label, feature], errors="coerce").dropna().to_numpy()
        if len(vals):
            groups.append(vals)
            labels.append(f"{label}\n(n={len(vals)})")
    if len(groups) < 2:
        return
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.boxplot(groups, labels=labels, showfliers=False)
    rng = np.random.default_rng(2468)
    for i, vals in enumerate(groups, 1):
        ax.scatter(rng.normal(i, 0.04, len(vals)), vals, alpha=0.6, s=24)
    ax.set_ylabel(_short_feature(feature))
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.22)
    _finish(fig, path, dpi)
