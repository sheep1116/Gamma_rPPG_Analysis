from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path

from gamma_rppg.common.statistics import holm_adjust_list as holm_adjust
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gamma_rppg.common.config import ensure_dir, load_config, resolve_mode_root


FEATURES = [
    "roi_mean_r", "roi_mean_g", "roi_mean_b", "roi_median_r", "roi_median_g", "roi_median_b",
    "r_over_g", "b_over_g", "r_over_b", "r_minus_g", "g_minus_b", "channel_range",
    "roi_luma_mean", "roi_luma_median", "roi_luma_std", "roi_luma_p05", "roi_luma_p25",
    "roi_luma_p75", "roi_luma_p95", "roi_luma_p99", "roi_luma_iqr",
    "roi_luma_p95_minus_p05", "dark_ratio_5", "dark_ratio_10", "dark_ratio_20",
    "bright_ratio_220", "bright_ratio_235", "sat_ratio_245", "sat_ratio_250",
    "temporal_luma_std", "temporal_luma_cv", "temporal_r_std", "temporal_g_std",
    "temporal_b_std", "temporal_r_cv", "temporal_g_cv", "temporal_b_cv",
    "mean_frame_to_frame_luma_change", "p95_frame_to_frame_luma_change",
    "spatial_luma_std", "spatial_luma_iqr", "highlight_component_ratio",
    "shadow_component_ratio", "left_right_luma_difference", "upper_lower_luma_difference",
    "valid_pixel_ratio_mean", "rejected_dark_pixel_ratio", "rejected_bright_pixel_ratio",
    "valid_pixel_luma_mean", "valid_pixel_luma_std", "roi_area_mean",
]


def safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("_") or "sample"




def adjacent_support(group: pd.DataFrame, baseline: float, best_gamma: float) -> bool:
    gammas = sorted(group["gamma"].dropna().unique().astype(float).tolist())
    if best_gamma not in gammas:
        return False
    best_index = gammas.index(best_gamma)
    neighbors = []
    if best_index > 0 and not np.isclose(gammas[best_index - 1], baseline):
        neighbors.append(gammas[best_index - 1])
    if best_index + 1 < len(gammas):
        neighbors.append(gammas[best_index + 1])
    base_row = group[np.isclose(group["gamma"], baseline)]
    if base_row.empty:
        return False
    base_mae = float(base_row.iloc[-1]["MAE"])
    return any(float(group[np.isclose(group["gamma"], gamma)].iloc[-1]["MAE"]) < base_mae for gamma in neighbors)


def classify_response(
    group: pd.DataFrame,
    baseline: float,
    min_abs: float,
    min_rel: float,
    require_adjacent: bool,
    stable_epsilon: float,
) -> Dict:
    group = group.sort_values("gamma").dropna(subset=["MAE"])
    base_rows = group[np.isclose(group["gamma"], baseline)]
    if base_rows.empty or group.empty:
        return {"response_type": "invalid"}
    base = float(base_rows.iloc[-1]["MAE"])
    best_row = group.loc[group["MAE"].idxmin()]
    best = float(best_row["MAE"])
    best_gamma = float(best_row["gamma"])
    improvement = base - best
    relative = improvement / base if base > 0 else np.nan
    deltas = group["MAE"].to_numpy(dtype=float) - base
    adjacent = adjacent_support(group, baseline, best_gamma)
    positive = improvement >= min_abs and relative >= min_rel and (adjacent or not require_adjacent)
    if positive:
        response = "positive_sensitive"
    elif np.nanmax(np.abs(deltas)) <= stable_epsilon:
        response = "stable"
    else:
        nonbaseline = deltas[~np.isclose(group["gamma"].to_numpy(dtype=float), baseline)]
        if nonbaseline.size and np.all(nonbaseline >= min_abs):
            response = "negative_sensitive"
        else:
            response = "non_monotonic"
    return {
        "base_MAE": base,
        "best_MAE": best,
        "best_gamma": best_gamma,
        "absolute_mae_improvement": improvement,
        "relative_mae_improvement": relative,
        "adjacent_gamma_support": adjacent,
        "response_type": response,
    }


def build_labels(raw: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sensitivity = config.get("sensitivity", {})
    baseline = float(config.get("analysis", {}).get("baseline_gamma", 1.0))
    min_abs = float(sensitivity.get("min_absolute_mae_improvement", 0.2))
    min_rel = float(sensitivity.get("min_relative_mae_improvement", 0.05))
    require_adjacent = bool(sensitivity.get("require_adjacent_gamma_support", True))
    stable_epsilon = float(sensitivity.get("stable_mae_epsilon", 0.2))
    min_valid_windows = int(sensitivity.get("min_valid_windows", 10))
    wsize = float(config.get("experiment", {}).get("wsize", 8))
    step = float(config.get("experiment", {}).get("step", 1))
    rows, curves = [], []
    expected_gammas = {round(float(value), 10) for value in pd.to_numeric(raw.get("gamma"), errors="coerce").dropna().unique()}
    ok = raw[(raw["status"] == "ok") & (raw["transform_mode"] == "pre_roi_gamma")].copy()
    ok["gamma"] = pd.to_numeric(ok["gamma"], errors="coerce")
    ok["MAE"] = pd.to_numeric(ok["MAE"], errors="coerce")
    ok["SNR"] = pd.to_numeric(ok.get("SNR"), errors="coerce")
    for (sample_uid, method), group in ok.groupby(["sample_uid", "method"]):
        result = classify_response(group, baseline, min_abs, min_rel, require_adjacent, stable_epsilon)
        first = group.iloc[0]
        fps = float(pd.to_numeric(pd.Series([first.get("fps", np.nan)]), errors="coerce").iloc[0])
        num_frames = float(pd.to_numeric(group.get("num_frames", pd.Series([np.nan])), errors="coerce").dropna().min()) if "num_frames" in group else np.nan
        duration = num_frames / fps if np.isfinite(num_frames) and np.isfinite(fps) and fps > 0 else np.nan
        valid_windows = max(0, int(np.floor((duration - wsize) / step) + 1)) if np.isfinite(duration) and duration >= wsize else 0
        present_gammas = {round(float(value), 10) for value in group["gamma"].dropna().unique()}
        gamma_complete = expected_gammas.issubset(present_gammas)
        eligible = valid_windows >= min_valid_windows and gamma_complete
        if not eligible:
            result["response_type"] = "invalid"
        row = {
            "dataset": first["dataset"], "sample_uid": sample_uid, "method": method,
            "subject_id": first.get("subject_id", ""), "motion_type": first.get("motion_type", ""),
            "source_label": first.get("source_label", ""), "input_type": first.get("input_type", ""),
            "input_path": first.get("input_path", ""), "fps": fps, "num_frames": num_frames,
            "estimated_valid_windows": valid_windows, "gamma_complete": gamma_complete,
            "eligible_for_sensitivity": eligible, **result,
        }
        rows.append(row)
        for _, item in group.iterrows():
            curves.append({
                **row, "gamma": item["gamma"], "MAE": item["MAE"], "SNR": item.get("SNR", np.nan),
                "MAE_delta_vs_baseline": item["MAE"] - result.get("base_MAE", np.nan),
            })
    return pd.DataFrame(rows), pd.DataFrame(curves)


def cross_method_consensus(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sample_uid, group in labels.groupby("sample_uid"):
        counts = group["response_type"].value_counts()
        positive = int(counts.get("positive_sensitive", 0))
        stable = int(counts.get("stable", 0))
        negative = int(counts.get("negative_sensitive", 0))
        nonmono = int(counts.get("non_monotonic", 0))
        invalid = int(counts.get("invalid", 0))
        method_count = len(group) - invalid
        if method_count <= 0:
            label = "invalid"
        elif positive >= max(2, math.ceil(method_count / 2)):
            label = "multi_method_positive"
        elif positive == 1:
            label = "single_method_positive"
        elif negative >= max(2, math.ceil(method_count / 2)):
            label = "mostly_negative"
        elif stable >= max(2, math.ceil(method_count / 2)):
            label = "mostly_stable"
        else:
            label = "mixed_response"
        first = group.iloc[0]
        rows.append({
            "dataset": first["dataset"], "sample_uid": sample_uid,
            "subject_id": first.get("subject_id", ""), "motion_type": first.get("motion_type", ""),
            "source_label": first.get("source_label", ""), "input_path": first.get("input_path", ""),
            "positive_method_count": positive, "stable_method_count": stable,
            "negative_method_count": negative, "non_monotonic_method_count": nonmono,
            "invalid_method_count": invalid, "eligible_method_count": method_count,
            "mean_relative_improvement": float(group["relative_mae_improvement"].mean()),
            "median_relative_improvement": float(group["relative_mae_improvement"].median()),
            "median_best_gamma": float(group["best_gamma"].median()),
            "cross_method_consensus": label,
        })
    return pd.DataFrame(rows)


def cliff_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if not len(x) or not len(y):
        return np.nan
    greater = sum(np.sum(value > y) for value in x)
    lower = sum(np.sum(value < y) for value in x)
    return float((greater - lower) / (len(x) * len(y)))


def spearman_bootstrap_ci(x: np.ndarray, y: np.ndarray, iterations: int, seed: int) -> Tuple[float, float]:
    from scipy.stats import spearmanr
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3 or iterations <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(iterations):
        idx = rng.integers(0, len(x), size=len(x))
        if np.unique(x[idx]).size < 2 or np.unique(y[idx]).size < 2:
            continue
        rho, _ = spearmanr(x[idx], y[idx])
        if np.isfinite(rho):
            values.append(rho)
    if not values:
        return np.nan, np.nan
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def correlation_analysis(merged: pd.DataFrame, config: Dict) -> pd.DataFrame:
    from scipy.stats import spearmanr
    rows = []
    for method, group in merged.groupby("method"):
        for feature in FEATURES:
            if feature not in group.columns:
                continue
            valid = group[[feature, "absolute_mae_improvement"]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(valid) < 5 or valid[feature].nunique() < 2:
                continue
            rho, pvalue = spearmanr(valid[feature], valid["absolute_mae_improvement"])
            iterations = int(config.get("sensitivity", {}).get("correlation_bootstrap_iterations", 500))
            low, high = spearman_bootstrap_ci(
                valid[feature].to_numpy(), valid["absolute_mae_improvement"].to_numpy(),
                iterations, int(config.get("analysis", {}).get("random_seed", 2026)) + len(rows) * 37,
            )
            rows.append({
                "dataset": group["dataset"].iloc[0], "method": method, "feature": feature,
                "n": len(valid), "spearman_rho": rho, "rho_ci95_low": low,
                "rho_ci95_high": high, "p_value": pvalue, "p_holm": np.nan,
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    for method, indices in frame.groupby("method").groups.items():
        adjusted = holm_adjust(frame.loc[indices, "p_value"].fillna(1.0).tolist())
        frame.loc[indices, "p_holm"] = adjusted
    return frame


def build_feature_changes(brightness: pd.DataFrame, curves: pd.DataFrame, baseline: float) -> pd.DataFrame:
    frame = brightness.copy()
    frame["gamma"] = pd.to_numeric(frame["gamma"], errors="coerce")
    available = [feature for feature in FEATURES if feature in frame.columns]
    for feature in available:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    base = frame[np.isclose(frame["gamma"], baseline)][["sample_uid", *available]].copy()
    base = base.rename(columns={feature: f"{feature}_baseline" for feature in available})
    changed = frame.merge(base, on="sample_uid", how="left")
    for feature in available:
        changed[f"delta_{feature}"] = changed[feature] - changed[f"{feature}_baseline"]
    curve_cols = ["dataset", "sample_uid", "method", "gamma", "MAE_delta_vs_baseline", "response_type"]
    return curves[curve_cols].merge(changed, on=["dataset", "sample_uid", "gamma"], how="left")


def feature_change_correlations(changes: pd.DataFrame, config: Dict) -> pd.DataFrame:
    from scipy.stats import spearmanr
    rows = []
    delta_features = [f"delta_{feature}" for feature in FEATURES if f"delta_{feature}" in changes.columns]
    for method, group in changes.groupby("method"):
        for feature in delta_features:
            valid = group[[feature, "MAE_delta_vs_baseline"]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(valid) < 5 or valid[feature].nunique() < 2:
                continue
            rho, pvalue = spearmanr(valid[feature], valid["MAE_delta_vs_baseline"])
            low, high = spearman_bootstrap_ci(
                valid[feature].to_numpy(), valid["MAE_delta_vs_baseline"].to_numpy(),
                int(config.get("sensitivity", {}).get("correlation_bootstrap_iterations", 500)),
                int(config.get("analysis", {}).get("random_seed", 2026)) + len(rows) * 43,
            )
            rows.append({
                "dataset": group["dataset"].iloc[0], "method": method, "feature_change": feature,
                "n": len(valid), "spearman_rho": rho, "rho_ci95_low": low,
                "rho_ci95_high": high, "p_value": pvalue, "p_holm": np.nan,
            })
    result = pd.DataFrame(rows)
    if not result.empty:
        for method, indices in result.groupby("method").groups.items():
            result.loc[indices, "p_holm"] = holm_adjust(result.loc[indices, "p_value"].fillna(1.0).tolist())
    return result


def group_tests(merged: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import kruskal, mannwhitneyu
    rows = []
    order = ["positive_sensitive", "stable", "negative_sensitive"]
    for method, method_group in merged.groupby("method"):
        for feature in FEATURES:
            if feature not in method_group.columns:
                continue
            arrays = {
                label: pd.to_numeric(
                    method_group.loc[method_group["response_type"] == label, feature], errors="coerce"
                ).dropna().to_numpy(dtype=float)
                for label in order
            }
            available = [array for array in arrays.values() if len(array) >= 2]
            if len(available) >= 2:
                try:
                    statistic, pvalue = kruskal(*available)
                except Exception:
                    statistic, pvalue = np.nan, np.nan
                rows.append({
                    "dataset": method_group["dataset"].iloc[0], "method": method,
                    "feature": feature, "test": "kruskal", "group_a": "all",
                    "group_b": "all", "n_a": sum(len(v) for v in arrays.values()), "n_b": np.nan,
                    "statistic": statistic, "p_value": pvalue, "p_holm": np.nan,
                    "cliffs_delta": np.nan,
                })
            for a, b in [("positive_sensitive", "stable"), ("positive_sensitive", "negative_sensitive")]:
                x, y = arrays[a], arrays[b]
                if len(x) >= 2 and len(y) >= 2:
                    statistic, pvalue = mannwhitneyu(x, y, alternative="two-sided")
                    rows.append({
                        "dataset": method_group["dataset"].iloc[0], "method": method,
                        "feature": feature, "test": "mannwhitney", "group_a": a, "group_b": b,
                        "n_a": len(x), "n_b": len(y), "statistic": statistic,
                        "p_value": pvalue, "p_holm": np.nan, "cliffs_delta": cliff_delta(x, y),
                    })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    for method, indices in frame.groupby("method").groups.items():
        frame.loc[indices, "p_holm"] = holm_adjust(frame.loc[indices, "p_value"].fillna(1.0).tolist())
    return frame


def plot_outputs(dataset_dir: Path, merged: pd.DataFrame, labels: pd.DataFrame, consensus: pd.DataFrame) -> None:
    figures = ensure_dir(dataset_dir / "sensitivity_figures")
    scatter_specs = [
        ("roi_luma_mean", "baseline_luma_vs_mae_improvement.png"),
        ("sat_ratio_245", "baseline_saturation_vs_mae_improvement.png"),
        ("valid_pixel_ratio_mean", "baseline_valid_pixel_ratio_vs_improvement.png"),
        ("temporal_luma_cv", "temporal_luma_cv_vs_improvement.png"),
    ]
    for feature, name in scatter_specs:
        if feature not in merged.columns:
            continue
        fig, ax = plt.subplots(figsize=(9, 6))
        for method, group in merged.groupby("method"):
            ax.scatter(group[feature], group["absolute_mae_improvement"], alpha=0.6, label=method)
        ax.axhline(0, linewidth=1)
        ax.set_xlabel(feature)
        ax.set_ylabel("Best MAE improvement (BPM)")
        ax.set_title(f"{dataset_dir.name}: {feature} vs Gamma benefit")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / name, dpi=180)
        plt.close(fig)

    for feature, name in [("roi_luma_mean", "sensitivity_group_luma_boxplot.png"), ("sat_ratio_245", "sensitivity_group_saturation_boxplot.png")]:
        if feature not in merged.columns:
            continue
        groups = ["positive_sensitive", "stable", "negative_sensitive", "non_monotonic"]
        data = [pd.to_numeric(merged.loc[merged["response_type"] == label, feature], errors="coerce").dropna() for label in groups]
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.boxplot(data, labels=groups, showfliers=False)
        ax.set_ylabel(feature)
        ax.set_title(f"{dataset_dir.name}: baseline ROI feature by response type")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(figures / name, dpi=180)
        plt.close(fig)

    pivot = labels.pivot(index="sample_uid", columns="method", values="best_gamma")
    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(10, max(5, len(pivot) * 0.22)))
        image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto")
        ax.set_xticks(range(len(pivot.columns)), pivot.columns)
        ax.set_yticks(range(len(pivot.index)), pivot.index, fontsize=6)
        ax.set_title(f"{dataset_dir.name}: best Gamma by video and method")
        fig.colorbar(image, ax=ax, label="Best Gamma")
        fig.tight_layout()
        fig.savefig(figures / "best_gamma_heatmap.png", dpi=180)
        plt.close(fig)

    counts = labels.groupby(["method", "response_type"]).size().unstack(fill_value=0)
    if not counts.empty:
        fig, ax = plt.subplots(figsize=(9, 6))
        counts.plot(kind="bar", stacked=True, ax=ax)
        ax.set_ylabel("Video count")
        ax.set_title(f"{dataset_dir.name}: response type distribution")
        fig.tight_layout()
        fig.savefig(figures / "response_type_distribution.png", dpi=180)
        plt.close(fig)

    if not consensus.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        consensus["positive_method_count"].value_counts().sort_index().plot(kind="bar", ax=ax)
        ax.set_xlabel("Number of methods with positive response")
        ax.set_ylabel("Video count")
        ax.set_title(f"{dataset_dir.name}: cross-method positive count")
        fig.tight_layout()
        fig.savefig(figures / "method_positive_count_distribution.png", dpi=180)
        plt.close(fig)


def generate_signal_diagnostics(
    input_path: str,
    fps: float,
    method_name: str,
    best_gamma: float,
    folder: Path,
    config: Dict,
) -> None:
    """Optionally rerun one pre-ROI case to create RGB and BVP diagnostic figures."""
    import pyVHR as vhr
    from pyVHR.BVP import BPfilter, RGB_sig_to_BVP
    from gamma_rppg.experiments.methods import resolve_method
    from gamma_rppg.experiments.pre_roi import extract_signal_for_gamma

    baseline = float(config.get("analysis", {}).get("baseline_gamma", 1.0))
    signals = {}
    diagnostic_config = deep_update(config, {"experiment": {"collect_brightness": False}})
    for gamma in [baseline, float(best_gamma)]:
        example = folder / ("roi_examples.png" if np.isclose(gamma, best_gamma) else "roi_examples_baseline.png")
        signal, _, _ = extract_signal_for_gamma(input_path, fps, gamma, diagnostic_config, example)
        signals[gamma] = np.asarray(signal, dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    channel_names = ["R", "G", "B"]
    for channel, ax in enumerate(axes):
        for gamma, signal in signals.items():
            time_axis = np.arange(len(signal)) / fps
            ax.plot(time_axis, signal[:, channel], label=f"Gamma={gamma:g}", linewidth=0.8)
        ax.set_ylabel(channel_names[channel])
        ax.grid(True, alpha=0.25)
    axes[0].legend()
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"ROI RGB mean time series: {method_name} case")
    fig.tight_layout()
    fig.savefig(folder / "rgb_timeseries.png", dpi=170)
    plt.close(fig)

    wsize = float(config.get("experiment", {}).get("wsize", 8))
    step = float(config.get("experiment", {}).get("step", 1))
    prefer_cuda = bool(config.get("experiment", {}).get("prefer_cuda", True))
    filter_params = dict(config.get("experiment", {}).get("bp_filter", {"order": 6, "minHz": 0.65, "maxHz": 4.0}))
    spec = resolve_method(vhr.BVP, method_name, prefer_cuda=prefer_cuda)
    method_function = getattr(vhr.BVP, spec.function_name)
    fig, ax = plt.subplots(figsize=(9, 5))
    for gamma, signal in signals.items():
        windowed, _ = vhr.extraction.sig_windowing(signal, wsize, step, fps)
        bvps = RGB_sig_to_BVP(
            windowed, fps, device_type=spec.device_type,
            method=method_function, params=dict(spec.params),
        )
        filter_call = dict(filter_params)
        filter_call["fps"] = fps
        bvps = vhr.BVP.apply_filter(bvps, BPfilter, params=filter_call)
        spectra = []
        frequencies = None
        for bvp in bvps:
            if hasattr(bvp, "get"):
                bvp = bvp.get()
            vector = np.asarray(bvp, dtype=float).reshape(-1)
            if vector.size < 4:
                continue
            vector = vector - np.mean(vector)
            power = np.abs(np.fft.rfft(vector)) ** 2
            freq = np.fft.rfftfreq(vector.size, d=1.0 / fps)
            mask = (freq >= 0.5) & (freq <= 4.0)
            if np.any(mask):
                frequencies = freq[mask]
                spectra.append(power[mask])
        if spectra and frequencies is not None:
            min_len = min(len(value) for value in spectra)
            mean_power = np.mean(np.vstack([value[:min_len] for value in spectra]), axis=0)
            ax.plot(frequencies[:min_len], mean_power, label=f"Gamma={gamma:g}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Mean BVP power")
    ax.set_title(f"BVP spectrum: {method_name}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "bvp_spectrum.png", dpi=170)
    plt.close(fig)


def create_case_packages(
    dataset_dir: Path,
    labels: pd.DataFrame,
    curves: pd.DataFrame,
    baseline_features: pd.DataFrame,
    feature_changes: pd.DataFrame,
    consensus: pd.DataFrame,
    config: Dict,
) -> None:
    case_cfg = config.get("sensitivity_cases", {})
    if not bool(case_cfg.get("enabled", True)):
        return
    selected = consensus.copy()
    if bool(case_cfg.get("multi_method_positive_only", True)):
        selected = selected[selected["cross_method_consensus"] == "multi_method_positive"]
    selected = selected.sort_values("mean_relative_improvement", ascending=False).head(
        int(case_cfg.get("max_cases_per_dataset", 30))
    )
    root = ensure_dir(dataset_dir / "sensitivity_cases")
    for case_index, (_, case) in enumerate(selected.iterrows()):
        uid = case["sample_uid"]
        folder = ensure_dir(root / safe_name(uid))
        label_rows = labels[labels["sample_uid"] == uid]
        curve_rows = curves[curves["sample_uid"] == uid]
        feature_rows = baseline_features[baseline_features["sample_uid"] == uid]
        summary = case.to_dict()
        summary["positive_methods"] = label_rows.loc[
            label_rows["response_type"] == "positive_sensitive", "method"
        ].tolist()
        (folder / "case_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        feature_rows.to_csv(folder / "baseline_roi_features.csv", index=False, encoding="utf-8-sig")
        curve_rows.to_csv(folder / "gamma_performance.csv", index=False, encoding="utf-8-sig")
        change_rows = feature_changes[feature_changes["sample_uid"] == uid]
        change_rows.to_csv(folder / "roi_feature_changes.csv", index=False, encoding="utf-8-sig")

        for metric, filename in [("MAE", "mae_gamma_curve.png"), ("SNR", "snr_gamma_curve.png")]:
            fig, ax = plt.subplots(figsize=(8, 5))
            for method, group in curve_rows.groupby("method"):
                ax.plot(group["gamma"], group[metric], marker="o", label=method)
            ax.set_xlabel("Gamma")
            ax.set_ylabel(metric)
            ax.set_title(f"{uid}: {metric} response")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(folder / filename, dpi=160)
            plt.close(fig)
        shutil.copyfile(folder / "mae_gamma_curve.png", folder / "method_comparison.png")

        source_examples = dataset_dir / "quality_examples" / safe_name(uid)
        if source_examples.exists():
            images = sorted(source_examples.glob("*.png"))
            if images:
                shutil.copyfile(images[0], folder / "roi_examples.png")

        generate_full = bool(case_cfg.get("generate_signal_diagnostics", True))
        max_full = int(case_cfg.get("max_signal_diagnostic_cases", 10))
        if generate_full and case_index < max_full and not label_rows.empty:
            positive_rows = label_rows[label_rows["response_type"] == "positive_sensitive"]
            representative = (
                positive_rows.sort_values("absolute_mae_improvement", ascending=False).iloc[0]
                if not positive_rows.empty else label_rows.iloc[0]
            )
            try:
                generate_signal_diagnostics(
                    str(case.get("input_path", representative.get("input_path", ""))),
                    float(representative.get("fps", 30.0)),
                    str(representative["method"]), float(representative["best_gamma"]),
                    folder, config,
                )
            except Exception as error:
                (folder / "signal_diagnostics_error.txt").write_text(repr(error), encoding="utf-8")
        elif not (folder / "roi_examples.png").exists():
            (folder / "roi_examples_not_available.txt").write_text(
                "未为该样本执行可选的信号诊断重跑。可在 sensitivity_cases 中开启 "
                "generate_signal_diagnostics 并提高 max_signal_diagnostic_cases。",
                encoding="utf-8",
            )


def analyse_dataset(dataset_dir: Path, config: Dict) -> None:
    raw_path = dataset_dir / "raw_results.csv"
    brightness_path = dataset_dir / "brightness_metrics.csv"
    if not raw_path.exists() or not brightness_path.exists():
        print(f"[WARN] Missing pre-ROI raw or brightness results in {dataset_dir}")
        return
    raw = pd.read_csv(raw_path)
    brightness = pd.read_csv(brightness_path)
    labels, curves = build_labels(raw, config)
    labels.to_csv(dataset_dir / "pre_roi_sensitivity_labels.csv", index=False, encoding="utf-8-sig")
    curves.to_csv(dataset_dir / "pre_roi_response_curves.csv", index=False, encoding="utf-8-sig")
    labels.to_csv(dataset_dir / "pre_roi_best_gamma.csv", index=False, encoding="utf-8-sig")

    consensus = cross_method_consensus(labels)
    consensus.to_csv(dataset_dir / "pre_roi_cross_method_consensus.csv", index=False, encoding="utf-8-sig")

    baseline = float(config.get("analysis", {}).get("baseline_gamma", 1.0))
    brightness["gamma"] = pd.to_numeric(brightness["gamma"], errors="coerce")
    baseline_features = brightness[np.isclose(brightness["gamma"], baseline)].copy()
    available_features = [feature for feature in FEATURES if feature in baseline_features.columns]
    baseline_columns = [
        "dataset", "sample_uid", "subject_id", "motion_type", "source_label", "input_type", "input_path",
        *available_features,
    ]
    baseline_features = baseline_features[[column for column in baseline_columns if column in baseline_features.columns]]
    baseline_features.to_csv(dataset_dir / "pre_roi_baseline_roi_features.csv", index=False, encoding="utf-8-sig")
    merged = labels.merge(baseline_features, on=["dataset", "sample_uid"], how="left", suffixes=("", "_feature"))
    merged.to_csv(dataset_dir / "pre_roi_sensitivity_with_features.csv", index=False, encoding="utf-8-sig")

    correlations = correlation_analysis(merged, config)
    correlations.to_csv(dataset_dir / "pre_roi_feature_improvement_correlation.csv", index=False, encoding="utf-8-sig")
    tests = group_tests(merged)
    tests.to_csv(dataset_dir / "pre_roi_feature_group_tests.csv", index=False, encoding="utf-8-sig")
    feature_changes = build_feature_changes(brightness, curves, baseline)
    feature_changes.to_csv(dataset_dir / "pre_roi_roi_feature_changes.csv", index=False, encoding="utf-8-sig")
    change_correlations = feature_change_correlations(feature_changes, config)
    change_correlations.to_csv(
        dataset_dir / "pre_roi_feature_change_vs_mae_delta.csv", index=False, encoding="utf-8-sig"
    )
    plot_outputs(dataset_dir, merged, labels, consensus)
    create_case_packages(
        dataset_dir, labels, curves, baseline_features, feature_changes, consensus, config
    )

    report = [f"# {dataset_dir.name} pre-ROI Gamma 敏感性与 ROI 特征分析\n"]
    report.append("本分析只使用 `pre_roi_gamma` 结果，基线特征只取 Gamma=1.0 的原始 ROI。\n")
    report.append("## 响应类型数量\n")
    for label, count in labels["response_type"].value_counts().items():
        report.append(f"- {label}: {count}")
    report.append("\n## 多算法一致性\n")
    for label, count in consensus["cross_method_consensus"].value_counts().items():
        report.append(f"- {label}: {count}")
    if not correlations.empty:
        top = correlations.sort_values("p_holm").head(10)
        report.append("\n## Holm 校正后最强的 ROI 特征关联\n")
        for _, row in top.iterrows():
            report.append(
                f"- {row['method']} / {row['feature']}: Spearman ρ={row['spearman_rho']:.3f}, "
                f"p_holm={row['p_holm']:.4g}, n={int(row['n'])}"
            )
    (dataset_dir / "pre_roi_sensitivity_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"[INFO] pre-ROI sensitivity analysis completed: {dataset_dir}")


def deep_update(base: Dict, override: Dict) -> Dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse pre-ROI Gamma-sensitive videos and ROI features.")
    parser.add_argument("--config", default="configs/sensitivity_analysis.yaml")
    parser.add_argument("--result-root", default=None)
    parser.add_argument("--datasets", default=None)
    args = parser.parse_args()
    overlay = load_config(args.config)
    source_config = overlay.get("source_pre_roi_config")
    config = deep_update(load_config(source_config), overlay) if source_config else overlay
    root = Path(args.result_root) if args.result_root else resolve_mode_root(config, "pre_roi_gamma")
    selected = None if not args.datasets or args.datasets.lower() == "all" else set(args.datasets.split(","))
    for path in sorted(root.iterdir()) if root.exists() else []:
        if not path.is_dir() or not (path / "raw_results.csv").exists():
            continue
        if selected is not None and path.name not in selected:
            continue
        analyse_dataset(path, config)


if __name__ == "__main__":
    main()
