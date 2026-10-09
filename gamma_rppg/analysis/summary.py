from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

from gamma_rppg.common.statistics import holm_adjust_list as holm_adjust
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gamma_rppg.common.config import ensure_dir, load_config, resolve_output_root


METRICS = ["RMSE", "MAE", "MAX", "PCC", "CCC", "SNR"]
SAMPLE_META_COLUMNS = [
    "video_idx", "video_name", "sample_id", "sample_uid", "subject_id", "trial_id",
    "motion_type", "source_label", "input_type", "input_path", "transform_mode",
]
RAW_UNIQUE_KEY = ["dataset", "sample_uid", "gamma", "method", "transform_mode"]
BRIGHTNESS_UNIQUE_KEY = ["dataset", "sample_uid", "gamma", "transform_mode"]


def parse_csv_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None or value.strip().lower() in {"", "all"}:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def classify_delta(delta: float, epsilon: float = 0.2) -> str:
    if pd.isna(delta):
        return "invalid"
    if delta < -epsilon:
        return "improved"
    if delta > epsilon:
        return "worsened"
    return "stable"


def _safe_filename(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("_") or "item"


def _ensure_columns(frame: pd.DataFrame, columns: Sequence[str], default="") -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column not in frame.columns:
            frame[column] = default
    return frame


def _legacy_sample_uid(row: pd.Series) -> str:
    dataset = str(row.get("dataset", "dataset"))
    source = str(row.get("source_label", ""))
    sample = str(row.get("sample_id", "") or row.get("video_name", "") or row.get("video_idx", ""))
    parts = [dataset]
    if source and source.lower() != "nan":
        parts.append(source)
    parts.append(sample)
    return "/".join(parts)


def normalize_raw(raw: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    raw = _ensure_columns(raw, SAMPLE_META_COLUMNS + ["dataset", "gamma", "method", "status"])
    missing_uid = raw["sample_uid"].fillna("").astype(str).isin(["", "nan"])
    raw.loc[missing_uid, "sample_uid"] = raw[missing_uid].apply(_legacy_sample_uid, axis=1)
    raw["transform_mode"] = raw["transform_mode"].replace("", np.nan).fillna("pre_roi_gamma").replace({"all": "pre_roi_gamma"}).astype(str)
    raw["gamma"] = pd.to_numeric(raw["gamma"], errors="coerce")
    for metric in METRICS:
        raw[metric] = pd.to_numeric(raw.get(metric), errors="coerce")
    before = len(raw)
    raw = raw.drop_duplicates(RAW_UNIQUE_KEY, keep="last").reset_index(drop=True)
    return raw, before - len(raw)


def normalize_brightness(frame: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    frame = _ensure_columns(frame, SAMPLE_META_COLUMNS + ["dataset", "gamma"])
    missing_uid = frame["sample_uid"].fillna("").astype(str).isin(["", "nan"])
    frame.loc[missing_uid, "sample_uid"] = frame[missing_uid].apply(_legacy_sample_uid, axis=1)
    frame["transform_mode"] = frame["transform_mode"].replace("", np.nan).fillna("pre_roi_gamma").replace({"all": "pre_roi_gamma"}).astype(str)
    frame["gamma"] = pd.to_numeric(frame["gamma"], errors="coerce")
    before = len(frame)
    frame = frame.drop_duplicates(BRIGHTNESS_UNIQUE_KEY, keep="last").reset_index(drop=True)
    return frame, before - len(frame)


def bootstrap_ci(values: np.ndarray, iterations: int, seed: int) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    if values.size == 1 or iterations <= 0:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=float)
    chunk = 250
    for start in range(0, iterations, chunk):
        count = min(chunk, iterations - start)
        indices = rng.integers(0, values.size, size=(count, values.size))
        means[start:start + count] = values[indices].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]).astype(float))


def aggregate_metrics(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    bootstrap_iterations: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for group_index, (group_key, group) in enumerate(frame.groupby(list(group_columns), dropna=False)):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = dict(zip(group_columns, group_key))
        for metric_index, metric in enumerate(METRICS):
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values)) if values.size else np.nan
            row[f"{metric}_median"] = float(np.median(values)) if values.size else np.nan
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else np.nan
            row[f"{metric}_count"] = int(values.size)
            low, high = bootstrap_ci(
                values, bootstrap_iterations, seed + group_index * 97 + metric_index * 1009
            )
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def expected_gammas(config: Dict, raw: pd.DataFrame) -> List[float]:
    configured = config.get("experiment", {}).get("gamma_values") if config else None
    if configured:
        return sorted({float(value) for value in configured})
    return sorted(raw["gamma"].dropna().unique().astype(float).tolist())


def complete_case_rows(ok: pd.DataFrame, gammas: Sequence[float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    selected = []
    sample_rows = []
    gamma_set = {round(float(value), 10) for value in gammas}
    group_columns = ["dataset", "method", "transform_mode", "sample_uid"]
    for key, group in ok.groupby(group_columns, dropna=False):
        present = {round(float(value), 10) for value in group["gamma"].dropna().unique()}
        complete = gamma_set.issubset(present)
        sample_row = dict(zip(group_columns, key))
        sample_row.update({
            "expected_gamma_count": len(gamma_set),
            "successful_gamma_count": len(present & gamma_set),
            "complete": bool(complete),
        })
        sample_rows.append(sample_row)
        if complete:
            selected.append(group[group["gamma"].round(10).isin(gamma_set)])
    paired = pd.concat(selected, ignore_index=True) if selected else ok.iloc[0:0].copy()
    return paired, pd.DataFrame(sample_rows)


def success_rate_table(raw: pd.DataFrame, gammas: Sequence[float]) -> pd.DataFrame:
    rows = []
    base_columns = ["dataset", "method", "transform_mode"]
    for key, method_group in raw.groupby(base_columns, dropna=False):
        union_samples = method_group["sample_uid"].nunique()
        for gamma in gammas:
            group = method_group[np.isclose(method_group["gamma"], gamma)]
            success = group.loc[group["status"] == "ok", "sample_uid"].nunique()
            attempted = group["sample_uid"].nunique()
            row = dict(zip(base_columns, key))
            row.update({
                "gamma": gamma,
                "dataset_method_sample_count": int(union_samples),
                "attempted_count": int(attempted),
                "success_count": int(success),
                "failed_count": int(max(attempted - success, 0)),
                "missing_task_count": int(max(union_samples - attempted, 0)),
                "success_rate": float(success / union_samples) if union_samples else np.nan,
            })
            rows.append(row)
    return pd.DataFrame(rows)




def paired_rank_biserial(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences) & (differences != 0)]
    if differences.size == 0:
        return 0.0
    ranks = pd.Series(np.abs(differences)).rank(method="average").to_numpy()
    positive = ranks[differences > 0].sum()
    negative = ranks[differences < 0].sum()
    total = positive + negative
    return float((positive - negative) / total) if total else 0.0


def statistical_tests(
    paired: pd.DataFrame,
    gammas: Sequence[float],
    baseline_gamma: float,
    metrics: Sequence[str],
) -> pd.DataFrame:
    try:
        from scipy.stats import friedmanchisquare, wilcoxon
    except Exception as error:
        return pd.DataFrame([{
            "test": "unavailable", "status": f"scipy unavailable: {error!r}"
        }])

    rows = []
    grouping = ["dataset", "method", "transform_mode"]
    for group_key, group in paired.groupby(grouping, dropna=False):
        group_identity = dict(zip(grouping, group_key))
        for metric in metrics:
            pivot = group.pivot_table(index="sample_uid", columns="gamma", values=metric, aggfunc="last")
            available_gammas = [gamma for gamma in gammas if gamma in pivot.columns]
            complete = pivot[available_gammas].dropna() if available_gammas else pd.DataFrame()
            if complete.shape[0] >= 2 and len(available_gammas) >= 3:
                try:
                    statistic, pvalue = friedmanchisquare(
                        *[complete[gamma].to_numpy() for gamma in available_gammas]
                    )
                    status = "ok"
                except Exception as error:
                    statistic, pvalue, status = np.nan, np.nan, repr(error)
            else:
                statistic, pvalue, status = np.nan, np.nan, "insufficient_complete_samples"
            friedman_row = dict(group_identity)
            friedman_row.update({
                "metric": metric, "test": "friedman", "baseline_gamma": baseline_gamma,
                "comparison_gamma": np.nan, "n": int(complete.shape[0]),
                "statistic": statistic, "p_value": pvalue, "p_holm": np.nan,
                "effect_rank_biserial": np.nan, "mean_delta_vs_baseline": np.nan,
                "status": status,
            })
            rows.append(friedman_row)

            pairwise_rows = []
            raw_pvalues = []
            if baseline_gamma in pivot.columns:
                for gamma in available_gammas:
                    if np.isclose(gamma, baseline_gamma):
                        continue
                    pair = pivot[[baseline_gamma, gamma]].dropna()
                    differences = pair[gamma].to_numpy() - pair[baseline_gamma].to_numpy()
                    if pair.shape[0] >= 2:
                        if np.allclose(differences, 0):
                            statistic, pvalue, status = 0.0, 1.0, "all_differences_zero"
                        else:
                            try:
                                kwargs = {"zero_method": "wilcox", "alternative": "two-sided"}
                                has_zero = bool(np.any(np.isclose(differences, 0.0)))
                                try:
                                    kwargs["method"] = "approx" if has_zero else "auto"
                                    statistic, pvalue = wilcoxon(
                                        pair[gamma].to_numpy(), pair[baseline_gamma].to_numpy(), **kwargs
                                    )
                                except TypeError:
                                    kwargs.pop("method", None)
                                    kwargs["mode"] = "approx" if has_zero else "auto"
                                    statistic, pvalue = wilcoxon(
                                        pair[gamma].to_numpy(), pair[baseline_gamma].to_numpy(), **kwargs
                                    )
                                status = "ok_approx" if has_zero else "ok"
                            except Exception as error:
                                statistic, pvalue, status = np.nan, np.nan, repr(error)
                    else:
                        statistic, pvalue, status = np.nan, np.nan, "insufficient_pairs"
                    pairwise_row = dict(group_identity)
                    pairwise_row.update({
                        "metric": metric, "test": "wilcoxon_vs_baseline",
                        "baseline_gamma": baseline_gamma, "comparison_gamma": gamma,
                        "n": int(pair.shape[0]), "statistic": statistic, "p_value": pvalue,
                        "p_holm": np.nan,
                        "effect_rank_biserial": paired_rank_biserial(differences),
                        "mean_delta_vs_baseline": float(np.mean(differences)) if differences.size else np.nan,
                        "status": status,
                    })
                    pairwise_rows.append(pairwise_row)
                    raw_pvalues.append(float(pvalue) if np.isfinite(pvalue) else 1.0)
            adjusted = holm_adjust(raw_pvalues)
            for row, adjusted_p in zip(pairwise_rows, adjusted):
                row["p_holm"] = adjusted_p
            rows.extend(pairwise_rows)
    return pd.DataFrame(rows)


def metadata_from_group(group: pd.DataFrame) -> dict:
    first = group.iloc[0]
    return {column: first.get(column, "") for column in SAMPLE_META_COLUMNS}


def delta_and_best(ok: pd.DataFrame, baseline_gamma: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (method, transform_mode, sample_uid), group in ok.groupby(
        ["method", "transform_mode", "sample_uid"], dropna=False
    ):
        group = group.sort_values("gamma")
        baseline = group[np.isclose(group["gamma"], baseline_gamma)]
        valid_mae = group.dropna(subset=["MAE"])
        if baseline.empty or valid_mae.empty:
            continue
        base = baseline.iloc[-1]
        best = valid_mae.loc[valid_mae["MAE"].idxmin()]
        metadata = metadata_from_group(group)
        for _, item in group.iterrows():
            rows.append({
                "dataset": item["dataset"], "method": method, "transform_mode": transform_mode,
                "sample_uid": sample_uid, **metadata, "gamma": item["gamma"],
                "MAE": item["MAE"], "SNR": item["SNR"],
                "MAE_delta_vs_baseline": item["MAE"] - base["MAE"],
                "SNR_delta_vs_baseline": item["SNR"] - base["SNR"],
                "base_MAE": base["MAE"], "best_gamma": best["gamma"],
                "best_MAE": best["MAE"], "best_delta_MAE": best["MAE"] - base["MAE"],
                "response_type": classify_delta(best["MAE"] - base["MAE"]),
            })
    delta = pd.DataFrame(rows)
    if delta.empty:
        return delta, pd.DataFrame()
    best_columns = list(dict.fromkeys([
        "dataset", "method", "transform_mode", "sample_uid", *SAMPLE_META_COLUMNS,
        "base_MAE", "best_gamma", "best_MAE", "best_delta_MAE", "response_type",
    ]))
    best = delta.drop_duplicates(["method", "transform_mode", "sample_uid"])[best_columns]
    return delta, best


def macro_summary(
    frame: pd.DataFrame,
    level_column: str,
    bootstrap_iterations: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    available = frame[frame[level_column].fillna("").astype(str).str.len() > 0].copy()
    if available.empty:
        return pd.DataFrame(), pd.DataFrame()
    level_columns = ["dataset", level_column, "method", "transform_mode", "gamma"]
    detail = aggregate_metrics(
        available, level_columns, bootstrap_iterations, seed
    )
    level_means = available.groupby(level_columns, dropna=False)[METRICS].mean().reset_index()
    macro = aggregate_metrics(
        level_means,
        ["dataset", "method", "transform_mode", "gamma"],
        bootstrap_iterations,
        seed + 100003,
    )
    macro = macro.rename(columns={f"{metric}_count": f"{metric}_{level_column}_count" for metric in METRICS})
    return detail, macro


def brightness_correlations(
    delta: pd.DataFrame,
    brightness: pd.DataFrame,
    baseline_gamma: float,
) -> pd.DataFrame:
    if delta.empty or brightness.empty:
        return pd.DataFrame()
    baseline = brightness[np.isclose(brightness["gamma"], baseline_gamma)].copy()
    merged = delta.merge(
        baseline,
        on=["dataset", "sample_uid", "transform_mode"],
        suffixes=("", "_brightness_base"),
        how="left",
    )
    columns = [
        "mean_rgb", "std_rgb", "p95_rgb", "p99_rgb", "sat_ratio_250", "sat_ratio_245",
        "dark_ratio_5", "dark_ratio_10", "rg_ratio", "bg_ratio", "raw_cv_r", "raw_cv_g",
        "raw_cv_b", "valid_pixel_ratio_mean", "roi_area_mean",
    ]
    rows = []
    for (method, transform_mode), method_group in merged.groupby(["method", "transform_mode"]):
        for gamma, gamma_group in method_group.groupby("gamma"):
            for column in columns:
                if column not in gamma_group.columns:
                    continue
                valid = gamma_group[[column, "MAE_delta_vs_baseline", "SNR_delta_vs_baseline"]].dropna()
                if len(valid) < 3:
                    continue
                x = valid[column]
                mae_delta = valid["MAE_delta_vs_baseline"]
                snr_delta = valid["SNR_delta_vs_baseline"]
                corr_mae = x.corr(mae_delta) if x.std() > 0 and mae_delta.std() > 0 else np.nan
                corr_snr = x.corr(snr_delta) if x.std() > 0 and snr_delta.std() > 0 else np.nan
                rows.append({
                    "dataset": str(gamma_group["dataset"].iloc[0]), "method": method,
                    "transform_mode": transform_mode, "gamma": gamma,
                    "brightness_metric": column,
                    "corr_with_MAE_delta": corr_mae,
                    "corr_with_SNR_delta": corr_snr,
                    "n": len(valid),
                })
    return pd.DataFrame(rows)


def quality_control(raw: pd.DataFrame, brightness: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    thresholds = config.get("analysis", {}).get("quality_thresholds", {})
    min_face_ratio = float(thresholds.get("min_face_detection_ratio", 0.80))
    min_roi_ratio = float(thresholds.get("min_geometric_roi_frame_ratio", 0.80))
    min_valid_pixel_ratio = float(thresholds.get("min_valid_pixel_ratio", 0.10))
    rows = []
    if not brightness.empty:
        for _, item in brightness.iterrows():
            frame_count = float(item.get("frame_count", 0) or 0)
            face_ratio = float(item.get("face_detected_frames", 0) or 0) / frame_count if frame_count else 0.0
            roi_ratio = float(item.get("geometric_roi_frames", 0) or 0) / frame_count if frame_count else 0.0
            valid_ratio = pd.to_numeric(pd.Series([item.get("valid_pixel_ratio_mean")]), errors="coerce").iloc[0]
            issues = []
            if face_ratio < min_face_ratio:
                issues.append("low_face_detection_ratio")
            if roi_ratio < min_roi_ratio:
                issues.append("low_geometric_roi_frame_ratio")
            if pd.isna(valid_ratio) or valid_ratio < min_valid_pixel_ratio:
                issues.append("low_valid_pixel_ratio")
            if issues:
                rows.append({
                    "dataset": item.get("dataset"), "sample_uid": item.get("sample_uid"),
                    "gamma": item.get("gamma"), "transform_mode": item.get("transform_mode"),
                    "method": "", "status": "quality_warning", "issues": ";".join(issues),
                    "face_detection_ratio": face_ratio, "geometric_roi_frame_ratio": roi_ratio,
                    "valid_pixel_ratio_mean": valid_ratio,
                })
    failures = raw[raw["status"] != "ok"]
    for _, item in failures.iterrows():
        rows.append({
            "dataset": item.get("dataset"), "sample_uid": item.get("sample_uid"),
            "gamma": item.get("gamma"), "transform_mode": item.get("transform_mode"),
            "method": item.get("method"), "status": item.get("status"),
            "issues": item.get("error_message", ""), "face_detection_ratio": np.nan,
            "geometric_roi_frame_ratio": np.nan, "valid_pixel_ratio_mean": np.nan,
        })
    detail = pd.DataFrame(rows)
    if detail.empty:
        summary = pd.DataFrame(columns=["status", "issue_count"])
    else:
        summary = detail.groupby("status", dropna=False).size().reset_index(name="issue_count")
    return detail, summary


def plot_metric_with_ci(summary: pd.DataFrame, dataset: str, metric: str, output: Path, label: str):
    if summary.empty:
        return
    figure = plt.figure(figsize=(9, 5.5))
    for (method, transform), group in summary.groupby(["method", "transform_mode"], dropna=False):
        group = group.sort_values("gamma")
        line_label = method if summary["transform_mode"].nunique() == 1 else f"{method} ({transform})"
        line, = plt.plot(group["gamma"], group[f"{metric}_mean"], marker="o", label=line_label)
        low = group[f"{metric}_ci95_low"].to_numpy(dtype=float)
        high = group[f"{metric}_ci95_high"].to_numpy(dtype=float)
        x = group["gamma"].to_numpy(dtype=float)
        if np.isfinite(low).any() and np.isfinite(high).any():
            plt.fill_between(x, low, high, alpha=0.15, color=line.get_color())
    plt.xlabel("Gamma value")
    plt.ylabel(f"Mean {metric}")
    plt.title(f"{dataset}: {label} {metric} with bootstrap 95% CI")
    plt.grid(True, alpha=0.3)
    plt.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_success_rate(table: pd.DataFrame, dataset: str, output: Path):
    if table.empty:
        return
    figure = plt.figure(figsize=(9, 5.5))
    for method, group in table.groupby("method"):
        group = group.sort_values("gamma")
        plt.plot(group["gamma"], group["success_rate"] * 100.0, marker="o", label=method)
    plt.xlabel("Gamma value")
    plt.ylabel("Success rate (%)")
    plt.ylim(-2, 102)
    plt.title(f"{dataset}: task success rate vs Gamma")
    plt.grid(True, alpha=0.3)
    plt.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_boxplots(paired: pd.DataFrame, dataset: str, output_dir: Path):
    for method, group in paired.groupby("method"):
        gamma_values = sorted(group["gamma"].dropna().unique())
        arrays = [group[np.isclose(group["gamma"], gamma)]["MAE"].dropna().to_numpy() for gamma in gamma_values]
        if not arrays or all(len(array) == 0 for array in arrays):
            continue
        figure = plt.figure(figsize=(10, 5.5))
        plt.boxplot(arrays, labels=[f"{gamma:g}" for gamma in gamma_values], showfliers=False)
        plt.xlabel("Gamma value")
        plt.ylabel("MAE")
        plt.title(f"{dataset}: paired-sample MAE distribution — {method}")
        plt.grid(True, axis="y", alpha=0.3)
        figure.tight_layout()
        figure.savefig(output_dir / f"{dataset}_paired_MAE_boxplot_{_safe_filename(method)}.png", dpi=180)
        plt.close(figure)


def plot_paired_trajectories(paired: pd.DataFrame, dataset: str, output_dir: Path):
    for method, group in paired.groupby("method"):
        pivot = group.pivot_table(index="sample_uid", columns="gamma", values="MAE", aggfunc="last")
        if pivot.empty:
            continue
        figure = plt.figure(figsize=(10, 5.5))
        for _, row in pivot.iterrows():
            valid = row.dropna()
            plt.plot(valid.index.astype(float), valid.values, alpha=0.25, linewidth=0.8)
        mean_series = pivot.mean(axis=0)
        plt.plot(mean_series.index.astype(float), mean_series.values, marker="o", linewidth=2.5, label="Paired mean")
        plt.xlabel("Gamma value")
        plt.ylabel("MAE")
        plt.title(f"{dataset}: paired video MAE trajectories — {method}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        figure.tight_layout()
        figure.savefig(output_dir / f"{dataset}_paired_MAE_trajectories_{_safe_filename(method)}.png", dpi=180)
        plt.close(figure)


def plot_best_gamma(best: pd.DataFrame, dataset: str, output_dir: Path):
    for method, group in best.groupby("method"):
        counts = group["best_gamma"].value_counts().sort_index()
        if counts.empty:
            continue
        figure = plt.figure(figsize=(8, 5))
        plt.bar([f"{value:g}" for value in counts.index], counts.values)
        plt.xlabel("Best Gamma by video")
        plt.ylabel("Video count")
        plt.title(f"{dataset}: best-Gamma distribution — {method}")
        plt.grid(True, axis="y", alpha=0.3)
        figure.tight_layout()
        figure.savefig(output_dir / f"{dataset}_best_gamma_distribution_{_safe_filename(method)}.png", dpi=180)
        plt.close(figure)


def plot_macro_mae(summary: pd.DataFrame, dataset: str, title_suffix: str, output: Path):
    if summary.empty:
        return
    figure = plt.figure(figsize=(9, 5.5))
    for method, group in summary.groupby("method"):
        group = group.sort_values("gamma")
        plt.plot(group["gamma"], group["MAE_mean"], marker="o", label=method)
    plt.xlabel("Gamma value")
    plt.ylabel("Macro-average MAE")
    plt.title(f"{dataset}: {title_suffix}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def write_report(
    dataset: str,
    output_dir: Path,
    primary: pd.DataFrame,
    paired_summary: pd.DataFrame,
    success: pd.DataFrame,
    tests: pd.DataFrame,
    complete_samples: pd.DataFrame,
    duplicate_count: int,
    baseline_gamma: float,
):
    lines = [f"# {dataset} Gamma-rPPG 分析报告\n"]
    lines.append("## 统计口径\n")
    lines.append(
        "- `summary_by_gamma.csv`：所有成功任务的描述统计。\n"
        "- `summary_paired_by_gamma.csv`：只保留在全部 Gamma 下均成功的完整配对视频，作为主比较口径。\n"
        "- `success_rate_by_gamma.csv`：显示每个 Gamma 的成功、失败和未完成任务数量。\n"
        f"- 基准 Gamma 为 `{baseline_gamma:g}`；统计置信区间使用视频级 bootstrap。\n"
    )
    if dataset.upper() == "PURE":
        lines.append(
            "PURE 的六类运动来源始终作为一个逻辑数据集统一计算主结果。"
            "运动条件宏平均和受试者宏平均仅是补充平衡统计，不会拆分为多个数据集。\n"
        )
    if duplicate_count:
        lines.append(f"汇总读取时检测并移除了 **{duplicate_count}** 条旧式重复任务记录。\n")

    complete_count = int(complete_samples["complete"].astype(bool).sum()) if not complete_samples.empty else 0
    attempted_count = int(len(complete_samples)) if not complete_samples.empty else 0
    lines.append(f"完整配对样本覆盖：**{complete_count}/{attempted_count}** 个方法—样本组合。\n")

    lines.append("## 完整配对样本的最佳平均 MAE\n")
    source = paired_summary if not paired_summary.empty else primary
    for method, group in source.groupby("method"):
        valid = group.dropna(subset=["MAE_mean"])
        if valid.empty:
            continue
        best = valid.loc[valid["MAE_mean"].idxmin()]
        baseline = valid[np.isclose(valid["gamma"], baseline_gamma)]
        base_value = float(baseline.iloc[0]["MAE_mean"]) if not baseline.empty else np.nan
        lines.append(
            f"- **{method}**：最佳 Gamma={best['gamma']:g}，平均 MAE={best['MAE_mean']:.4f}，"
            f"相对基准变化={best['MAE_mean'] - base_value:.4f}。"
        )

    lines.append("\n## 成功率提示\n")
    for method, group in success.groupby("method"):
        minimum = group.loc[group["success_rate"].idxmin()]
        lines.append(
            f"- **{method}**：最低成功率出现在 Gamma={minimum['gamma']:g}，"
            f"成功率={minimum['success_rate'] * 100:.1f}%。"
        )

    lines.append("\n## 统计检验\n")
    if tests.empty or "test" not in tests.columns or "p_holm" not in tests.columns:
        lines.append("未生成完整统计检验结果；请确认 SciPy 已安装且配对样本数量充足。")
    else:
        significant = tests[(tests["test"] == "wilcoxon_vs_baseline") & (pd.to_numeric(tests["p_holm"], errors="coerce") < 0.05)]
        lines.append(
            f"Holm 校正后显著的 Gamma—基准配对比较共有 **{len(significant)}** 项。"
            "详细数值见 `paired_statistical_tests.csv`。"
        )

    lines.append("\n## 图表\n")
    lines.append(
        "主趋势图使用完整配对样本及 bootstrap 95% 置信区间；另提供成功率曲线、"
        "MAE 箱线图、视频级配对轨迹、最佳 Gamma 分布和质量控制示例。"
    )
    (output_dir / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def summarize_dataset(dataset_dir: Path, config: Dict):
    raw_path = dataset_dir / "raw_results.csv"
    if not raw_path.exists():
        return
    dataset = dataset_dir.name
    analysis = config.get("analysis", {}) if config else {}
    bootstrap_iterations = int(analysis.get("bootstrap_iterations", 2000))
    seed = int(analysis.get("random_seed", 2026))
    baseline_gamma = float(analysis.get("baseline_gamma", 1.0))
    statistical_metrics = analysis.get("statistical_metrics", ["MAE", "SNR"])

    raw, duplicate_count = normalize_raw(pd.read_csv(raw_path))
    raw.to_csv(dataset_dir / "raw_results_deduplicated.csv", index=False, encoding="utf-8-sig")
    gammas = expected_gammas(config, raw)
    ok = raw[raw["status"] == "ok"].copy()
    if ok.empty:
        print(f"[WARN] No successful rows in {raw_path}")
        success_rate_table(raw, gammas).to_csv(
            dataset_dir / "success_rate_by_gamma.csv", index=False, encoding="utf-8-sig"
        )
        return

    primary = aggregate_metrics(
        ok, ["dataset", "method", "transform_mode", "gamma"], bootstrap_iterations, seed
    )
    primary.to_csv(dataset_dir / "summary_by_gamma.csv", index=False, encoding="utf-8-sig")

    paired, complete_samples = complete_case_rows(ok, gammas)
    complete_samples.to_csv(dataset_dir / "complete_case_samples.csv", index=False, encoding="utf-8-sig")
    paired_summary = aggregate_metrics(
        paired, ["dataset", "method", "transform_mode", "gamma"], bootstrap_iterations, seed + 11
    ) if not paired.empty else pd.DataFrame()
    paired_summary.to_csv(
        dataset_dir / "summary_paired_by_gamma.csv", index=False, encoding="utf-8-sig"
    )

    success = success_rate_table(raw, gammas)
    success.to_csv(dataset_dir / "success_rate_by_gamma.csv", index=False, encoding="utf-8-sig")

    tests = statistical_tests(paired, gammas, baseline_gamma, statistical_metrics)
    tests.to_csv(dataset_dir / "paired_statistical_tests.csv", index=False, encoding="utf-8-sig")

    delta, best = delta_and_best(ok, baseline_gamma)
    if not delta.empty:
        delta.to_csv(dataset_dir / "per_video_delta.csv", index=False, encoding="utf-8-sig")
    if not best.empty:
        best.to_csv(dataset_dir / "per_video_best_gamma.csv", index=False, encoding="utf-8-sig")
        best.groupby(["method", "transform_mode", "response_type"]).size().reset_index(name="count").to_csv(
            dataset_dir / "gamma_response_type.csv", index=False, encoding="utf-8-sig"
        )

    motion_level, motion_macro = macro_summary(
        ok, "motion_type", bootstrap_iterations, seed + 21
    )
    subject_level, subject_macro = macro_summary(
        ok, "subject_id", bootstrap_iterations, seed + 31
    )
    if not motion_level.empty:
        motion_level.to_csv(dataset_dir / "summary_by_motion_gamma.csv", index=False, encoding="utf-8-sig")
        motion_macro.to_csv(dataset_dir / "summary_motion_macro_by_gamma.csv", index=False, encoding="utf-8-sig")
    if not subject_level.empty:
        subject_level.to_csv(dataset_dir / "summary_by_subject_gamma.csv", index=False, encoding="utf-8-sig")
        subject_macro.to_csv(dataset_dir / "summary_subject_macro_by_gamma.csv", index=False, encoding="utf-8-sig")

    brightness_path = dataset_dir / "brightness_metrics.csv"
    brightness = pd.DataFrame()
    brightness_duplicates = 0
    if brightness_path.exists():
        brightness, brightness_duplicates = normalize_brightness(pd.read_csv(brightness_path))
        brightness.to_csv(
            dataset_dir / "brightness_metrics_deduplicated.csv", index=False, encoding="utf-8-sig"
        )
        correlation = brightness_correlations(delta, brightness, baseline_gamma)
        if not correlation.empty:
            correlation.to_csv(
                dataset_dir / "brightness_delta_correlation.csv", index=False, encoding="utf-8-sig"
            )

    qc_detail, qc_summary = quality_control(raw, brightness, config)
    qc_detail.to_csv(dataset_dir / "quality_control.csv", index=False, encoding="utf-8-sig")
    qc_summary.to_csv(dataset_dir / "quality_control_summary.csv", index=False, encoding="utf-8-sig")

    figures = ensure_dir(dataset_dir / "figures")
    chart_source = paired_summary if not paired_summary.empty else primary
    chart_label = "paired-sample mean" if not paired_summary.empty else "successful-task mean"
    for metric in METRICS:
        plot_metric_with_ci(
            chart_source, dataset, metric,
            figures / f"{dataset}_paired_mean_{metric}_ci95.png", chart_label,
        )
    plot_success_rate(success, dataset, figures / f"{dataset}_success_rate.png")
    if not paired.empty:
        plot_boxplots(paired, dataset, figures)
        plot_paired_trajectories(paired, dataset, figures)
    if not best.empty:
        plot_best_gamma(best, dataset, figures)
    if not motion_macro.empty:
        plot_macro_mae(
            motion_macro, dataset, "motion-condition macro-average MAE",
            figures / f"{dataset}_motion_macro_MAE.png",
        )
    if not subject_macro.empty:
        plot_macro_mae(
            subject_macro, dataset, "subject macro-average MAE",
            figures / f"{dataset}_subject_macro_MAE.png",
        )

    write_report(
        dataset, dataset_dir, primary, paired_summary, success, tests, complete_samples,
        duplicate_count + brightness_duplicates, baseline_gamma,
    )
    print(f"[INFO] Summarized dataset folder: {dataset_dir}")


def dataset_dirs(root: Path, selected_names: Optional[Iterable[str]] = None) -> List[Path]:
    selected = set(selected_names) if selected_names is not None else None
    if not root.exists():
        return []
    return [
        path for path in sorted(root.iterdir())
        if path.is_dir() and (path / "raw_results.csv").exists()
        and (selected is None or path.name in selected)
    ]


def main():
    parser = argparse.ArgumentParser(description="Summarize every dataset folder independently.")
    parser.add_argument("--config", default="configs/full_pre_roi_experiment.yaml")
    parser.add_argument(
        "--result-root", default=None,
        help="Override the complete experiment result directory; otherwise derive it from YAML.",
    )
    parser.add_argument("--datasets", default=None, help="Comma-separated result folder names.")
    args = parser.parse_args()

    config = load_config(args.config)
    root = resolve_output_root(config, args.result_root)
    directories = dataset_dirs(root, parse_csv_list(args.datasets))
    if not directories:
        print(f"[WARN] No dataset result folders found under {root}")
        return
    for directory in directories:
        summarize_dataset(directory, config)
    print(f"[INFO] Per-dataset summaries completed under {root}; no cross-dataset merge was created.")


if __name__ == "__main__":
    main()
