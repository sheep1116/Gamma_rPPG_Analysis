from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from gamma_rppg.common.config import dump_resolved_config
from scipy import stats

from .io import discover_dataset_dirs, ensure_output_tree, load_dataset_tables, resolve_result_root, write_json
from .plotting import (
    plot_consensus_distribution,
    plot_consensus_feature,
    plot_failure_heatmap,
    plot_failure_rate,
    plot_forest,
    plot_group_distribution,
    plot_mechanism_curves,
    plot_paired_heatmaps,
    plot_pixel_composition,
    plot_scatter,
)
from gamma_rppg.common.statistics import (
    bootstrap_median_ci,
    cliffs_delta,
    holm_adjust,
    spearman_with_bootstrap,
    wilcoxon_against_zero,
    within_video_rhos,
)

LOG = logging.getLogger(__name__)


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def _method_list(df: pd.DataFrame, selected: list[str] | None) -> list[str]:
    methods = sorted(df["method"].dropna().astype(str).unique().tolist())
    if selected:
        wanted = set(selected)
        methods = [m for m in methods if m in wanted]
    return methods


def compute_baseline_associations(
    sensitivity: pd.DataFrame,
    features: list[str],
    methods: list[str],
    min_n: int,
    iterations: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    d0 = sensitivity[sensitivity.get("eligible_for_sensitivity", True).fillna(False).astype(bool)].copy()
    for method in methods:
        dm = d0[d0["method"] == method]
        for feature in features:
            if feature not in dm.columns:
                continue
            result = spearman_with_bootstrap(dm[feature], dm["absolute_mae_improvement"], iterations, rng)
            rows.append({
                "dataset": str(dm["dataset"].dropna().iloc[0]) if not dm.empty else "",
                "method": method,
                "feature": feature,
                "n": result["n"],
                "spearman_rho": result["rho"],
                "rho_ci95_low": result["ci_low"],
                "rho_ci95_high": result["ci_high"],
                "p_value": result["p"],
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out.loc[out["n"] < min_n, ["spearman_rho", "rho_ci95_low", "rho_ci95_high", "p_value"]] = np.nan
    out["p_holm"] = np.nan
    for method, idx in out.groupby("method").groups.items():
        out.loc[idx, "p_holm"] = holm_adjust(out.loc[idx, "p_value"])
    return out


def compute_group_comparisons(
    sensitivity: pd.DataFrame,
    features: list[str],
    methods: list[str],
    positive: str,
    reference: str,
    min_group_n: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in methods:
        dm = sensitivity[(sensitivity["method"] == method) & sensitivity["response_type"].isin([positive, reference])]
        for feature in features:
            if feature not in dm.columns:
                continue
            a = pd.to_numeric(dm.loc[dm["response_type"] == positive, feature], errors="coerce").dropna().to_numpy()
            b = pd.to_numeric(dm.loc[dm["response_type"] == reference, feature], errors="coerce").dropna().to_numpy()
            if len(a) >= min_group_n and len(b) >= min_group_n:
                u = stats.mannwhitneyu(a, b, alternative="two-sided")
                p = float(u.pvalue)
                statistic = float(u.statistic)
                delta = cliffs_delta(a, b)
            else:
                p = statistic = delta = np.nan
            rows.append({
                "dataset": str(dm["dataset"].dropna().iloc[0]) if not dm.empty else "",
                "method": method,
                "feature": feature,
                "positive_group": positive,
                "reference_group": reference,
                "n_positive": len(a),
                "n_reference": len(b),
                "median_positive": float(np.median(a)) if len(a) else np.nan,
                "median_reference": float(np.median(b)) if len(b) else np.nan,
                "mannwhitney_u": statistic,
                "p_value": p,
                "cliffs_delta_positive_minus_reference": delta,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_holm"] = np.nan
    for method, idx in out.groupby("method").groups.items():
        out.loc[idx, "p_holm"] = holm_adjust(out.loc[idx, "p_value"])
    return out


def compute_within_video_associations(
    changes: pd.DataFrame,
    mechanism_features: list[str],
    methods: list[str],
    response_types: list[str],
    min_points: int,
    iterations: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    d0 = changes.copy()
    d0["MAE_benefit"] = -pd.to_numeric(d0["MAE_delta_vs_baseline"], errors="coerce")
    d0 = d0[d0["gamma"] != 1.0]
    rows: list[dict[str, Any]] = []
    for method in methods:
        for response_type in response_types:
            dm = d0[(d0["method"] == method) & (d0["response_type"] == response_type)]
            for feature in mechanism_features:
                delta = f"delta_{feature}"
                if delta not in dm.columns:
                    continue
                rhos = within_video_rhos(dm, delta, "MAE_benefit", min_points)
                med, lo, hi = bootstrap_median_ci(rhos, iterations, rng)
                rows.append({
                    "dataset": str(dm["dataset"].dropna().iloc[0]) if not dm.empty else "",
                    "method": method,
                    "response_type": response_type,
                    "feature": feature,
                    "feature_change": delta,
                    "n_videos_with_rho": len(rhos),
                    "median_within_video_rho": med,
                    "rho_ci95_low": lo,
                    "rho_ci95_high": hi,
                    "positive_rho_fraction": float(np.mean(rhos > 0)) if len(rhos) else np.nan,
                    "negative_rho_fraction": float(np.mean(rhos < 0)) if len(rhos) else np.nan,
                    "p_value_wilcoxon_vs_zero": wilcoxon_against_zero(rhos),
                })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_holm"] = np.nan
    for (_, _), idx in out.groupby(["method", "response_type"]).groups.items():
        out.loc[idx, "p_holm"] = holm_adjust(out.loc[idx, "p_value_wilcoxon_vs_zero"])
    return out


def compute_gamma_group_summary(
    changes: pd.DataFrame,
    mechanism_features: list[str],
    methods: list[str],
    response_types: list[str],
    iterations: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    d0 = changes.copy()
    d0["MAE_benefit"] = -pd.to_numeric(d0["MAE_delta_vs_baseline"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for method in methods:
        dm = d0[(d0["method"] == method) & d0["response_type"].isin(response_types)]
        for feature in mechanism_features:
            delta = f"delta_{feature}"
            if delta not in dm.columns:
                continue
            for (response, gamma), g in dm.groupby(["response_type", "gamma"], dropna=False):
                bmed, blo, bhi = bootstrap_median_ci(g["MAE_benefit"], iterations, rng)
                fmed, flo, fhi = bootstrap_median_ci(g[delta], iterations, rng)
                rows.append({
                    "dataset": str(g["dataset"].dropna().iloc[0]) if not g.empty else "",
                    "method": method,
                    "response_type": response,
                    "gamma": float(gamma),
                    "feature": feature,
                    "n": int(g["sample_uid"].nunique()),
                    "benefit_median": bmed,
                    "benefit_ci_low": blo,
                    "benefit_ci_high": bhi,
                    "feature_delta_median": fmed,
                    "feature_delta_ci_low": flo,
                    "feature_delta_ci_high": fhi,
                })
    return pd.DataFrame(rows)


def compute_pixel_composition_summary(
    brightness: pd.DataFrame,
    sensitivity: pd.DataFrame,
    methods: list[str],
    response_types: list[str],
) -> pd.DataFrame:
    labels = sensitivity[["sample_uid", "method", "response_type"]].drop_duplicates()
    merged = labels.merge(
        brightness[["sample_uid", "gamma", "valid_pixel_ratio_mean", "rejected_dark_pixel_ratio", "rejected_bright_pixel_ratio"]],
        on="sample_uid", how="inner",
    )
    merged = merged[merged["method"].isin(methods) & merged["response_type"].isin(response_types)]
    if merged.empty:
        return merged
    return (
        merged.groupby(["method", "response_type", "gamma"], as_index=False)
        .agg(
            n=("sample_uid", "nunique"),
            valid_pixel_ratio_median=("valid_pixel_ratio_mean", "median"),
            rejected_dark_pixel_ratio_median=("rejected_dark_pixel_ratio", "median"),
            rejected_bright_pixel_ratio_median=("rejected_bright_pixel_ratio", "median"),
        )
    )


def compute_failure_feature_tests(
    raw: pd.DataFrame,
    sensitivity: pd.DataFrame,
    features: list[str],
    methods: list[str],
    min_group_n: int,
) -> pd.DataFrame:
    d = raw[raw["gamma"] > 1.0].copy()
    status = (
        d.groupby(["sample_uid", "method"], as_index=False)
        .agg(ever_failed=("status", lambda x: bool(np.any(x != "ok"))), failed_task_count=("status", lambda x: int(np.sum(x != "ok"))))
    )
    base_cols = ["sample_uid", "method"] + [f for f in features if f in sensitivity.columns]
    merged = status.merge(sensitivity[base_cols].drop_duplicates(["sample_uid", "method"]), on=["sample_uid", "method"], how="left")
    rows: list[dict[str, Any]] = []
    for method in methods:
        dm = merged[merged["method"] == method]
        for feature in features:
            if feature not in dm.columns:
                continue
            a = pd.to_numeric(dm.loc[dm["ever_failed"], feature], errors="coerce").dropna().to_numpy()
            b = pd.to_numeric(dm.loc[~dm["ever_failed"], feature], errors="coerce").dropna().to_numpy()
            if len(a) >= min_group_n and len(b) >= min_group_n:
                u = stats.mannwhitneyu(a, b, alternative="two-sided")
                p = float(u.pvalue)
                stat = float(u.statistic)
                delta = cliffs_delta(a, b)
            else:
                p = stat = delta = np.nan
            rows.append({
                "dataset": str(raw["dataset"].dropna().iloc[0]),
                "method": method,
                "feature": feature,
                "n_ever_failed": len(a),
                "n_never_failed": len(b),
                "median_ever_failed": float(np.median(a)) if len(a) else np.nan,
                "median_never_failed": float(np.median(b)) if len(b) else np.nan,
                "mannwhitney_u": stat,
                "p_value": p,
                "cliffs_delta_failed_minus_success": delta,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_holm"] = np.nan
    for method, idx in out.groupby("method").groups.items():
        out.loc[idx, "p_holm"] = holm_adjust(out.loc[idx, "p_value"])
    return out


def select_detailed_pairs_and_features(
    dataset: str,
    methods: list[str],
    baseline_assoc: pd.DataFrame,
    within_assoc: pd.DataFrame,
    group_tests: pd.DataFrame,
    sensitivity: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, list[str]]:
    priorities = set((config.get("priority_pairs") or {}).get(dataset, []))
    significant_methods: set[str] = set()
    if config.get("auto_include_significant_pairs", True):
        significant_methods.update(baseline_assoc.loc[baseline_assoc["p_holm"] < 0.05, "method"].dropna().astype(str))
        positive_within = within_assoc[within_assoc.get("response_type", "") == "positive_sensitive"]
        significant_methods.update(positive_within.loc[positive_within["p_holm"] < 0.05, "method"].dropna().astype(str))
        significant_methods.update(group_tests.loc[group_tests["p_holm"] < 0.05, "method"].dropna().astype(str))
    detailed_methods = priorities | significant_methods
    # If no pair is selected, retain methods with enough positive and stable examples.
    if not detailed_methods:
        counts = sensitivity.groupby(["method", "response_type"]).size().unstack(fill_value=0)
        for method in methods:
            if counts.loc[method].get("positive_sensitive", 0) >= 3 and counts.loc[method].get("stable", 0) >= 3:
                detailed_methods.add(method)
    max_features = int(config.get("max_features_per_pair", 5))
    result: dict[str, list[str]] = {}
    for method in sorted(detailed_methods & set(methods)):
        candidates: list[str] = []
        b = baseline_assoc[baseline_assoc["method"] == method].copy()
        w = within_assoc[(within_assoc["method"] == method) & (within_assoc.get("response_type", "") == "positive_sensitive")].copy()
        for frame, pcol, effect_col, feature_col in [
            (b, "p_holm", "spearman_rho", "feature"),
            (w, "p_holm", "median_within_video_rho", "feature"),
        ]:
            sig = frame[frame[pcol] < 0.05].sort_values(effect_col, key=lambda s: s.abs(), ascending=False)
            candidates.extend(sig[feature_col].astype(str).tolist())
        if not candidates:
            candidates.extend(b.sort_values("spearman_rho", key=lambda s: s.abs(), ascending=False)["feature"].astype(str).head(max_features).tolist())
        # Prefer interpretable pixel-composition features if present.
        for f in ["valid_pixel_ratio_mean", "rejected_dark_pixel_ratio", "rejected_bright_pixel_ratio"]:
            if f in b["feature"].values or f in w["feature"].values:
                candidates.append(f)
        seen: set[str] = set()
        ordered = []
        for f in candidates:
            if f not in seen:
                seen.add(f)
                ordered.append(f)
        result[method] = ordered[:max_features]
    return result


def generate_report(
    output_path: Path,
    result_root: Path,
    baseline: pd.DataFrame,
    groups: pd.DataFrame,
    within: pd.DataFrame,
    failure: pd.DataFrame,
    selected: pd.DataFrame,
) -> None:
    def fmt(x: Any) -> str:
        try:
            if pd.isna(x):
                return "NA"
            return f"{float(x):.3f}"
        except Exception:
            return str(x)

    def fmt_p(x: Any) -> str:
        try:
            if pd.isna(x):
                return "NA"
            value = float(x)
            return f"{value:.2e}" if value < 0.001 else f"{value:.3f}"
        except Exception:
            return str(x)

    lines = [
        "# ROI 像素信息与 Gamma 正向响应机制分析",
        "",
        f"- 结果根目录：`{result_root}`",
        "- MAE benefit 定义：`MAE(gamma=1.0) - MAE(gamma)`，正值表示改善。",
        "- 基线分析以每个视频—方法为一个独立观测。",
        "- Gamma 机制分析先在每个视频内部计算 ROI 特征变化与 MAE benefit 的 Spearman 相关，再在视频间汇总，避免把多个 Gamma 点简单当作独立样本。",
        "- 本项目仅使用 v5 已保存的 ROI 汇总指标，不能从 CSV 恢复逐像素直方图。",
        "",
        "## 1. 基线 ROI 特征与最佳 MAE 改善",
        "",
    ]
    sig = baseline[baseline["p_holm"] < 0.05].sort_values(["dataset", "method", "p_holm", "spearman_rho"])
    if sig.empty:
        lines.append("没有基线特征在 Holm 校正后达到 0.05。应结合森林图、效应量和置信区间进行探索性解释。")
    else:
        lines.append("Holm 校正后显著的基线特征：")
        for _, r in sig.iterrows():
            lines.append(
                f"- {r['dataset']}–{r['method']}：`{r['feature']}`，n={int(r['n'])}，rho={fmt(r['spearman_rho'])}，"
                f"95% CI [{fmt(r['rho_ci95_low'])}, {fmt(r['rho_ci95_high'])}]，p_Holm={fmt_p(r['p_holm'])}。"
            )
    lines += ["", "## 2. 正向敏感与稳定型的 ROI 差异", ""]
    sigg = groups[groups["p_holm"] < 0.05].sort_values(["dataset", "method", "p_holm"])
    if sigg.empty:
        lines.append("没有正向敏感组与稳定组的单变量差异在 Holm 校正后达到 0.05。")
    else:
        for _, r in sigg.iterrows():
            lines.append(
                f"- {r['dataset']}–{r['method']}：`{r['feature']}`，正向组 n={int(r['n_positive'])}、中位数={fmt(r['median_positive'])}，"
                f"稳定组 n={int(r['n_reference'])}、中位数={fmt(r['median_reference'])}，Cliff's delta={fmt(r['cliffs_delta_positive_minus_reference'])}，"
                f"p_Holm={fmt_p(r['p_holm'])}。"
            )
    lines += ["", "## 3. 同一视频内的 Gamma–ROI–MAE 同步关系", ""]
    sigw = within[(within["response_type"] == "positive_sensitive") & (within["p_holm"] < 0.05)].sort_values(["dataset", "method", "p_holm"])
    if sigw.empty:
        lines.append("没有视频内相关的中位数在 Holm 校正后显著偏离 0。机制曲线仍可用于观察方向和阈值。")
    else:
        for _, r in sigw.iterrows():
            direction = "同向" if r["median_within_video_rho"] > 0 else "反向"
            lines.append(
                f"- {r['dataset']}–{r['method']} 正向敏感组：Gamma 引起的 `{r['feature']}` 变化与 MAE benefit 呈{direction}关系，"
                f"有效视频 n={int(r['n_videos_with_rho'])}，视频内 rho 中位数={fmt(r['median_within_video_rho'])}，"
                f"95% CI [{fmt(r['rho_ci95_low'])}, {fmt(r['rho_ci95_high'])}]，p_Holm={fmt_p(r['p_holm'])}。"
            )
    lines += ["", "## 4. 高 Gamma 方法失败", ""]
    sigf = failure[failure["p_holm"] < 0.05].sort_values(["dataset", "method", "p_holm"])
    if sigf.empty:
        lines.append("未发现基线 ROI 特征在 Holm 校正后显著区分“曾失败”和“从未失败”的视频—方法组合。请结合成功状态热图检查具体 Gamma 阈值。")
    else:
        for _, r in sigf.iterrows():
            lines.append(
                f"- {r['dataset']}–{r['method']}：`{r['feature']}` 区分曾失败 n={int(r['n_ever_failed'])} 与从未失败 n={int(r['n_never_failed'])} 的样本，"
                f"Cliff's delta={fmt(r['cliffs_delta_failed_minus_success'])}，p_Holm={fmt_p(r['p_holm'])}。"
            )
    lines += ["", "## 5. 详细绘图选择", ""]
    if selected.empty:
        lines.append("没有选择到详细绘图组合。")
    else:
        for (dataset, method), g in selected.groupby(["dataset", "method"]):
            lines.append(f"- {dataset}–{method}：" + "、".join(f"`{v}`" for v in g["feature"]))
    lines += [
        "",
        "## 6. 图表阅读顺序",
        "",
        "1. `01_correlation_forest`：判断哪些原始 ROI 特征更可能预测正向响应。",
        "2. `02_feature_scatter`：检查相关是否由离群点驱动，并观察响应类型分布。",
        "3. `03_positive_vs_stable`：直接比较正向敏感与稳定型的视频。",
        "4. `04_gamma_mechanism`：观察 MAE benefit 与 ROI 特征变化在 Gamma 轴上是否同步。",
        "5. `05_heatmaps`：定位具体视频及最佳/恶化 Gamma 区间。",
        "6. `06_pixel_composition`：观察有效、过暗和过亮像素比例如何随 Gamma 改变。",
        "7. `07_failure_analysis`：检查高 Gamma 是否造成算法可运行性下降。",
        "8. `08_cross_method_consensus`：寻找跨算法都受益的视频级 ROI 规律。",
        "",
        "## 7. 解释边界",
        "",
        "这些结果能够建立统计关联和时间顺序，但不能单独证明因果。要证明具体像素区间如何改变 rPPG 信号，仍需对少量典型视频重新提取逐帧 ROI 像素直方图、RGB 时间序列和 BVP 频谱。",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_analysis(config: dict[str, Any], cli_result_root: str | None = None, cli_output: str | None = None) -> Path:
    result_root = resolve_result_root(cli_result_root or config["result_root"])
    output_root = Path(cli_output).expanduser().resolve() if cli_output else result_root / str(config.get("output_subdir", "roi_gamma_mechanism_analysis"))
    dirs = ensure_output_tree(output_root)
    seed = int(config.get("random_seed", 20260701))
    rng = np.random.default_rng(seed)
    iterations = int(config.get("bootstrap_iterations", 2000))
    min_corr_n = int(config.get("minimum_correlation_n", 8))
    min_group_n = int(config.get("minimum_group_n", 3))
    min_within = int(config.get("minimum_within_video_points", 4))
    dpi = int(config.get("figure_dpi", 180))
    fmt = str(config.get("figure_format", "png"))
    core_features = list(config.get("core_features", []))
    mechanism_features = list(config.get("mechanism_features", []))
    selected_methods_cfg = list(config.get("methods", [])) or None
    dataset_dirs = discover_dataset_dirs(result_root, list(config.get("datasets", [])) or None)

    all_baseline: list[pd.DataFrame] = []
    all_groups: list[pd.DataFrame] = []
    all_within: list[pd.DataFrame] = []
    all_gamma_summary: list[pd.DataFrame] = []
    all_pixels: list[pd.DataFrame] = []
    all_failure: list[pd.DataFrame] = []
    selected_rows: list[dict[str, str]] = []

    for dataset_dir in dataset_dirs:
        dataset = dataset_dir.name
        LOG.info("Loading %s", dataset)
        tables = load_dataset_tables(dataset_dir)
        sensitivity = tables["sensitivity"].copy()
        methods = _method_list(sensitivity, selected_methods_cfg)
        if config.get("reuse_existing_baseline_statistics", True) and "correlation" in tables:
            baseline = tables["correlation"].copy()
            baseline = baseline[baseline["method"].isin(methods) & baseline["feature"].isin(core_features)].copy()
            # Normalize expected columns from the v5 output table.
            needed = ["dataset", "method", "feature", "n", "spearman_rho", "rho_ci95_low", "rho_ci95_high", "p_value", "p_holm"]
            baseline = baseline[[c for c in needed if c in baseline.columns]]
            for c in needed:
                if c not in baseline.columns:
                    baseline[c] = np.nan
            baseline = baseline[needed]
            LOG.info("%s: reused existing baseline correlation table", dataset)
        else:
            baseline = compute_baseline_associations(sensitivity, core_features, methods, min_corr_n, iterations, rng)
        groups = compute_group_comparisons(
            sensitivity, core_features, methods,
            str(config["response_groups"]["positive"]),
            str(config["response_groups"]["reference"]),
            min_group_n,
        )
        analysis_response_types = [
            str(config["response_groups"]["positive"]),
            str(config["response_groups"]["reference"]),
        ]
        within = compute_within_video_associations(
            tables["changes"], mechanism_features, methods, analysis_response_types,
            min_within, iterations, rng,
        )
        gamma_summary = compute_gamma_group_summary(
            tables["changes"], mechanism_features, methods,
            analysis_response_types, iterations, rng,
        )
        pixels = compute_pixel_composition_summary(
            tables["brightness"], sensitivity, methods,
            analysis_response_types,
        )
        failure = compute_failure_feature_tests(tables["raw"], sensitivity, core_features, methods, min_group_n)

        all_baseline.append(baseline)
        all_groups.append(groups)
        all_within.append(within)
        all_gamma_summary.append(gamma_summary)
        all_pixels.append(pixels.assign(dataset=dataset))
        all_failure.append(failure)

        detailed = select_detailed_pairs_and_features(dataset, methods, baseline, within, groups, sensitivity, config)
        for method, features in detailed.items():
            for feature in features:
                selected_rows.append({"dataset": dataset, "method": method, "feature": feature})

        if config.get("make_all_forest_plots", True):
            for method in methods:
                d = baseline[baseline["method"] == method]
                plot_forest(
                    d, f"{dataset} – {method}: baseline ROI associations",
                    dirs["forest"] / f"{slug(dataset)}_{slug(method)}_baseline_forest.{fmt}", dpi,
                )

        for method, features in detailed.items():
            pair_dir_scatter = dirs["scatter"] / slug(dataset) / slug(method)
            pair_dir_group = dirs["groups"] / slug(dataset) / slug(method)
            pair_dir_mech = dirs["mechanism"] / slug(dataset) / slug(method)
            pair_dir_heat = dirs["heatmaps"] / slug(dataset) / slug(method)
            for p in [pair_dir_scatter, pair_dir_group, pair_dir_mech, pair_dir_heat]:
                p.mkdir(parents=True, exist_ok=True)
            dm = sensitivity[sensitivity["method"] == method]
            change_m = tables["changes"][tables["changes"]["method"] == method].copy()
            change_m["MAE_benefit"] = -pd.to_numeric(change_m["MAE_delta_vs_baseline"], errors="coerce")
            for feature in features:
                br = baseline[(baseline["method"] == method) & (baseline["feature"] == feature)]
                gr = groups[(groups["method"] == method) & (groups["feature"] == feature)]
                annotation = ""
                if not br.empty:
                    r = br.iloc[0]
                    annotation = f"rho={r['spearman_rho']:.3f}; 95% CI [{r['rho_ci95_low']:.3f}, {r['rho_ci95_high']:.3f}]; p_Holm={r['p_holm']:.3g}"
                if config.get("make_scatter_plots", True) and feature in dm.columns:
                    plot_scatter(
                        dm, feature, f"{dataset} – {method}: {_safe_label(feature)} vs MAE benefit", annotation,
                        pair_dir_scatter / f"{slug(feature)}_vs_mae_benefit.{fmt}", dpi,
                    )
                if config.get("make_group_plots", True) and feature in dm.columns:
                    group_annotation = ""
                    if not gr.empty:
                        r = gr.iloc[0]
                        group_annotation = f"Cliff's delta={r['cliffs_delta_positive_minus_reference']:.3f}; p_Holm={r['p_holm']:.3g}"
                    plot_group_distribution(
                        dm, feature,
                        str(config["response_groups"]["positive"]),
                        str(config["response_groups"]["reference"]),
                        f"{dataset} – {method}: positive response vs stable", group_annotation,
                        pair_dir_group / f"{slug(feature)}_positive_vs_stable.{fmt}", dpi,
                    )
                if feature in mechanism_features and f"delta_{feature}" in change_m.columns:
                    if config.get("make_mechanism_curves", True):
                        plot_mechanism_curves(
                            gamma_summary[gamma_summary["method"] == method], feature,
                            f"{dataset} – {method}: Gamma mechanism for {_safe_label(feature)}",
                            pair_dir_mech / f"{slug(feature)}_gamma_mechanism.{fmt}", dpi,
                        )
                    if config.get("make_heatmaps", True):
                        plot_paired_heatmaps(
                            change_m, f"delta_{feature}",
                            f"{dataset} – {method}: sample-level MAE benefit and ROI change",
                            pair_dir_heat / f"{slug(feature)}_benefit_feature_heatmaps.{fmt}", dpi,
                        )

        if config.get("make_pixel_composition_plots", True):
            for method in detailed:
                plot_pixel_composition(
                    pixels[pixels["method"] == method],
                    f"{dataset} – {method}: usable / rejected ROI pixel composition",
                    dirs["pixels"] / f"{slug(dataset)}_{slug(method)}_pixel_composition.{fmt}", dpi,
                )

        if config.get("make_failure_plots", True):
            for method in methods:
                rm = tables["raw"][tables["raw"]["method"] == method]
                if rm.empty:
                    continue
                plot_failure_heatmap(
                    rm, f"{dataset} – {method}: task status by sample and Gamma",
                    dirs["failure"] / f"{slug(dataset)}_{slug(method)}_failure_heatmap.{fmt}", dpi,
                )
                plot_failure_rate(
                    rm, f"{dataset} – {method}: success rate by Gamma",
                    dirs["failure"] / f"{slug(dataset)}_{slug(method)}_success_rate.{fmt}", dpi,
                )

        if config.get("make_consensus_plots", True) and "consensus" in tables and "baseline" in tables:
            consensus = tables["consensus"]
            merged = consensus.merge(tables["baseline"], on=["dataset", "sample_uid"], how="left", suffixes=("", "_feature"))
            plot_consensus_distribution(
                consensus, f"{dataset}: cross-method response consensus",
                dirs["consensus"] / f"{slug(dataset)}_consensus_distribution.{fmt}", dpi,
            )
            consensus_features = []
            sig_dataset = baseline[baseline["p_holm"] < 0.05].sort_values("p_holm")
            consensus_features.extend(sig_dataset["feature"].astype(str).tolist())
            for f in ["temporal_luma_cv", "p95_frame_to_frame_luma_change", "spatial_luma_std", "valid_pixel_ratio_mean", "rejected_bright_pixel_ratio"]:
                consensus_features.append(f)
            seen = set()
            for feature in consensus_features:
                if feature in seen or feature not in merged.columns:
                    continue
                seen.add(feature)
                plot_consensus_feature(
                    merged, feature, f"{dataset}: cross-method consensus and {_safe_label(feature)}",
                    dirs["consensus"] / f"{slug(dataset)}_{slug(feature)}_consensus_groups.{fmt}", dpi,
                )
                if len(seen) >= 5:
                    break

    baseline_all = pd.concat(all_baseline, ignore_index=True) if all_baseline else pd.DataFrame()
    groups_all = pd.concat(all_groups, ignore_index=True) if all_groups else pd.DataFrame()
    within_all = pd.concat(all_within, ignore_index=True) if all_within else pd.DataFrame()
    gamma_all = pd.concat(all_gamma_summary, ignore_index=True) if all_gamma_summary else pd.DataFrame()
    pixels_all = pd.concat(all_pixels, ignore_index=True) if all_pixels else pd.DataFrame()
    failure_all = pd.concat(all_failure, ignore_index=True) if all_failure else pd.DataFrame()
    selected_all = pd.DataFrame(selected_rows)

    tables_to_save = {
        "baseline_feature_associations.csv": baseline_all,
        "positive_vs_stable_feature_tests.csv": groups_all,
        "within_video_gamma_mechanism_associations.csv": within_all,
        "gamma_response_group_summary.csv": gamma_all,
        "pixel_composition_group_summary.csv": pixels_all,
        "failure_feature_tests.csv": failure_all,
        "selected_detailed_features.csv": selected_all,
    }
    for name, df in tables_to_save.items():
        df.to_csv(dirs["tables"] / name, index=False)

    generate_report(
        dirs["reports"] / "ANALYSIS_SUMMARY.md", result_root,
        baseline_all, groups_all, within_all, failure_all, selected_all,
    )
    manifest = {
        "result_root": str(result_root),
        "output_root": str(output_root),
        "datasets": [p.name for p in dataset_dirs],
        "core_features": core_features,
        "mechanism_features": mechanism_features,
        "bootstrap_iterations": iterations,
        "reuse_existing_baseline_statistics": bool(config.get("reuse_existing_baseline_statistics", True)),
        "random_seed": seed,
        "note": "Uses existing v5 ROI summary tables; no raw-pixel histogram reconstruction.",
    }
    write_json(output_root / "run_manifest.json", manifest)
    dump_resolved_config(config, output_root / "resolved_config.yaml")
    LOG.info("Analysis complete: %s", output_root)
    return output_root


def _safe_label(feature: str) -> str:
    return feature.replace("_", " ")
