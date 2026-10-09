from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from gamma_rppg.common.config import load_config

REQUIRED_FILES = {
    "sensitivity": "pre_roi_sensitivity_with_features.csv",
    "correlation": "pre_roi_feature_improvement_correlation.csv",
    "group_tests": "pre_roi_feature_group_tests.csv",
    "response": "pre_roi_response_curves.csv",
    "changes": "pre_roi_roi_feature_changes.csv",
    "change_corr": "pre_roi_feature_change_vs_mae_delta.csv",
    "consensus": "pre_roi_cross_method_consensus.csv",
    "baseline": "pre_roi_baseline_roi_features.csv",
    "brightness": "brightness_metrics.csv",
    "raw": "raw_results.csv",
    "quality": "quality_control.csv",
}




def resolve_result_root(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    candidates = [
        p,
        p / "GammaBrightness_rPPG_v5",
    ]
    for c in candidates:
        if (c / "pre_roi_gamma").is_dir():
            return c
    raise FileNotFoundError(
        f"Cannot find pre_roi_gamma under {p}. Pass the GammaBrightness_rPPG_v5 result root."
    )


def discover_dataset_dirs(result_root: Path, selected: list[str] | None = None) -> list[Path]:
    base = result_root / "pre_roi_gamma"
    dirs = sorted([p for p in base.iterdir() if p.is_dir() and (p / REQUIRED_FILES["sensitivity"]).is_file()])
    if selected:
        wanted = set(selected)
        dirs = [p for p in dirs if p.name in wanted]
    if not dirs:
        raise FileNotFoundError(f"No dataset result folders found under {base}")
    return dirs


def load_dataset_tables(dataset_dir: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for key, name in REQUIRED_FILES.items():
        path = dataset_dir / name
        if path.is_file():
            tables[key] = pd.read_csv(path)
    must = {"sensitivity", "response", "changes", "brightness", "raw"}
    missing = sorted(must - set(tables))
    if missing:
        raise FileNotFoundError(f"{dataset_dir.name}: missing required tables: {missing}")
    return tables


def ensure_output_tree(output_root: Path) -> dict[str, Path]:
    dirs = {
        "root": output_root,
        "logs": output_root / "logs",
        "tables": output_root / "tables",
        "reports": output_root / "reports",
        "forest": output_root / "figures" / "01_correlation_forest",
        "scatter": output_root / "figures" / "02_feature_scatter",
        "groups": output_root / "figures" / "03_positive_vs_stable",
        "mechanism": output_root / "figures" / "04_gamma_mechanism",
        "heatmaps": output_root / "figures" / "05_heatmaps",
        "pixels": output_root / "figures" / "06_pixel_composition",
        "failure": output_root / "figures" / "07_failure_analysis",
        "consensus": output_root / "figures" / "08_cross_method_consensus",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Compatibility name for the shared YAML/JSON loader."""
    return load_config(path, allow_empty=True)
