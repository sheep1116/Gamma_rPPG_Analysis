from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from gamma_rppg.common.config import dump_resolved_config, ensure_dir, load_config, resolve_mode_root, resolve_project_root
from gamma_rppg.signal.image_math import encoded_luma, make_gamma_table


@dataclass
class CaseSpec:
    dataset: str
    sample_uid: str
    method: str
    category: str
    input_path: str
    fps: float
    baseline_gamma: float
    target_gamma: float
    base_mae: float = np.nan
    target_mae: float = np.nan
    best_gamma: float = np.nan
    best_mae: float = np.nan
    response_type: str = ""
    status_at_target: str = ""
    error_at_target: str = ""
    reason: str = ""


def safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("_") or "sample"


def finite_float(value, default: float = np.nan) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    return result if np.isfinite(result) else default


def configure_plot_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 15,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    array = np.asarray(signal, dtype=float)
    array = np.squeeze(array)
    if array.ndim == 0:
        return array.reshape(1, 1)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    return array


def rgb_timeseries_view(signal: np.ndarray) -> np.ndarray:
    """Return a frame-by-channel view for plotting without changing pyVHR input semantics."""
    array = np.asarray(signal, dtype=float)
    array = np.squeeze(array)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim == 2:
        return array
    if array.ndim >= 3 and array.shape[-1] == 3:
        return array.reshape(array.shape[0], -1, 3).mean(axis=1)
    return array.reshape(array.shape[0], -1)


def select_positive_cases(
    labels: pd.DataFrame,
    count: int,
    baseline_gamma: float,
) -> List[CaseSpec]:
    if labels.empty or count <= 0:
        return []
    frame = labels[labels["response_type"] == "positive_sensitive"].copy()
    if frame.empty:
        return []
    frame["absolute_mae_improvement"] = pd.to_numeric(frame.get("absolute_mae_improvement"), errors="coerce")
    frame = frame.sort_values("absolute_mae_improvement", ascending=False).head(count)
    cases = []
    for _, row in frame.iterrows():
        cases.append(CaseSpec(
            dataset=str(row.get("dataset", "")),
            sample_uid=str(row.get("sample_uid", "")),
            method=str(row.get("method", "")),
            category="positive",
            input_path=str(row.get("input_path", "")),
            fps=finite_float(row.get("fps"), 30.0),
            baseline_gamma=baseline_gamma,
            target_gamma=finite_float(row.get("best_gamma"), baseline_gamma),
            base_mae=finite_float(row.get("base_MAE")),
            target_mae=finite_float(row.get("best_MAE")),
            best_gamma=finite_float(row.get("best_gamma")),
            best_mae=finite_float(row.get("best_MAE")),
            response_type=str(row.get("response_type", "")),
            reason="largest MAE improvement among positive-sensitive cases",
        ))
    return cases


def select_stable_cases(
    labels: pd.DataFrame,
    raw: pd.DataFrame,
    count: int,
    baseline_gamma: float,
) -> List[CaseSpec]:
    if labels.empty or raw.empty or count <= 0:
        return []
    frame = labels[labels["response_type"] == "stable"].copy()
    if frame.empty:
        return []
    frame["abs_relative_improvement"] = pd.to_numeric(
        frame.get("relative_mae_improvement"), errors="coerce"
    ).abs()
    frame = frame.sort_values("abs_relative_improvement").head(count)
    raw_ok = raw[raw["status"] == "ok"].copy()
    raw_ok["gamma"] = pd.to_numeric(raw_ok["gamma"], errors="coerce")
    raw_ok["MAE"] = pd.to_numeric(raw_ok["MAE"], errors="coerce")
    max_gamma = finite_float(raw_ok["gamma"].max(), baseline_gamma)
    cases = []
    for _, row in frame.iterrows():
        subset = raw_ok[
            (raw_ok["sample_uid"] == row.get("sample_uid"))
            & (raw_ok["method"] == row.get("method"))
            & np.isclose(raw_ok["gamma"], max_gamma)
        ]
        target_mae = finite_float(subset.iloc[0]["MAE"]) if not subset.empty else np.nan
        cases.append(CaseSpec(
            dataset=str(row.get("dataset", "")),
            sample_uid=str(row.get("sample_uid", "")),
            method=str(row.get("method", "")),
            category="stable",
            input_path=str(row.get("input_path", "")),
            fps=finite_float(row.get("fps"), 30.0),
            baseline_gamma=baseline_gamma,
            target_gamma=max_gamma,
            base_mae=finite_float(row.get("base_MAE")),
            target_mae=target_mae,
            best_gamma=finite_float(row.get("best_gamma")),
            best_mae=finite_float(row.get("best_MAE")),
            response_type=str(row.get("response_type", "")),
            status_at_target="ok" if not subset.empty else "",
            reason="smallest absolute response among stable cases",
        ))
    return cases


def select_failure_cases(
    raw: pd.DataFrame,
    count: int,
    baseline_gamma: float,
) -> List[CaseSpec]:
    if raw.empty or count <= 0:
        return []
    frame = raw.copy()
    frame["gamma"] = pd.to_numeric(frame.get("gamma"), errors="coerce")
    frame["MAE"] = pd.to_numeric(frame.get("MAE"), errors="coerce")
    failures = frame[(frame["status"] != "ok") & (~np.isclose(frame["gamma"], baseline_gamma))].copy()
    if failures.empty:
        return []
    failures["gamma_rank"] = failures["gamma"].rank(method="dense", ascending=False)
    failures = failures.sort_values(["gamma", "method", "sample_uid"], ascending=[False, True, True])
    cases = []
    seen = set()
    for _, row in failures.iterrows():
        key = (row.get("sample_uid"), row.get("method"))
        if key in seen:
            continue
        seen.add(key)
        base = frame[
            (frame["sample_uid"] == row.get("sample_uid"))
            & (frame["method"] == row.get("method"))
            & np.isclose(frame["gamma"], baseline_gamma)
        ]
        cases.append(CaseSpec(
            dataset=str(row.get("dataset", "")),
            sample_uid=str(row.get("sample_uid", "")),
            method=str(row.get("method", "")),
            category="failure",
            input_path=str(row.get("input_path", "")),
            fps=finite_float(row.get("fps"), 30.0),
            baseline_gamma=baseline_gamma,
            target_gamma=finite_float(row.get("gamma"), baseline_gamma),
            base_mae=finite_float(base.iloc[0]["MAE"]) if not base.empty else np.nan,
            target_mae=np.nan,
            status_at_target=str(row.get("status", "")),
            error_at_target=str(row.get("error_message", "")),
            reason="non-ok rPPG task at a non-baseline Gamma",
        ))
        if len(cases) >= count:
            break
    return cases


def select_cases(dataset_dir: Path, config: Dict) -> List[CaseSpec]:
    raw = read_csv_if_exists(dataset_dir / "raw_results.csv")
    labels = read_csv_if_exists(dataset_dir / "pre_roi_sensitivity_labels.csv")
    if raw.empty:
        return []
    baseline_gamma = float(config.get("analysis", {}).get("baseline_gamma", 1.0))
    case_cfg = config.get("case_mechanism", {})
    per_category = case_cfg.get("cases_per_dataset", {})
    raw["status"] = raw.get("status", "").astype(str)
    return [
        *select_positive_cases(labels, int(per_category.get("positive", 2)), baseline_gamma),
        *select_stable_cases(labels, raw, int(per_category.get("stable", 2)), baseline_gamma),
        *select_failure_cases(raw, int(per_category.get("failure", 2)), baseline_gamma),
    ]


def select_explicit_cases(dataset_dir: Path, config: Dict) -> List[CaseSpec]:
    """Select only the manuscript cases specified in the configuration."""
    dataset_name = dataset_dir.name
    case_cfg = config.get("case_mechanism", {})
    definitions = [
        item for item in case_cfg.get("explicit_cases", [])
        if str(item.get("dataset", "")) == dataset_name
    ]
    if not definitions:
        return []

    raw = read_csv_if_exists(dataset_dir / "raw_results.csv")
    labels = read_csv_if_exists(dataset_dir / "pre_roi_sensitivity_labels.csv")
    if raw.empty:
        raise FileNotFoundError(f"No raw results found for configured cases: {dataset_dir}")

    raw = raw.copy()
    raw["gamma"] = pd.to_numeric(raw.get("gamma"), errors="coerce")
    raw["MAE"] = pd.to_numeric(raw.get("MAE"), errors="coerce")
    raw["method"] = raw.get("method", "").astype(str)
    raw["sample_uid"] = raw.get("sample_uid", "").astype(str)
    baseline_gamma = float(config.get("analysis", {}).get("baseline_gamma", 1.0))
    cases: List[CaseSpec] = []

    for definition in definitions:
        sample_uid = str(definition["sample_uid"])
        method = str(definition["method"])
        target_gamma = float(definition["target_gamma"])
        subset = raw[(raw["sample_uid"] == sample_uid) & (raw["method"] == method)]
        baseline = subset[np.isclose(subset["gamma"], baseline_gamma)]
        target = subset[np.isclose(subset["gamma"], target_gamma)]
        if baseline.empty:
            raise ValueError(
                f"Configured case has no baseline result: {dataset_name}/{sample_uid}/{method}"
            )
        if target.empty:
            raise ValueError(
                f"Configured case has no target-Gamma result: "
                f"{dataset_name}/{sample_uid}/{method}/Gamma={target_gamma:g}"
            )

        base_row = baseline.iloc[0]
        target_row = target.iloc[0]
        best_gamma = np.nan
        best_mae = np.nan
        if not labels.empty:
            label_match = labels[
                (labels.get("sample_uid", "").astype(str) == sample_uid)
                & (labels.get("method", "").astype(str) == method)
            ]
            if not label_match.empty:
                best_gamma = finite_float(label_match.iloc[0].get("best_gamma"))
                best_mae = finite_float(label_match.iloc[0].get("best_MAE"))

        cases.append(CaseSpec(
            dataset=dataset_name,
            sample_uid=sample_uid,
            method=method,
            category=str(definition["category"]),
            input_path=str(base_row.get("input_path", "")),
            fps=finite_float(base_row.get("fps"), 30.0),
            baseline_gamma=baseline_gamma,
            target_gamma=target_gamma,
            base_mae=finite_float(base_row.get("MAE")),
            target_mae=finite_float(target_row.get("MAE")),
            best_gamma=best_gamma,
            best_mae=best_mae,
            response_type=str(definition.get("response_type", "")),
            status_at_target=str(target_row.get("status", "")),
            error_at_target=str(target_row.get("error_message", "")),
            reason=str(definition.get("reason", "configured representative manuscript case")),
        ))
    return cases


def landmarks_from_result(result, width: int, height: int):
    import mediapipe as mp

    landmarks = np.zeros((468, 5), dtype=np.float32)
    landmarks[:, :2] = -1.0
    if result is None or not result.multi_face_landmarks:
        return landmarks, False
    drawing = mp.solutions.drawing_utils
    for idx, landmark in enumerate(result.multi_face_landmarks[0].landmark):
        invisible = (
            landmark.HasField("visibility") and landmark.visibility < 0.5
        ) or (
            landmark.HasField("presence") and landmark.presence < 0.5
        )
        if invisible:
            continue
        coords = drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y, width, height)
        if coords:
            landmarks[idx, 0] = coords[1]
            landmarks[idx, 1] = coords[0]
    return landmarks, True


def geometry_pixels_from_frame(image_rgb: np.ndarray, skin_extractor, landmarks: np.ndarray) -> np.ndarray:
    cropped, _ = skin_extractor.extract_skin(image_rgb, landmarks)
    white = np.full_like(image_rgb, 255, dtype=np.uint8)
    cropped_mask, _ = skin_extractor.extract_skin(white, landmarks)
    if cropped is None or cropped.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if cropped_mask is not None and cropped_mask.shape[:2] == cropped.shape[:2]:
        mask = np.any(cropped_mask > 0, axis=2)
    else:
        mask = np.any(cropped > 0, axis=2)
    if not np.any(mask):
        return np.empty((0, 3), dtype=np.float32)
    return cropped[mask].reshape(-1, 3).astype(np.float32)


def collect_roi_luma(
    input_path: str,
    gammas: Sequence[float],
    config: Dict,
) -> Dict[float, np.ndarray]:
    import cv2
    import mediapipe as mp
    import pyVHR as vhr
    from gamma_rppg.signal.processing import iter_frames_inline

    case_cfg = config.get("case_mechanism", {})
    max_frames = int(case_cfg.get("max_histogram_frames", 180))
    frame_stride = max(1, int(case_cfg.get("frame_stride", 3)))
    max_pixels = int(case_cfg.get("max_luma_pixels_per_gamma", 250000))
    skin_device = config.get("experiment", {}).get("skin_extractor_device", "GPU")
    tables = {float(gamma): make_gamma_table(float(gamma)) for gamma in gammas}
    skin_extractor = vhr.extraction.SkinExtractionConvexHull(skin_device)
    values = {float(gamma): [] for gamma in gammas}
    sampled_frames = 0

    with mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        for frame_index, frame in enumerate(iter_frames_inline(input_path)):
            if frame_index % frame_stride != 0:
                continue
            if sampled_frames >= max_frames:
                break
            original_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for gamma in gammas:
                transformed = cv2.LUT(original_rgb, tables[float(gamma)])
                result = face_mesh.process(transformed)
                landmarks, detected = landmarks_from_result(
                    result, transformed.shape[1], transformed.shape[0]
                )
                if not detected:
                    continue
                pixels = geometry_pixels_from_frame(transformed, skin_extractor, landmarks)
                if pixels.size:
                    luma = encoded_luma(pixels)
                    if luma.size > 4000:
                        idx = np.linspace(0, luma.size - 1, 4000).astype(int)
                        luma = luma[idx]
                    values[float(gamma)].append(luma)
            sampled_frames += 1

    result = {}
    for gamma, chunks in values.items():
        if chunks:
            joined = np.concatenate(chunks)
            if joined.size > max_pixels:
                idx = np.linspace(0, joined.size - 1, max_pixels).astype(int)
                joined = joined[idx]
            result[gamma] = joined
        else:
            result[gamma] = np.empty(0, dtype=float)
    return result


def plot_luma_histogram(case: CaseSpec, folder: Path, config: Dict) -> None:
    import matplotlib.pyplot as plt

    luma_by_gamma = collect_roi_luma(
        case.input_path, [case.baseline_gamma, case.target_gamma], config
    )
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    bins = np.linspace(0, 255, 64)
    for gamma, values in luma_by_gamma.items():
        if values.size == 0:
            continue
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.6,
            label=f"Gamma={gamma:g}",
        )
    ax.set_xlabel("Encoded ROI luma")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 255)
    ax.set_title("ROI luma distribution")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(folder / "roi_luma_histogram.png")
    plt.close(fig)
    summary_rows = []
    for gamma, values in luma_by_gamma.items():
        row = {"gamma": gamma, "pixel_count": int(values.size)}
        if values.size:
            row.update({
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "p05": float(np.percentile(values, 5)),
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
            })
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(folder / "roi_luma_histogram_stats.csv", index=False)


def extract_signal_for_case(case: CaseSpec, gamma: float, config: Dict) -> np.ndarray:
    from gamma_rppg.experiments.pre_roi import extract_signal_for_gamma

    diagnostic_config = dict(config)
    diagnostic_config["experiment"] = dict(config.get("experiment", {}))
    diagnostic_config["experiment"]["collect_brightness"] = False
    signal, _, _ = extract_signal_for_gamma(case.input_path, case.fps, gamma, diagnostic_config)
    return np.asarray(signal, dtype=float)


def plot_first_frame_comparison(case: CaseSpec, folder: Path) -> None:
    """Save the first frame before and after the case-specific Gamma mapping."""
    import cv2
    import matplotlib.pyplot as plt
    from gamma_rppg.signal.processing import iter_frames_inline

    first_frame = next(iter(iter_frames_inline(case.input_path)), None)
    if first_frame is None:
        raise ValueError(f"Could not read a frame from {case.input_path}")
    original_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    gammas = [case.baseline_gamma, case.target_gamma]
    fig, axes = plt.subplots(1, len(gammas), figsize=(9.6, 4.6))
    if len(gammas) == 1:
        axes = [axes]
    for axis, gamma in zip(axes, gammas):
        transformed = cv2.LUT(original_rgb, make_gamma_table(float(gamma)))
        axis.imshow(transformed)
        axis.set_title(f"Gamma={gamma:g}", pad=10)
        axis.axis("off")
    fig.suptitle("First video frame", fontsize=24, y=0.97)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(folder / "first_frame_comparison.png")
    plt.close(fig)


def plot_rgb_timeseries(
    case: CaseSpec,
    folder: Path,
    signals: Dict[float, np.ndarray],
) -> None:
    import matplotlib.pyplot as plt

    views = {gamma: rgb_timeseries_view(signal) for gamma, signal in signals.items()}
    max_channels = max(signal.shape[1] for signal in views.values() if signal.size)
    channels = min(3, max_channels)
    names = ["R", "G", "B"] if channels >= 3 else ["Signal"]
    fig, axes = plt.subplots(channels, 1, figsize=(8.8, 3.0 * channels), sharex=True)
    if channels == 1:
        axes = [axes]
    for channel_index, ax in enumerate(axes):
        for gamma, signal in views.items():
            if signal.size == 0:
                continue
            y = signal[:, min(channel_index, signal.shape[1] - 1)]
            t = np.arange(len(y)) / case.fps
            ax.plot(t, y, linewidth=1.25, label=f"Gamma={gamma:g}")
        ax.set_ylabel(names[channel_index])
        ax.legend(frameon=False)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        "ROI RGB mean time series" if channels >= 3 else "ROI signal time series",
        fontsize=24,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(folder / "rgb_timeseries.png")
    plt.close(fig)


def as_numpy(array) -> np.ndarray:
    if hasattr(array, "get"):
        array = array.get()
    return np.asarray(array, dtype=float)


def iter_bvp_vectors(bvps) -> Iterable[np.ndarray]:
    array = as_numpy(bvps)
    array = np.squeeze(array)
    if array.ndim == 1:
        yield array
    elif array.ndim == 2:
        for row in array:
            yield np.asarray(row, dtype=float).reshape(-1)
    else:
        flat = array.reshape(-1, array.shape[-1])
        for row in flat:
            yield np.asarray(row, dtype=float).reshape(-1)


def spectrum_from_signal(
    signal: np.ndarray,
    fps: float,
    method_name: str,
    config: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    import pyVHR as vhr
    from pyVHR.BVP import BPfilter, RGB_sig_to_BVP
    from gamma_rppg.experiments.methods import resolve_method

    signal = np.asarray(signal, dtype=float)
    rgb_view = rgb_timeseries_view(signal)
    if rgb_view.shape[1] < 3:
        raise ValueError(f"Expected at least 3 RGB channels, got shape {signal.shape}")
    wsize = float(config.get("experiment", {}).get("wsize", 8))
    step = float(config.get("experiment", {}).get("step", 1))
    prefer_cuda = bool(config.get("experiment", {}).get("prefer_cuda", True))
    filter_params = dict(config.get("experiment", {}).get("bp_filter", {
        "order": 6, "minHz": 0.65, "maxHz": 4.0,
    }))
    spec = resolve_method(vhr.BVP, method_name, prefer_cuda=prefer_cuda)
    windowed, _ = vhr.extraction.sig_windowing(signal, wsize, step, fps)
    bvps = RGB_sig_to_BVP(
        windowed,
        fps,
        device_type=spec.device_type,
        method=getattr(vhr.BVP, spec.function_name),
        params=dict(spec.params),
    )
    filter_params["fps"] = fps
    bvps = vhr.BVP.apply_filter(bvps, BPfilter, params=filter_params)
    spectra = []
    frequencies = None
    for vector in iter_bvp_vectors(bvps):
        if vector.size < 8 or not np.all(np.isfinite(vector)):
            continue
        vector = vector - np.mean(vector)
        freq = np.fft.rfftfreq(vector.size, d=1.0 / fps)
        power = np.abs(np.fft.rfft(vector)) ** 2
        mask = (freq >= 0.5) & (freq <= 4.0)
        if np.any(mask):
            frequencies = freq[mask]
            spectra.append(power[mask])
    if not spectra or frequencies is None:
        raise ValueError("No valid BVP windows were available for spectral analysis")
    min_len = min(len(item) for item in spectra)
    mean_power = np.mean(np.vstack([item[:min_len] for item in spectra]), axis=0)
    if np.max(mean_power) > 0:
        mean_power = mean_power / np.max(mean_power)
    return frequencies[:min_len], mean_power


def spectrum_from_waveform(
    waveform: np.ndarray,
    sampling_rate: float,
    config: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a GT spectrum with the same sliding-window duration as rPPG spectra."""
    waveform = np.asarray(waveform, dtype=float).squeeze()
    if waveform.ndim != 1:
        waveform = waveform.reshape(-1)
    waveform = waveform[np.isfinite(waveform)]
    if waveform.size < 8 or sampling_rate <= 0:
        raise ValueError("Ground-truth waveform is unavailable or too short")

    experiment = config.get("experiment", {})
    window_seconds = float(experiment.get("wsize", 8))
    step_seconds = float(experiment.get("step", 1))
    window_size = max(8, int(round(window_seconds * sampling_rate)))
    step_size = max(1, int(round(step_seconds * sampling_rate)))
    if waveform.size < window_size:
        raise ValueError("Ground-truth waveform is shorter than the spectral window")

    spectra = []
    frequencies = None
    for start in range(0, waveform.size - window_size + 1, step_size):
        vector = waveform[start:start + window_size]
        vector = vector - np.mean(vector)
        freq = np.fft.rfftfreq(vector.size, d=1.0 / sampling_rate)
        power = np.abs(np.fft.rfft(vector)) ** 2
        mask = (freq >= 0.5) & (freq <= 4.0)
        if np.any(mask):
            frequencies = freq[mask]
            spectra.append(power[mask])
    if not spectra or frequencies is None:
        raise ValueError("No valid ground-truth spectral windows were available")
    mean_power = np.mean(np.vstack(spectra), axis=0)
    if np.max(mean_power) > 0:
        mean_power = mean_power / np.max(mean_power)
    return frequencies, mean_power


class GroundTruthProvider:
    """Resolve a selected case to the same dataset reference signal used in evaluation."""

    def __init__(self, config: Dict):
        self.config = config
        self._datasets: Dict[str, object] = {}
        self._spectrum_cache: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}

    @staticmethod
    def _waveform_from_signal(signal_gt) -> Tuple[np.ndarray, float]:
        waveform = None
        for attribute in ("data", "signal", "bvp", "values"):
            value = getattr(signal_gt, attribute, None)
            if value is not None:
                waveform = value
                break
        if waveform is None:
            raise AttributeError("The dataset reference signal has no readable waveform attribute")
        sampling_rate = finite_float(
            getattr(signal_gt, "fs", getattr(signal_gt, "sample_rate", np.nan))
        )
        if not np.isfinite(sampling_rate) or sampling_rate <= 0:
            raise ValueError("The dataset reference signal has no valid sampling rate")
        return np.asarray(waveform, dtype=float), sampling_rate

    def _dataset(self, dataset_name: str):
        if dataset_name not in self._datasets:
            from gamma_rppg.experiments.pre_roi import build_dataset

            dataset_cfg = self.config.get("datasets", {}).get(dataset_name)
            if not dataset_cfg:
                raise KeyError(f"No dataset configuration for {dataset_name}")
            self._datasets[dataset_name] = build_dataset(dataset_cfg)
        return self._datasets[dataset_name]

    def _find_video_index(self, case: CaseSpec) -> int:
        from gamma_rppg.experiments.pre_roi import derive_sample_metadata

        dataset_cfg = self.config.get("datasets", {}).get(case.dataset, {})
        dataset = self._dataset(case.dataset)
        video_files = getattr(dataset, "videoFilenames", [])
        for video_index in range(len(video_files)):
            video_file = dataset.getVideoFilename(video_index)
            native = dataset.getSampleMetadata(video_index) if hasattr(dataset, "getSampleMetadata") else {}
            metadata = derive_sample_metadata(case.dataset, dataset_cfg, video_file, native)
            if str(metadata.get("sample_uid", "")) == case.sample_uid:
                return video_index
        raise LookupError(f"Selected case was not found in the dataset loader: {case.sample_uid}")

    def spectrum_for_case(self, case: CaseSpec) -> Tuple[np.ndarray, np.ndarray]:
        cache_key = (case.dataset, case.sample_uid)
        if cache_key not in self._spectrum_cache:
            dataset = self._dataset(case.dataset)
            video_index = self._find_video_index(case)
            signal_gt = dataset.readSigfile(dataset.getSigFilename(video_index))
            waveform, sampling_rate = self._waveform_from_signal(signal_gt)
            self._spectrum_cache[cache_key] = spectrum_from_waveform(
                waveform, sampling_rate, self.config
            )
        return self._spectrum_cache[cache_key]


def plot_bvp_spectrum(
    case: CaseSpec,
    folder: Path,
    signals: Dict[float, np.ndarray],
    config: Dict,
    ground_truth: GroundTruthProvider,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    status_rows = []
    for gamma, signal in signals.items():
        try:
            freq, power = spectrum_from_signal(signal, case.fps, case.method, config)
            ax.plot(freq, power, linewidth=2.4, label=f"Gamma={gamma:g}")
            status_rows.append({"gamma": gamma, "status": "ok", "message": ""})
        except Exception as exc:
            status_rows.append({"gamma": gamma, "status": "error", "message": repr(exc)})
    try:
        gt_frequency, gt_power = ground_truth.spectrum_for_case(case)
        ax.plot(
            gt_frequency,
            gt_power,
            color="#54A24B",
            linestyle="--",
            linewidth=2.6,
            label="Ground-truth BVP",
        )
        status_rows.append({"gamma": "GT", "status": "ok", "message": ""})
    except Exception as exc:
        status_rows.append({"gamma": "GT", "status": "error", "message": repr(exc)})
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized BVP power")
    ax.set_title("BVP spectrum")
    ax.set_xlim(0.5, 4.0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(folder / "bvp_spectrum.png")
    plt.close(fig)
    pd.DataFrame(status_rows).to_csv(folder / "bvp_spectrum_status.csv", index=False)


def render_case(
    case: CaseSpec,
    output_root: Path,
    config: Dict,
    ground_truth: GroundTruthProvider,
) -> None:
    folder = ensure_dir(
        output_root / case.dataset / case.category / f"{safe_name(case.sample_uid)}__{safe_name(case.method)}"
    )
    (folder / "case_metadata.json").write_text(
        json.dumps(asdict(case), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    pd.DataFrame([asdict(case)]).to_csv(folder / "case_metadata.csv", index=False, encoding="utf-8-sig")
    gammas = [case.baseline_gamma]
    if not np.isclose(case.target_gamma, case.baseline_gamma):
        gammas.append(case.target_gamma)
    signals = {}
    signal_status = []
    for gamma in gammas:
        try:
            signals[gamma] = extract_signal_for_case(case, gamma, config)
            signal_status.append({
                "gamma": gamma, "status": "ok", "shape": str(np.asarray(signals[gamma]).shape), "message": "",
            })
        except Exception as exc:
            signals[gamma] = np.empty((0, 1), dtype=float)
            signal_status.append({
                "gamma": gamma, "status": "error", "shape": "", "message": repr(exc),
            })
    pd.DataFrame(signal_status).to_csv(folder / "signal_extraction_status.csv", index=False)
    try:
        plot_first_frame_comparison(case, folder)
    except Exception as exc:
        (folder / "first_frame_comparison_error.txt").write_text(repr(exc), encoding="utf-8")
    try:
        plot_luma_histogram(case, folder, config)
    except Exception as exc:
        (folder / "roi_luma_histogram_error.txt").write_text(repr(exc), encoding="utf-8")
    if any(signal.size for signal in signals.values()):
        try:
            plot_rgb_timeseries(case, folder, signals)
        except Exception as exc:
            (folder / "rgb_timeseries_error.txt").write_text(repr(exc), encoding="utf-8")
        try:
            plot_bvp_spectrum(case, folder, signals, config, ground_truth)
        except Exception as exc:
            (folder / "bvp_spectrum_error.txt").write_text(repr(exc), encoding="utf-8")


def write_index(cases: Sequence[CaseSpec], output_root: Path, config: Dict) -> None:
    rows = [asdict(case) for case in cases]
    pd.DataFrame(rows).to_csv(output_root / "case_index.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# Case mechanism figures",
        "",
        "This folder contains paper-ready diagnostic figures with English labels.",
        "Each selected case includes a Gamma first-frame comparison and BVP spectra overlaid with the dataset ground-truth BVP spectrum.",
        "",
        "| Dataset | Category | Sample | Method | Baseline Gamma | Target Gamma | Reason |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for case in cases:
        lines.append(
            f"| {case.dataset} | {case.category} | `{case.sample_uid}` | {case.method} | "
            f"{case.baseline_gamma:g} | {case.target_gamma:g} | {case.reason} |"
        )
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    dump_resolved_config(config, output_root / "resolved_config.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper-ready mechanism figures for positive, stable, and failure Gamma-rPPG cases."
    )
    parser.add_argument("--config", default="configs/case_mechanism_figures.yaml")
    parser.add_argument("--pre-roi-root", default=None, help="Path to an existing pre_roi_gamma result folder.")
    parser.add_argument("--output-root", default=None, help="Folder for generated case mechanism figures.")
    parser.add_argument("--datasets", default="all", help="Comma-separated dataset names, or all.")
    args = parser.parse_args()

    configure_plot_style()
    config = load_config(args.config)
    source_config = config.get("source_pre_roi_config")
    if source_config:
        merged = load_config(source_config)
        for key, value in config.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        config = merged

    pre_root = Path(args.pre_roi_root) if args.pre_roi_root else resolve_mode_root(config, "pre_roi_gamma")
    if args.output_root:
        output_root = ensure_dir(args.output_root)
    else:
        project_root = resolve_project_root(config)
        output_root = ensure_dir(project_root / config.get("output", {}).get("folders", {}).get(
            "case_mechanism_figures", "case_mechanism_figures"
        ))

    selected = None
    if args.datasets and args.datasets.lower() != "all":
        selected = {item.strip() for item in args.datasets.split(",") if item.strip()}

    case_cfg = config.get("case_mechanism", {})
    use_explicit_cases = bool(case_cfg.get("explicit_cases"))
    all_cases: List[CaseSpec] = []
    ground_truth = GroundTruthProvider(config)
    for dataset_dir in sorted(pre_root.iterdir()) if pre_root.exists() else []:
        if not dataset_dir.is_dir() or not (dataset_dir / "raw_results.csv").exists():
            continue
        if selected is not None and dataset_dir.name not in selected:
            continue
        cases = (
            select_explicit_cases(dataset_dir, config)
            if use_explicit_cases
            else select_cases(dataset_dir, config)
        )
        print(f"[INFO] {dataset_dir.name}: selected {len(cases)} cases")
        for case in cases:
            render_case(case, output_root, config, ground_truth)
        all_cases.extend(cases)
    if use_explicit_cases:
        configured = case_cfg.get("explicit_cases", [])
        selected = {(case.dataset, case.sample_uid, case.method) for case in all_cases}
        missing = [
            item for item in configured
            if (str(item.get("dataset", "")), str(item.get("sample_uid", "")), str(item.get("method", "")))
            not in selected
        ]
        if missing:
            raise RuntimeError(f"Configured manuscript cases were not selected: {missing}")
    write_index(all_cases, output_root, config)
    print(f"[INFO] Case mechanism figures written to: {output_root}")


if __name__ == "__main__":
    main()
