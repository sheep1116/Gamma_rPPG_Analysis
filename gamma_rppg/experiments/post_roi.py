from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
import traceback
from pathlib import Path

from gamma_rppg.common.statistics import holm_adjust_list as holm_adjust
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyVHR as vhr
from pyVHR.BVP import BPfilter, RGB_sig_to_BVP
from pyVHR.extraction.sig_extraction_methods import holistic_mean
from pyVHR.extraction.sig_processing import SignalProcessingParams
from pyVHR.utils.errors import getErrors, printErrors

from gamma_rppg.common.config import dump_resolved_config, ensure_dir, load_config, resolve_mode_root
from gamma_rppg.signal.processing import iter_frames_inline
from gamma_rppg.signal.image_math import apply_gamma_pixels, make_gamma_table
from gamma_rppg.experiments.methods import resolve_method
from gamma_rppg.common.result_store import AtomicCSVStore
from gamma_rppg.experiments.pre_roi import (
    ERROR_COLUMNS,
    bpm_from_bvps,
    build_dataset,
    derive_sample_metadata,
    get_fps_for_video,
    parse_indices,
    scalar,
    select_datasets,
    select_methods,
    should_run,
)


POST_MODE = "post_roi_gamma_fixed_mask"
POST_COLUMNS = [
    "dataset", "video_idx", "video_name", "sample_id", "sample_uid", "subject_id", "trial_id",
    "motion_type", "source_label", "input_type", "input_path", "transform_mode", "roi_cache_version",
    "gamma", "method", "method_function", "device_type", "fps", "seconds", "num_frames",
    "face_detected_frames", "roi_area_mean", "valid_pixel_ratio_mean", "status", "error_message",
    *ERROR_COLUMNS,
]
POST_KEY = [
    "dataset", "sample_uid", "gamma", "method", "transform_mode", "roi_cache_version", "input_type"
]
QUALITY_COLUMNS = [
    "dataset", "sample_uid", "input_path", "input_type", "roi_cache_version", "fps", "num_frames",
    "face_detected_frames", "face_detection_ratio", "roi_area_mean", "roi_area_std",
    "valid_pixel_ratio_mean", "cache_used", "status", "error_message",
]
QUALITY_KEY = ["dataset", "sample_uid", "roi_cache_version", "input_type"]


def _safe(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("_") or "sample"


def _landmarks_from_result(result, width: int, height: int) -> Tuple[np.ndarray, bool]:
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


def _roi_crop_and_mask(
    image_rgb: np.ndarray, skin_extractor, landmarks: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    cropped, _ = skin_extractor.extract_skin(image_rgb, landmarks)
    white = np.full_like(image_rgb, 255, dtype=np.uint8)
    cropped_mask, _ = skin_extractor.extract_skin(white, landmarks)
    if cropped is None or cropped.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8), np.zeros((1, 1), dtype=bool)
    mask = np.any(cropped_mask > 0, axis=2) if cropped_mask is not None else np.any(cropped > 0, axis=2)
    return cropped.astype(np.uint8), mask


def _post_roi_signal(cropped: np.ndarray, mask: np.ndarray, gamma_table: np.ndarray) -> Tuple[np.ndarray, float]:
    if mask.size == 0 or not np.any(mask):
        return np.zeros(3, dtype=np.float32), 0.0
    transformed = np.zeros_like(cropped, dtype=np.uint8)
    transformed_pixels = apply_gamma_pixels(cropped[mask], gamma_table)
    transformed[mask] = transformed_pixels
    lo = int(SignalProcessingParams.RGB_LOW_TH)
    hi = int(SignalProcessingParams.RGB_HIGH_TH)
    signal = holistic_mean(transformed, np.int32(lo), np.int32(hi))
    valid = np.all((transformed_pixels >= lo) & (transformed_pixels <= hi), axis=1)
    ratio = float(np.mean(valid)) if valid.size else 0.0
    return np.asarray(signal, dtype=np.float32), ratio


def _cache_path(cache_root: Path, dataset_key: str, sample_uid: str) -> Path:
    digest = hashlib.sha1(sample_uid.encode("utf-8")).hexdigest()[:10]
    return cache_root / dataset_key / f"{_safe(sample_uid)}_{digest}.npz"


def extract_fixed_roi_signals(
    video_file: str,
    fps: float,
    gamma_values: Sequence[float],
    total_frames: int,
    skin_extractor,
    cache_path: Path,
    cache_version: str,
    use_cache: bool,
) -> Tuple[Dict[float, np.ndarray], Dict]:
    cached_landmarks = None
    cached_valid = None
    cache_used = False
    input_obj = Path(str(video_file))
    try:
        stat = input_obj.stat()
        input_signature = f"{input_obj.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    except Exception:
        input_signature = str(input_obj)

    if use_cache and cache_path.exists():
        try:
            cached = np.load(cache_path, allow_pickle=False)
            cached_version = str(cached["cache_version"].item())
            cached_fps = float(cached["fps"].item()) if "fps" in cached else fps
            candidate_landmarks = cached["landmarks"]
            candidate_valid = cached["face_valid"].astype(bool)
            cached_signature = str(cached["input_signature"].item()) if "input_signature" in cached else ""
            length_ok = total_frames > 0 and len(candidate_landmarks) >= total_frames
            if (cached_version == cache_version and abs(cached_fps - fps) < 1e-6
                    and length_ok and cached_signature == input_signature):
                cached_landmarks = candidate_landmarks
                cached_valid = candidate_valid
                cache_used = True
        except Exception as error:
            print(f"[WARN] Cannot load ROI cache {cache_path}: {error!r}")

    tables = {float(g): make_gamma_table(float(g)) for g in gamma_values}
    signals = {float(g): [] for g in gamma_values}
    ratio_values = {float(g): [] for g in gamma_values}
    landmarks_rows: List[np.ndarray] = []
    valid_rows: List[bool] = []
    roi_areas: List[int] = []
    face_detected = 0
    frame_count = 0

    face_mesh_context = None
    face_mesh = None
    if not cache_used:
        face_mesh_context = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        face_mesh = face_mesh_context.__enter__()

    try:
        for frame_index, frame in enumerate(iter_frames_inline(video_file)):
            if total_frames > 0 and frame_index >= total_frames:
                break
            original_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if cache_used and frame_index < len(cached_landmarks):
                landmarks = cached_landmarks[frame_index].astype(np.float32)
                face_valid = bool(cached_valid[frame_index])
            else:
                result = face_mesh.process(original_rgb) if face_mesh is not None else None
                landmarks, face_valid = _landmarks_from_result(result, original_rgb.shape[1], original_rgb.shape[0])

            landmarks_rows.append(landmarks)
            valid_rows.append(face_valid)
            frame_count += 1
            if face_valid:
                face_detected += 1
                cropped, roi_mask = _roi_crop_and_mask(original_rgb, skin_extractor, landmarks)
            else:
                cropped = np.zeros((1, 1, 3), dtype=np.uint8)
                roi_mask = np.zeros((1, 1), dtype=bool)
            roi_areas.append(int(np.sum(roi_mask)))

            for gamma in gamma_values:
                mean_rgb, ratio = _post_roi_signal(cropped, roi_mask, tables[float(gamma)])
                signals[float(gamma)].append(mean_rgb)
                ratio_values[float(gamma)].append(ratio)
    finally:
        if face_mesh_context is not None:
            face_mesh_context.__exit__(None, None, None)

    if not cache_used and landmarks_rows:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            cache_version=np.asarray(cache_version),
            landmarks=np.asarray(landmarks_rows, dtype=np.float32),
            face_valid=np.asarray(valid_rows, dtype=np.uint8),
            fps=np.asarray(float(fps)),
            frame_count=np.asarray(frame_count),
            input_signature=np.asarray(input_signature),
        )

    arrays = {gamma: np.asarray(values, dtype=np.float32) for gamma, values in signals.items()}
    all_ratios = [value for gamma in gamma_values for value in ratio_values[float(gamma)]]
    quality = {
        "num_frames": frame_count,
        "face_detected_frames": face_detected,
        "face_detection_ratio": float(face_detected / frame_count) if frame_count else 0.0,
        "roi_area_mean": float(np.mean(roi_areas)) if roi_areas else np.nan,
        "roi_area_std": float(np.std(roi_areas)) if roi_areas else np.nan,
        "valid_pixel_ratio_mean": float(np.mean(all_ratios)) if all_ratios else np.nan,
        "cache_used": cache_used,
    }
    return arrays, quality


def clear_dataset(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def run_post_dataset(
    dataset_key: str,
    dataset_cfg: Dict,
    methods: List[str],
    config: Dict,
    output_root: Path,
    run_mode: str,
) -> None:
    dataset_out = output_root / dataset_key
    if run_mode == "overwrite":
        clear_dataset(dataset_out)
    else:
        ensure_dir(dataset_out)
    raw_store = AtomicCSVStore(dataset_out / "post_roi_raw_results.csv", POST_COLUMNS, POST_KEY)
    quality_store = AtomicCSVStore(dataset_out / "post_roi_quality_control.csv", QUALITY_COLUMNS, QUALITY_KEY)

    dataset = build_dataset(dataset_cfg)
    experiment = config.get("experiment", {})
    gamma_values = [float(g) for g in experiment.get("gamma_values", [1.0])]
    seconds = float(experiment.get("seconds", 30))
    wsize = float(experiment.get("wsize", 8))
    step = float(experiment.get("step", 1))
    prefer_cuda = bool(experiment.get("prefer_cuda", True))
    try:
        device_selector = vhr.extraction.SignalProcessing()
        device_selector.choose_cuda_device(int(experiment.get("cuda_device", 0)))
    except Exception:
        pass
    filter_params = experiment.get("bp_filter", {"order": 6, "minHz": 0.65, "maxHz": 4.0})
    cache_cfg = config.get("post_roi", {}).get("roi_cache", {})
    cache_version = str(cache_cfg.get("version", "v1"))
    use_cache = bool(cache_cfg.get("enabled", True))
    cache_root = ensure_dir(output_root / "roi_cache")
    indices = parse_indices(dataset_cfg.get("video_indices", "all"), len(dataset.videoFilenames))

    specs = [resolve_method(vhr.BVP, name, prefer_cuda=prefer_cuda) for name in methods]
    skin_extractor = vhr.extraction.SkinExtractionConvexHull(experiment.get("skin_extractor_device", "GPU"))
    executed = 0
    skipped = 0

    for video_idx in indices:
        video_file = dataset.getVideoFilename(video_idx)
        native = dataset.getSampleMetadata(video_idx) if hasattr(dataset, "getSampleMetadata") else {}
        metadata = derive_sample_metadata(dataset_key, dataset_cfg, video_file, native)
        identity = {
            "dataset": dataset_key, "video_idx": video_idx, **metadata,
            "transform_mode": POST_MODE, "roi_cache_version": cache_version,
        }
        try:
            fps = get_fps_for_video(dataset, video_file)
            signal_gt = dataset.readSigfile(dataset.getSigFilename(video_idx))
            bpm_gt, times_gt = signal_gt.getBPM(wsize)
        except Exception as error:
            print(f"[ERROR] Cannot initialize post-ROI sample {metadata['sample_uid']}: {error!r}")
            traceback.print_exc()
            continue

        missing_specs = {}
        for gamma in gamma_values:
            needed = []
            for spec in specs:
                key = {**identity, "gamma": gamma, "method": spec.normalized_name}
                if should_run(raw_store.get(key), run_mode):
                    needed.append(spec)
                else:
                    skipped += 1
            if needed:
                missing_specs[gamma] = needed
        if not missing_specs:
            print(f"[SKIP] post-ROI completed: {metadata['sample_uid']}")
            continue

        print(f"\n[INFO] ===== post-ROI {dataset_key} {metadata['sample_uid']} =====")
        started = time.time()
        try:
            total_frames = int(seconds * fps) if seconds > 0 else 0
            signal_map, quality = extract_fixed_roi_signals(
                video_file, fps, gamma_values, total_frames, skin_extractor,
                _cache_path(cache_root, dataset_key, metadata["sample_uid"]),
                cache_version, use_cache,
            )
            quality_store.upsert({
                "dataset": dataset_key,
                "sample_uid": metadata["sample_uid"],
                "input_path": metadata["input_path"],
                "input_type": metadata["input_type"],
                "roi_cache_version": cache_version,
                "fps": fps,
                **quality,
                "status": "ok",
                "error_message": "",
            })
        except Exception as error:
            message = f"post-ROI extraction failed: {error!r}"
            print(f"[ERROR] {message}")
            traceback.print_exc()
            quality_store.upsert({
                "dataset": dataset_key, "sample_uid": metadata["sample_uid"],
                "input_path": metadata["input_path"], "input_type": metadata["input_type"],
                "roi_cache_version": cache_version, "fps": fps, "status": "signal_error",
                "error_message": message,
            })
            for gamma, needed in missing_specs.items():
                for spec in needed:
                    raw_store.upsert({
                        **identity, "gamma": gamma, "method": spec.normalized_name,
                        "method_function": spec.function_name, "device_type": spec.device_type,
                        "fps": fps, "seconds": seconds, "num_frames": 0,
                        "status": "signal_error", "error_message": message,
                    })
                    executed += 1
            continue

        for gamma, needed in missing_specs.items():
            signal = signal_map[float(gamma)]
            try:
                windowed_signal, times_estimated = vhr.extraction.sig_windowing(signal, wsize, step, fps)
            except Exception as error:
                for spec in needed:
                    raw_store.upsert({
                        **identity, "gamma": gamma, "method": spec.normalized_name,
                        "method_function": spec.function_name, "device_type": spec.device_type,
                        "fps": fps, "seconds": seconds, "num_frames": len(signal),
                        **{k: quality.get(k, "") for k in ["face_detected_frames", "roi_area_mean", "valid_pixel_ratio_mean"]},
                        "status": "window_error", "error_message": repr(error),
                    })
                    executed += 1
                continue
            for spec in needed:
                row = {
                    **identity, "gamma": gamma, "method": spec.normalized_name,
                    "method_function": spec.function_name, "device_type": spec.device_type,
                    "fps": fps, "seconds": seconds, "num_frames": len(signal),
                    **{k: quality.get(k, "") for k in ["face_detected_frames", "roi_area_mean", "valid_pixel_ratio_mean"]},
                }
                try:
                    method_function = getattr(vhr.BVP, spec.function_name)
                    bvps = RGB_sig_to_BVP(
                        windowed_signal, fps, device_type=spec.device_type,
                        method=method_function, params=dict(spec.params),
                    )
                    call_filter = dict(filter_params)
                    call_filter["fps"] = fps
                    bvps = vhr.BVP.apply_filter(bvps, BPfilter, params=call_filter)
                    bpm_estimated = bpm_from_bvps(bvps, fps, prefer_cuda=prefer_cuda)
                    values = getErrors(bvps, fps, bpm_estimated, bpm_gt, times_estimated, times_gt)
                    print(f"[RESULT] post-ROI sample={metadata['sample_uid']} gamma={gamma:g} method={spec.normalized_name}")
                    printErrors(*values)
                    row.update({"status": "ok", "error_message": ""})
                    row.update(dict(zip(ERROR_COLUMNS, [scalar(v) for v in values])))
                except Exception as error:
                    row.update({"status": "method_error", "error_message": repr(error)})
                    print(f"[ERROR] post-ROI BVP failed: {error!r}")
                    traceback.print_exc()
                raw_store.upsert(row)
                executed += 1
        print(f"[INFO] post-ROI sample completed in {time.time() - started:.1f}s")

    print(f"[INFO] Finished post-ROI {dataset_key}. executed={executed}, skipped={skipped}, rows={len(raw_store)}")


def bootstrap_ci(values: np.ndarray, iterations: int, seed: int) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    means = np.empty(iterations)
    for index in range(iterations):
        means[index] = np.mean(rng.choice(values, size=values.size, replace=True))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))




def wilcoxon_safe(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, str]:
    from scipy.stats import wilcoxon
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    if diff.size < 2:
        return np.nan, np.nan, "insufficient_pairs"
    if np.allclose(diff, 0):
        return 0.0, 1.0, "all_differences_zero"
    kwargs = {"zero_method": "wilcox", "alternative": "two-sided"}
    try:
        kwargs["method"] = "approx" if np.any(np.isclose(diff, 0)) else "auto"
        stat, pvalue = wilcoxon(x, y, **kwargs)
    except TypeError:
        kwargs.pop("method", None)
        kwargs["mode"] = "approx" if np.any(np.isclose(diff, 0)) else "auto"
        stat, pvalue = wilcoxon(x, y, **kwargs)
    return float(stat), float(pvalue), "ok"



def _aggregate_post(frame: pd.DataFrame, iterations: int, seed: int) -> pd.DataFrame:
    rows = []
    for index, ((method, gamma), group) in enumerate(frame.groupby(["method", "gamma"])):
        row = {"dataset": group["dataset"].iloc[0], "method": method, "gamma": gamma,
               "n": int(group["sample_uid"].nunique())}
        for metric in ERROR_COLUMNS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values)) if values.size else np.nan
            row[f"{metric}_median"] = float(np.median(values)) if values.size else np.nan
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else np.nan
            low, high = bootstrap_ci(values, iterations, seed + index * 97)
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_post_dataset(dataset_key: str, config: Dict, post_root: Path) -> None:
    dataset_dir = post_root / dataset_key
    raw_path = dataset_dir / "post_roi_raw_results.csv"
    if not raw_path.exists():
        return
    raw = pd.read_csv(raw_path)
    raw["gamma"] = pd.to_numeric(raw["gamma"], errors="coerce")
    for metric in ERROR_COLUMNS:
        raw[metric] = pd.to_numeric(raw.get(metric), errors="coerce")
    gammas = sorted(float(g) for g in config.get("experiment", {}).get("gamma_values", raw["gamma"].dropna().unique()))
    iterations = int(config.get("analysis", {}).get("bootstrap_iterations", 2000))
    seed = int(config.get("analysis", {}).get("random_seed", 2026))
    ok = raw[raw["status"] == "ok"].copy()
    summary = _aggregate_post(ok, iterations, seed) if not ok.empty else pd.DataFrame()
    summary.to_csv(dataset_dir / "post_roi_summary_by_gamma.csv", index=False, encoding="utf-8-sig")

    gamma_set = {round(g, 10) for g in gammas}
    complete_rows = []
    complete_keys = []
    for (sample_uid, method), group in ok.groupby(["sample_uid", "method"]):
        present = {round(float(g), 10) for g in group["gamma"].dropna().unique()}
        complete = gamma_set.issubset(present)
        complete_rows.append({
            "sample_uid": sample_uid, "method": method,
            "successful_gamma_count": len(present & gamma_set),
            "expected_gamma_count": len(gamma_set), "complete": complete,
        })
        if complete:
            complete_keys.append((sample_uid, method))
    complete_table = pd.DataFrame(complete_rows)
    complete_table.to_csv(dataset_dir / "post_roi_complete_case_samples.csv", index=False, encoding="utf-8-sig")
    if complete_keys:
        key_index = pd.MultiIndex.from_tuples(complete_keys, names=["sample_uid", "method"])
        paired = ok.set_index(["sample_uid", "method"])
        paired = paired[paired.index.isin(key_index)].reset_index()
    else:
        paired = ok.iloc[0:0].copy()
    paired_summary = _aggregate_post(paired, iterations, seed + 11) if not paired.empty else pd.DataFrame()
    paired_summary.to_csv(
        dataset_dir / "post_roi_summary_paired_by_gamma.csv", index=False, encoding="utf-8-sig"
    )

    success_rows = []
    for method, method_group in raw.groupby("method"):
        union_count = method_group["sample_uid"].nunique()
        for gamma in gammas:
            group = method_group[np.isclose(method_group["gamma"], gamma)]
            attempted = group["sample_uid"].nunique()
            success = group.loc[group["status"] == "ok", "sample_uid"].nunique()
            success_rows.append({
                "method": method, "gamma": gamma, "sample_count": union_count,
                "attempted_count": attempted, "success_count": success,
                "failed_count": max(attempted - success, 0),
                "missing_count": max(union_count - attempted, 0),
                "success_rate": success / union_count if union_count else np.nan,
            })
    success = pd.DataFrame(success_rows)
    success.to_csv(dataset_dir / "post_roi_success_rate.csv", index=False, encoding="utf-8-sig")

    figures = ensure_dir(dataset_dir / "figures")
    if not paired_summary.empty:
        fig, ax = plt.subplots(figsize=(9, 6))
        for method, group in paired_summary.groupby("method"):
            group = group.sort_values("gamma")
            ax.plot(group["gamma"], group["MAE_mean"], marker="o", label=method)
        ax.set_xlabel("Gamma")
        ax.set_ylabel("MAE")
        ax.set_title(f"{dataset_key}: post-ROI paired mean MAE")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / "post_roi_paired_mean_MAE.png", dpi=180)
        plt.close(fig)
    if not success.empty:
        fig, ax = plt.subplots(figsize=(9, 6))
        for method, group in success.groupby("method"):
            ax.plot(group["gamma"], group["success_rate"] * 100, marker="o", label=method)
        ax.set_xlabel("Gamma")
        ax.set_ylabel("Success rate (%)")
        ax.set_ylim(0, 105)
        ax.set_title(f"{dataset_key}: post-ROI success rate")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / "post_roi_success_rate.png", dpi=180)
        plt.close(fig)
    print(f"[INFO] Summarized post-ROI dataset: {dataset_dir}")


def _mode_success(raw: pd.DataFrame, mode_name: str, gammas: Sequence[float]) -> pd.DataFrame:
    rows = []
    for method, method_group in raw.groupby("method"):
        total = method_group["sample_uid"].nunique()
        for gamma in gammas:
            group = method_group[np.isclose(method_group["gamma"], gamma)]
            success = group.loc[group["status"] == "ok", "sample_uid"].nunique()
            rows.append({
                "method": method, "gamma": gamma, "mode": mode_name,
                "sample_count": total, "success_count": success,
                "success_rate": success / total if total else np.nan,
            })
    return pd.DataFrame(rows)


def compare_dataset(dataset_key: str, config: Dict, post_root: Path) -> None:
    post_dir = post_root / dataset_key
    post_path = post_dir / "post_roi_raw_results.csv"
    pre_root = resolve_mode_root(config, "pre_roi_gamma")
    pre_path = pre_root / dataset_key / "raw_results.csv"
    if not post_path.exists() or not pre_path.exists():
        print(f"[WARN] Cannot compare {dataset_key}: missing pre or post results")
        return

    pre_all = pd.read_csv(pre_path)
    post_all = pd.read_csv(post_path)
    for frame in [pre_all, post_all]:
        frame["gamma"] = pd.to_numeric(frame["gamma"], errors="coerce")
    pre = pre_all[pre_all["status"] == "ok"].copy()
    post = post_all[post_all["status"] == "ok"].copy()
    keys = ["dataset", "sample_uid", "gamma", "method"]
    common = pre.merge(post, on=keys, suffixes=("_pre", "_post"), how="inner")
    for metric in ERROR_COLUMNS:
        common[f"{metric}_pre"] = pd.to_numeric(common[f"{metric}_pre"], errors="coerce")
        common[f"{metric}_post"] = pd.to_numeric(common[f"{metric}_post"], errors="coerce")
        common[f"absolute_{metric}_post_minus_pre"] = common[f"{metric}_post"] - common[f"{metric}_pre"]
    common.to_csv(post_dir / "pre_vs_post_common_task_results.csv", index=False, encoding="utf-8-sig")

    gammas = sorted(float(g) for g in config.get("experiment", {}).get("gamma_values", common["gamma"].dropna().unique()))
    gamma_set = {round(g, 10) for g in gammas}
    complete_keys = []
    complete_rows = []
    for (sample_uid, method), group in common.groupby(["sample_uid", "method"]):
        present = {round(float(g), 10) for g in group["gamma"].dropna().unique()}
        complete = gamma_set.issubset(present)
        complete_rows.append({
            "sample_uid": sample_uid, "method": method, "complete": complete,
            "successful_common_gamma_count": len(present & gamma_set),
            "expected_gamma_count": len(gamma_set),
        })
        if complete:
            complete_keys.append((sample_uid, method))
    pd.DataFrame(complete_rows).to_csv(
        post_dir / "pre_vs_post_complete_case_samples.csv", index=False, encoding="utf-8-sig"
    )
    if complete_keys:
        index = pd.MultiIndex.from_tuples(complete_keys, names=["sample_uid", "method"])
        merged = common.set_index(["sample_uid", "method"])
        merged = merged[merged.index.isin(index)].reset_index()
    else:
        merged = common.iloc[0:0].copy()
    if merged.empty:
        print(f"[WARN] No complete common pre/post samples for {dataset_key}")
        return

    baseline = float(config.get("analysis", {}).get("baseline_gamma", 1.0))
    base_columns = ["sample_uid", "method", *[f"{m}_pre" for m in ERROR_COLUMNS], *[f"{m}_post" for m in ERROR_COLUMNS]]
    base = merged[np.isclose(merged["gamma"], baseline)][base_columns].copy()
    base = base.rename(columns={f"{m}_pre": f"{m}_pre_baseline" for m in ERROR_COLUMNS})
    base = base.rename(columns={f"{m}_post": f"{m}_post_baseline" for m in ERROR_COLUMNS})
    merged = merged.merge(base, on=["sample_uid", "method"], how="left")
    for metric in ERROR_COLUMNS:
        merged[f"delta_{metric}_pre"] = merged[f"{metric}_pre"] - merged[f"{metric}_pre_baseline"]
        merged[f"delta_{metric}_post"] = merged[f"{metric}_post"] - merged[f"{metric}_post_baseline"]
        merged[f"difference_in_delta_{metric}"] = merged[f"delta_{metric}_pre"] - merged[f"delta_{metric}_post"]
    merged.to_csv(post_dir / "pre_vs_post_paired_results.csv", index=False, encoding="utf-8-sig")

    iterations = int(config.get("analysis", {}).get("bootstrap_iterations", 2000))
    seed = int(config.get("analysis", {}).get("random_seed", 2026))
    summaries = []
    tests = []
    pvalue_groups: Dict[Tuple[str, str], List[int]] = {}
    for (method, gamma), group in merged.groupby(["method", "gamma"]):
        summary = {"dataset": dataset_key, "method": method, "gamma": gamma, "n": len(group)}
        for metric in ERROR_COLUMNS:
            for mode in ["pre", "post"]:
                values = group[f"{metric}_{mode}"].dropna().to_numpy(dtype=float)
                summary[f"{metric}_{mode}_mean"] = float(np.mean(values)) if values.size else np.nan
                low, high = bootstrap_ci(values, iterations, seed + len(summaries) * 31)
                summary[f"{metric}_{mode}_ci95_low"] = low
                summary[f"{metric}_{mode}_ci95_high"] = high
                delta_values = group[f"delta_{metric}_{mode}"].dropna().to_numpy(dtype=float)
                summary[f"delta_{metric}_{mode}_mean"] = float(np.mean(delta_values)) if delta_values.size else np.nan
            summary[f"mean_difference_in_delta_{metric}"] = float(
                pd.to_numeric(group[f"difference_in_delta_{metric}"], errors="coerce").mean()
            )
        summaries.append(summary)

        for metric in ["MAE", "SNR"]:
            valid = group[[f"{metric}_pre", f"{metric}_post"]].dropna()
            stat, pvalue, status = wilcoxon_safe(
                valid[f"{metric}_pre"].to_numpy(), valid[f"{metric}_post"].to_numpy()
            )
            tests.append({
                "dataset": dataset_key, "method": method, "gamma": gamma, "metric": metric,
                "test": "wilcoxon_pre_vs_post_absolute", "n": len(valid), "statistic": stat,
                "p_value": pvalue, "p_holm": np.nan, "status": status,
                "mean_post_minus_pre": float((valid[f"{metric}_post"] - valid[f"{metric}_pre"]).mean()) if len(valid) else np.nan,
            })
            pvalue_groups.setdefault((method, metric), []).append(len(tests) - 1)

            delta_valid = group[[f"delta_{metric}_pre", f"delta_{metric}_post"]].dropna()
            stat, pvalue, status = wilcoxon_safe(
                delta_valid[f"delta_{metric}_pre"].to_numpy(), delta_valid[f"delta_{metric}_post"].to_numpy()
            )
            tests.append({
                "dataset": dataset_key, "method": method, "gamma": gamma, "metric": metric,
                "test": "wilcoxon_difference_in_delta", "n": len(delta_valid), "statistic": stat,
                "p_value": pvalue, "p_holm": np.nan, "status": status,
                "mean_post_minus_pre": float(
                    (delta_valid[f"delta_{metric}_post"] - delta_valid[f"delta_{metric}_pre"]).mean()
                ) if len(delta_valid) else np.nan,
            })
            pvalue_groups.setdefault((method, f"delta_{metric}"), []).append(len(tests) - 1)

    for _, indices in pvalue_groups.items():
        adjusted = holm_adjust([
            float(tests[i]["p_value"]) if np.isfinite(tests[i]["p_value"]) else 1.0 for i in indices
        ])
        for i, value in zip(indices, adjusted):
            tests[i]["p_holm"] = value

    summary_frame = pd.DataFrame(summaries)
    test_frame = pd.DataFrame(tests)
    summary_frame.to_csv(post_dir / "pre_vs_post_summary.csv", index=False, encoding="utf-8-sig")
    test_frame.to_csv(post_dir / "pre_vs_post_statistical_tests.csv", index=False, encoding="utf-8-sig")
    merged[np.isclose(merged["gamma"], baseline)].to_csv(
        post_dir / "pre_vs_post_baseline_check.csv", index=False, encoding="utf-8-sig"
    )

    success = pd.concat([
        _mode_success(pre_all, "pre_roi_gamma", gammas),
        _mode_success(post_all, "post_roi_gamma_fixed_mask", gammas),
    ], ignore_index=True)
    success.to_csv(post_dir / "pre_vs_post_success_rate.csv", index=False, encoding="utf-8-sig")

    figures = ensure_dir(post_dir / "figures")
    for metric in ["MAE", "SNR"]:
        fig, ax = plt.subplots(figsize=(9, 6))
        for method, group in summary_frame.groupby("method"):
            group = group.sort_values("gamma")
            ax.plot(group["gamma"], group[f"{metric}_pre_mean"], marker="o", label=f"{method} pre")
            ax.plot(group["gamma"], group[f"{metric}_post_mean"], marker="x", linestyle="--", label=f"{method} post")
        ax.set_xlabel("Gamma")
        ax.set_ylabel(metric)
        ax.set_title(f"{dataset_key}: pre-ROI vs post-ROI mean {metric}")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / f"pre_vs_post_mean_{metric}.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    for method, group in summary_frame.groupby("method"):
        group = group.sort_values("gamma")
        ax.plot(group["gamma"], group["delta_MAE_pre_mean"], marker="o", label=f"{method} pre")
        ax.plot(group["gamma"], group["delta_MAE_post_mean"], marker="x", linestyle="--", label=f"{method} post")
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Gamma")
    ax.set_ylabel("MAE change from own Gamma=1 baseline")
    ax.set_title(f"{dataset_key}: pre/post Gamma effect")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "pre_vs_post_delta_MAE.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    for (mode, method), group in success.groupby(["mode", "method"]):
        ax.plot(group["gamma"], group["success_rate"] * 100, marker="o", label=f"{method} {mode}")
    ax.set_xlabel("Gamma")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"{dataset_key}: pre/post success rate")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "pre_vs_post_success_rate.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    for method, group in summary_frame.groupby("method"):
        group = group.sort_values("gamma")
        ax.plot(group["gamma"], group["mean_difference_in_delta_MAE"], marker="o", label=method)
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Gamma")
    ax.set_ylabel("ΔMAE(pre) - ΔMAE(post)")
    ax.set_title(f"{dataset_key}: difference in Gamma effect by transform stage")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "pre_vs_post_difference_in_delta.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(merged["MAE_pre"], merged["MAE_post"], alpha=0.45)
    values = pd.concat([merged["MAE_pre"], merged["MAE_post"]]).dropna()
    if not values.empty:
        low, high = float(values.min()), float(values.max())
        ax.plot([low, high], [low, high], linestyle="--")
    ax.set_xlabel("pre-ROI MAE")
    ax.set_ylabel("post-ROI MAE")
    ax.set_title(f"{dataset_key}: paired MAE")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / "pre_vs_post_sample_scatter.png", dpi=180)
    plt.close(fig)

    line_data = merged[["MAE_pre", "MAE_post"]].dropna().head(250)
    fig, ax = plt.subplots(figsize=(7, 6))
    for _, row in line_data.iterrows():
        ax.plot([0, 1], [row["MAE_pre"], row["MAE_post"]], alpha=0.15)
    ax.set_xticks([0, 1], ["pre-ROI", "post-ROI"])
    ax.set_ylabel("MAE")
    ax.set_title(f"{dataset_key}: paired task MAE lines")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / "pre_vs_post_paired_lines.png", dpi=180)
    plt.close(fig)
    print(f"[INFO] Compared pre/post modes for {dataset_key}: {post_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-mask post-ROI experiment and compare with pre-ROI.")
    parser.add_argument("--config", default="configs/post_roi_analysis.yaml")
    parser.add_argument("--stage", choices=["experiment", "compare", "all"], default="all")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--methods", default=None)
    parser.add_argument("--output-root", default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--retry-errors", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_mode = "overwrite" if args.overwrite else "retry-errors" if args.retry_errors else "resume"
    config = load_config(args.config)
    output_root = ensure_dir(Path(args.output_root) if args.output_root else resolve_mode_root(config, "post_roi_analyse"))
    dump_resolved_config(config, output_root / "resolved_config.yaml")
    datasets = select_datasets(config, args.datasets)
    methods = select_methods(config, args.methods)
    print(f"[INFO] post_roi_analyse root: {output_root}")

    if args.stage in {"experiment", "all"}:
        for dataset_key, dataset_cfg in datasets.items():
            run_post_dataset(dataset_key, dataset_cfg, methods, config, output_root, run_mode)
            summarize_post_dataset(dataset_key, config, output_root)
    if args.stage in {"compare", "all"}:
        for dataset_key in datasets:
            summarize_post_dataset(dataset_key, config, output_root)
            compare_dataset(dataset_key, config, output_root)


if __name__ == "__main__":
    main()
