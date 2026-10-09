from __future__ import annotations

import argparse
import os
import re
import shutil
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pyVHR as vhr
from pyVHR.BVP import BPfilter, RGB_sig_to_BVP
from pyVHR.utils.errors import getErrors, printErrors

from gamma_rppg.common.config import dump_resolved_config, ensure_dir, load_config, resolve_output_root
from gamma_rppg.signal.processing import BrightnessStats, GammaSignalProcessing
from gamma_rppg.experiments.methods import normalize_method_name, resolve_method
from gamma_rppg.datasets.pure import PUREDirectoryDataset
from gamma_rppg.common.result_store import AtomicCSVStore


ERROR_COLUMNS = ["RMSE", "MAE", "MAX", "PCC", "CCC", "SNR"]
IDENTITY_COLUMNS = [
    "dataset", "video_idx", "video_name", "sample_id", "sample_uid", "subject_id", "trial_id",
    "motion_type", "source_label", "input_type", "input_path", "transform_mode",
]
RAW_COLUMNS = IDENTITY_COLUMNS + [
    "gamma", "method", "method_function", "device_type", "fps", "seconds", "num_frames",
    "status", "error_message", *ERROR_COLUMNS,
]
BRIGHTNESS_COLUMNS = IDENTITY_COLUMNS + [
    "gamma", "fps", "seconds", "num_frames", *BrightnessStats.__dataclass_fields__.keys(),
]
RAW_COLUMNS = list(dict.fromkeys(RAW_COLUMNS))
BRIGHTNESS_COLUMNS = list(dict.fromkeys(BRIGHTNESS_COLUMNS))
RAW_KEY = ["dataset", "sample_uid", "gamma", "method", "transform_mode", "input_type"]
BRIGHTNESS_KEY = ["dataset", "sample_uid", "gamma", "transform_mode", "input_type"]


def scalar(value):
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.reshape(-1)[0])
        return float(np.nanmean(value))
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if value is None:
        return np.nan
    try:
        return float(value)
    except Exception:
        return value


def parse_csv_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None or value.strip().lower() in {"", "all"}:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_indices(value, count: int) -> List[int]:
    if value is None:
        return list(range(count))
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"all", "*"}:
            return list(range(count))
        if ":" in normalized:
            parts = [part.strip() for part in normalized.split(":")]
            start = int(parts[0]) if parts[0] else 0
            stop = int(parts[1]) if len(parts) > 1 and parts[1] else count
            step = int(parts[2]) if len(parts) > 2 and parts[2] else 1
            return [index for index in range(start, min(stop, count), step)]
        return [int(item) for item in normalized.split(",") if item.strip()]
    if isinstance(value, int):
        return [value]
    return [int(item) for item in value]


def select_datasets(config: Dict, cli_datasets: Optional[str]) -> Dict[str, Dict]:
    datasets_config = config["datasets"]
    names = parse_csv_list(cli_datasets)
    if names is None:
        names = config.get("run", {}).get("datasets", list(datasets_config.keys()))
    selected = {}
    for name in names:
        if name not in datasets_config:
            raise KeyError(f"Dataset '{name}' not found. Available: {list(datasets_config)}")
        selected[name] = datasets_config[name]
    return selected


def select_methods(config: Dict, cli_methods: Optional[str]) -> List[str]:
    methods = parse_csv_list(cli_methods)
    if methods is None:
        methods = config.get("run", {}).get("methods", config.get("methods", []))
    if isinstance(methods, str):
        methods = [methods]
    return [normalize_method_name(method) for method in methods]


def build_dataset(dataset_cfg: Dict):
    name = str(dataset_cfg["name"]).upper()
    loader = str(dataset_cfg.get("loader", "")).lower()
    if name == "PURE" or loader in {"pure_safe", "pure_directory", "pure_frames", "pure_multi"}:
        options = {
            "prefer_avi": bool(dataset_cfg.get("prefer_avi", True)),
            "allow_frames_fallback": bool(dataset_cfg.get("allow_frames_fallback", False)),
        }
        sources = dataset_cfg.get("sources")
        if sources:
            return PUREDirectoryDataset(sources=sources, **options)
        return PUREDirectoryDataset(
            dataset_cfg.get("video_DIR"),
            dataset_cfg.get("BVP_DIR", dataset_cfg.get("video_DIR")),
            **options,
        )
    return vhr.datasets.datasetFactory(
        dataset_cfg["name"],
        videodataDIR=dataset_cfg["video_DIR"],
        BVPdataDIR=dataset_cfg["BVP_DIR"],
    )


def get_fps_for_video(dataset, video_file: str) -> float:
    if os.path.isdir(str(video_file)):
        return float(getattr(dataset, "frameRate", 30.0))
    try:
        fps = float(vhr.extraction.get_fps(video_file))
        if fps > 0:
            return fps
    except Exception:
        pass
    return float(getattr(dataset, "frameRate", 30.0))


def _safe_component(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z._-]+", "_", value.strip())
    return value.strip("_") or "sample"


def derive_sample_metadata(dataset_key: str, dataset_cfg: Dict, video_file: str, metadata: Dict) -> Dict[str, str]:
    video_path = Path(str(video_file))
    sample_id = str(metadata.get("sample_id") or video_path.stem or video_path.name)
    source_label = str(metadata.get("source_label") or "")
    motion_type = str(metadata.get("motion_type") or "")
    subject_id = str(metadata.get("subject_id") or video_path.parent.name or sample_id)
    trial_id = str(metadata.get("trial_id") or sample_id)
    input_type = str(metadata.get("input_type") or ("frames_dir" if video_path.is_dir() else "video_file"))
    sample_uid = str(metadata.get("sample_uid") or "")
    if not sample_uid:
        roots = [Path(str(dataset_cfg["video_DIR"]))] if dataset_cfg.get("video_DIR") else []
        relative = None
        for root in roots:
            try:
                relative = video_path.resolve().relative_to(root.resolve())
                break
            except Exception:
                continue
        stable_part = str(relative if relative is not None else Path(video_path.parent.name) / video_path.name)
        sample_uid = f"{dataset_key}/{stable_part}"
    return {
        "video_name": video_path.name,
        "sample_id": sample_id,
        "sample_uid": sample_uid.replace("\\", "/"),
        "subject_id": subject_id,
        "trial_id": trial_id,
        "motion_type": motion_type,
        "source_label": source_label,
        "input_type": input_type,
        "input_path": str(metadata.get("input_path") or video_path),
    }


def extract_signal_for_gamma(
    video_file: str,
    fps: float,
    gamma: float,
    config: Dict,
    example_output_path: Optional[Path] = None,
) -> tuple:
    experiment = config.get("experiment", {})
    seconds = experiment.get("seconds", 30)
    extractor = GammaSignalProcessing(
        gamma_value=gamma,
        transform_mode=experiment.get("transform_mode", "pre_roi_gamma"),
        collect_brightness=bool(experiment.get("collect_brightness", True)),
        example_output_path=str(example_output_path) if example_output_path else None,
    )
    try:
        extractor.display_cuda_device()
    except Exception:
        pass
    try:
        extractor.choose_cuda_device(int(experiment.get("cuda_device", 0)))
    except Exception:
        pass
    extractor.visualize_skin = False
    extractor.set_skin_extractor(
        vhr.extraction.SkinExtractionConvexHull(experiment.get("skin_extractor_device", "GPU"))
    )
    if seconds is None or float(seconds) <= 0:
        extractor.set_total_frames(0)
        reported_seconds = 0.0
    else:
        extractor.set_total_frames(int(float(seconds) * fps))
        reported_seconds = float(seconds)
    holistic_signal, brightness_stats = extractor.extract_holistic_with_metrics(video_file)
    return holistic_signal, brightness_stats, reported_seconds


def bpm_from_bvps(bvps, fps: float, prefer_cuda: bool = True):
    if prefer_cuda and hasattr(vhr.BPM, "BVP_to_BPM_cuda"):
        return vhr.BPM.BVP_to_BPM_cuda(bvps, fps)
    return vhr.BPM.BVP_to_BPM(bvps, fps)


def should_run(existing: Optional[Dict], run_mode: str) -> bool:
    if existing is None:
        return True
    if run_mode == "retry-errors":
        return str(existing.get("status", "")) != "ok"
    return False


def _clear_dataset_output(dataset_dir: Path) -> None:
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)


def _example_path(config: Dict, dataset_out: Path, sample_uid: str, sample_position: int, gamma: float) -> Optional[Path]:
    quality = config.get("analysis", {}).get("quality_examples", {})
    if not bool(quality.get("enabled", True)):
        return None
    if sample_position >= int(quality.get("max_samples_per_dataset", 3)):
        return None
    allowed = [float(value) for value in quality.get("gammas", [1.0, 2.0, 2.8])]
    if not any(np.isclose(gamma, value) for value in allowed):
        return None
    folder = dataset_out / "quality_examples" / _safe_component(sample_uid)
    return folder / f"gamma_{gamma:g}.png"


def run_one_dataset(
    dataset_key: str,
    dataset_cfg: Dict,
    methods: List[str],
    config: Dict,
    output_root: Path,
    run_mode: str,
):
    dataset_out = output_root / dataset_key
    if run_mode == "overwrite":
        _clear_dataset_output(dataset_out)
    else:
        ensure_dir(dataset_out)

    raw_store = AtomicCSVStore(dataset_out / "raw_results.csv", RAW_COLUMNS, RAW_KEY)
    brightness_store = AtomicCSVStore(
        dataset_out / "brightness_metrics.csv", BRIGHTNESS_COLUMNS, BRIGHTNESS_KEY
    )

    dataset = build_dataset(dataset_cfg)
    video_count = len(dataset.videoFilenames)
    print(f"[INFO] Dataset {dataset_key}: found {video_count} videos")
    for index, path in enumerate(dataset.videoFilenames[:10]):
        print(f"[INFO]   video[{index}] = {path}")

    experiment = config.get("experiment", {})
    transform_mode = "pre_roi_gamma"
    gamma_values = [float(value) for value in experiment.get("gamma_values", [1.0])]
    prefer_cuda = bool(experiment.get("prefer_cuda", True))
    window_size = float(experiment.get("wsize", 8))
    step = float(experiment.get("step", 1))
    filter_params = experiment.get("bp_filter", {"order": 6, "minHz": 0.65, "maxHz": 4.0})
    video_indices = parse_indices(
        dataset_cfg.get("video_indices", experiment.get("video_indices", "all")), video_count
    )

    method_specs = []
    for method in methods:
        spec = resolve_method(vhr.BVP, method, prefer_cuda=prefer_cuda)
        method_specs.append(spec)
        print(f"[INFO] Method {method}: {spec.function_name}, device={spec.device_type}")

    skipped_tasks = 0
    executed_tasks = 0
    for sample_position, video_idx in enumerate(video_indices):
        video_file = dataset.getVideoFilename(video_idx)
        native_metadata = dataset.getSampleMetadata(video_idx) if hasattr(dataset, "getSampleMetadata") else {}
        metadata = derive_sample_metadata(dataset_key, dataset_cfg, video_file, native_metadata)
        identity = {
            "dataset": dataset_key,
            "video_idx": video_idx,
            **metadata,
            "transform_mode": transform_mode,
        }
        try:
            signal_file = dataset.getSigFilename(video_idx)
            signal_gt = dataset.readSigfile(signal_file)
            bpm_gt, times_gt = signal_gt.getBPM(window_size)
            fps = get_fps_for_video(dataset, video_file)
        except Exception as error:
            print(f"[ERROR] Cannot initialize {metadata['sample_uid']}: {error}")
            traceback.print_exc()
            continue

        for gamma in gamma_values:
            task_base = {**identity, "gamma": gamma}
            specs_to_run = []
            for spec in method_specs:
                key = {**task_base, "method": spec.normalized_name}
                if should_run(raw_store.get(key), run_mode):
                    specs_to_run.append(spec)
                else:
                    skipped_tasks += 1
            brightness_existing = brightness_store.get(task_base)
            if not specs_to_run and brightness_existing is not None:
                print(f"[SKIP] completed: {metadata['sample_uid']} gamma={gamma:g}")
                continue

            print(f"\n[INFO] ===== {dataset_key} {metadata['sample_uid']} gamma={gamma:g} =====")
            start_time = time.time()
            try:
                example_path = _example_path(
                    config, dataset_out, metadata["sample_uid"], sample_position, gamma
                )
                holistic_signal, brightness_stats, seconds = extract_signal_for_gamma(
                    video_file, fps, gamma, config, example_path
                )
                windowed_signal, times_estimated = vhr.extraction.sig_windowing(
                    holistic_signal, window_size, step, fps
                )
                num_frames = int(holistic_signal.shape[0]) if hasattr(holistic_signal, "shape") else 0
                brightness_store.upsert({
                    **task_base,
                    "fps": fps,
                    "seconds": seconds,
                    "num_frames": num_frames,
                    **brightness_stats,
                })
            except Exception as error:
                message = f"Signal extraction failed: {error!r}"
                print(f"[ERROR] {message}")
                traceback.print_exc()
                for spec in specs_to_run:
                    raw_store.upsert({
                        **task_base,
                        "method": spec.normalized_name,
                        "method_function": spec.function_name,
                        "device_type": spec.device_type,
                        "fps": fps,
                        "seconds": experiment.get("seconds", 30),
                        "num_frames": 0,
                        "status": "signal_error",
                        "error_message": message,
                    })
                    executed_tasks += 1
                continue

            for spec in specs_to_run:
                row = {
                    **task_base,
                    "method": spec.normalized_name,
                    "method_function": spec.function_name,
                    "device_type": spec.device_type,
                    "fps": fps,
                    "seconds": seconds,
                    "num_frames": num_frames,
                }
                try:
                    method_function = getattr(vhr.BVP, spec.function_name)
                    bvps = RGB_sig_to_BVP(
                        windowed_signal,
                        fps,
                        device_type=spec.device_type,
                        method=method_function,
                        params=dict(spec.params),
                    )
                    call_filter = dict(filter_params)
                    call_filter["fps"] = fps
                    bvps = vhr.BVP.apply_filter(bvps, BPfilter, params=call_filter)
                    bpm_estimated = bpm_from_bvps(bvps, fps, prefer_cuda=prefer_cuda)
                    RMSE, MAE, MAX, PCC, CCC, SNR = getErrors(
                        bvps, fps, bpm_estimated, bpm_gt, times_estimated, times_gt
                    )
                    print(
                        f"[RESULT] sample={metadata['sample_uid']} gamma={gamma:g} "
                        f"method={spec.normalized_name}"
                    )
                    printErrors(RMSE, MAE, MAX, PCC, CCC, SNR)
                    row.update({
                        "status": "ok", "error_message": "", "RMSE": scalar(RMSE),
                        "MAE": scalar(MAE), "MAX": scalar(MAX), "PCC": scalar(PCC),
                        "CCC": scalar(CCC), "SNR": scalar(SNR),
                    })
                except Exception as error:
                    message = repr(error)
                    print(f"[ERROR] BVP failed: {message}")
                    traceback.print_exc()
                    row.update({"status": "method_error", "error_message": message})
                raw_store.upsert(row)
                executed_tasks += 1

            print(f"[INFO] Gamma task completed in {time.time() - start_time:.1f}s")

    print(
        f"[INFO] Finished {dataset_key}. executed={executed_tasks}, skipped={skipped_tasks}, "
        f"rows={len(raw_store)}"
    )


def main():
    parser = argparse.ArgumentParser(description="Run v5 pre-ROI inline Gamma rPPG experiments.")
    parser.add_argument("--config", default="configs/full_pre_roi_experiment.yaml")
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset keys, or all.")
    parser.add_argument("--methods", default=None, help="Comma-separated methods, or all configured methods.")
    parser.add_argument(
        "--output-root", default=None,
        help="Override the complete pre_roi_gamma result directory.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="Skip every existing task (default).")
    mode.add_argument("--retry-errors", action="store_true", help="Keep successful rows and rerun errors/missing tasks.")
    mode.add_argument("--overwrite", action="store_true", help="Delete selected dataset result folders and rerun them.")
    args = parser.parse_args()

    run_mode = "overwrite" if args.overwrite else "retry-errors" if args.retry_errors else "resume"
    config = load_config(args.config)
    selected_datasets = select_datasets(config, args.datasets)
    methods = select_methods(config, args.methods)
    output_root = ensure_dir(resolve_output_root(config, args.output_root))
    dump_resolved_config(config, output_root / "resolved_config.yaml")

    print(f"[INFO] Experiment module: pre_roi_gamma")
    print(f"[INFO] Output root: {output_root}")
    print(f"[INFO] Run mode: {run_mode}")
    print(f"[INFO] Datasets: {list(selected_datasets)}")
    print(f"[INFO] Methods: {methods}")
    for dataset_key, dataset_cfg in selected_datasets.items():
        run_one_dataset(dataset_key, dataset_cfg, methods, config, output_root, run_mode)
    print("[INFO] All requested pre-ROI experiments finished.")
    print(f"[INFO] Summarize with: python -m src.summarize_results --config {args.config}")


if __name__ == "__main__":
    main()
