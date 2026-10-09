from __future__ import annotations

import argparse
import csv
import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np
import pyVHR as vhr
from pyVHR.BVP import BPfilter, RGB_sig_to_BVP
from pyVHR.utils.errors import getErrors

from gamma_rppg.common.config import ensure_dir, load_config, resolve_output_root
from gamma_rppg.signal.processing import GammaSignalProcessing
from gamma_rppg.experiments.methods import normalize_method_name, resolve_method
from gamma_rppg.experiments.pre_roi import (
    bpm_from_bvps,
    build_dataset,
    derive_sample_metadata,
    get_fps_for_video,
    scalar,
    select_datasets,
    select_methods,
)


COLUMNS = [
    "dataset", "video_idx", "video_name", "sample_id", "sample_uid", "subject_id", "trial_id",
    "motion_type", "source_label", "input_type", "gamma", "method", "method_function",
    "device_type", "status", "fps", "num_frames", "RMSE", "MAE", "MAX", "PCC", "CCC",
    "SNR", "error_message",
]


def write_rows(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in COLUMNS})


def quick_test_indices(dataset, dataset_key: str, dataset_cfg: Dict) -> List[int]:
    if dataset_key.upper() != "PURE":
        return [int(dataset_cfg.get("quick_test_video_idx", 0))]
    # Test at least one sample from every configured PURE source/motion condition.
    indices = []
    seen = set()
    for index in range(len(dataset.videoFilenames)):
        metadata = dataset.getSampleMetadata(index)
        source = metadata.get("source_label") or metadata.get("motion_type") or "PURE"
        if source not in seen:
            seen.add(source)
            indices.append(index)
    return indices


def main():
    parser = argparse.ArgumentParser(description="Quickly test dataset-method compatibility under pyVHR.")
    parser.add_argument("--config", default="configs/quick_pre_roi_test.yaml")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--methods", default=None)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    datasets = select_datasets(config, args.datasets)
    methods = select_methods(config, args.methods)
    experiment = config.get("experiment", {})
    gammas = [float(value) for value in experiment.get("gamma_values", [1.0])]
    prefer_cuda = bool(experiment.get("prefer_cuda", True))
    window_size = float(experiment.get("wsize", 8))
    step = float(experiment.get("step", 1))
    seconds = float(experiment.get("seconds", 10))
    skin_device = experiment.get("skin_extractor_device", "GPU")
    transform_mode = experiment.get("transform_mode", "pre_roi_gamma")
    cuda_device = int(experiment.get("cuda_device", 0))
    filter_params = experiment.get("bp_filter", {"order": 6, "minHz": 0.65, "maxHz": 4.0})
    output_root = ensure_dir(resolve_output_root(config, args.output_root))
    print(f"[INFO] Quick-test output root: {output_root}")

    for dataset_key, dataset_cfg in datasets.items():
        rows: List[Dict] = []
        output_csv = output_root / dataset_key / "quick_test_results.csv"
        print(f"\n[INFO] ===== Dataset {dataset_key} =====")
        try:
            dataset = build_dataset(dataset_cfg)
            indices = quick_test_indices(dataset, dataset_key, dataset_cfg)
        except Exception as error:
            message = f"dataset init failed: {error!r}"
            print(f"[ERROR] {message}")
            for method in methods:
                for gamma in gammas:
                    rows.append({
                        "dataset": dataset_key, "gamma": gamma, "method": method,
                        "status": "dataset_error", "error_message": message,
                    })
            write_rows(output_csv, rows)
            continue

        for video_idx in indices:
            try:
                video_file = dataset.getVideoFilename(video_idx)
                native = dataset.getSampleMetadata(video_idx) if hasattr(dataset, "getSampleMetadata") else {}
                metadata = derive_sample_metadata(dataset_key, dataset_cfg, video_file, native)
                signal_file = dataset.getSigFilename(video_idx)
                signal_gt = dataset.readSigfile(signal_file)
                bpm_gt, times_gt = signal_gt.getBPM(window_size)
                fps = get_fps_for_video(dataset, video_file)
                print(f"[INFO] Test sample: {metadata['sample_uid']}, fps={fps}")
            except Exception as error:
                message = f"sample init failed: {error!r}"
                print(f"[ERROR] {message}")
                continue

            identity = {"dataset": dataset_key, "video_idx": video_idx, **metadata}
            for gamma in gammas:
                try:
                    extractor = GammaSignalProcessing(gamma, transform_mode=transform_mode, collect_brightness=False)
                    try:
                        extractor.choose_cuda_device(cuda_device)
                    except Exception:
                        pass
                    extractor.visualize_skin = False
                    extractor.set_skin_extractor(vhr.extraction.SkinExtractionConvexHull(skin_device))
                    extractor.set_total_frames(int(seconds * fps))
                    signal = extractor.extract_holistic(video_file)
                    windowed_signal, times_estimated = vhr.extraction.sig_windowing(
                        signal, window_size, step, fps
                    )
                    num_frames = int(signal.shape[0])
                    signal_status, signal_error = "ok", ""
                except Exception as error:
                    signal_status, signal_error = "signal_error", repr(error)
                    num_frames, windowed_signal, times_estimated = 0, [], []
                    print(f"[ERROR] Signal extraction failed: {signal_error}")
                    traceback.print_exc()

                for method in methods:
                    try:
                        normalized = normalize_method_name(method)
                        spec = resolve_method(vhr.BVP, normalized, prefer_cuda=prefer_cuda)
                        if signal_status != "ok":
                            raise RuntimeError(signal_error)
                        function = getattr(vhr.BVP, spec.function_name)
                        bvps = RGB_sig_to_BVP(
                            windowed_signal, fps, device_type=spec.device_type,
                            method=function, params=dict(spec.params),
                        )
                        call_filter = dict(filter_params)
                        call_filter["fps"] = fps
                        bvps = vhr.BVP.apply_filter(bvps, BPfilter, params=call_filter)
                        bpm_estimated = bpm_from_bvps(bvps, fps, prefer_cuda=prefer_cuda)
                        RMSE, MAE, MAX, PCC, CCC, SNR = getErrors(
                            bvps, fps, bpm_estimated, bpm_gt, times_estimated, times_gt
                        )
                        row = {
                            **identity, "gamma": gamma, "method": normalized,
                            "method_function": spec.function_name, "device_type": spec.device_type,
                            "status": "ok", "fps": fps, "num_frames": num_frames,
                            "RMSE": scalar(RMSE), "MAE": scalar(MAE), "MAX": scalar(MAX),
                            "PCC": scalar(PCC), "CCC": scalar(CCC), "SNR": scalar(SNR),
                            "error_message": "",
                        }
                        print(f"[OK] {metadata['sample_uid']} gamma={gamma:g} method={normalized}")
                    except Exception as error:
                        row = {
                            **identity, "gamma": gamma, "method": method, "status": "error",
                            "fps": fps, "num_frames": num_frames, "error_message": repr(error),
                        }
                        print(f"[FAIL] {metadata['sample_uid']} gamma={gamma:g} method={method}: {error!r}")
                    rows.append(row)
        write_rows(output_csv, rows)
        print(f"[INFO] Quick-test CSV: {output_csv}")

    print("[INFO] Quick test completed.")


if __name__ == "__main__":
    main()
