from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import pandas as pd

from gamma_rppg.common.config import ensure_dir, load_config, resolve_mode_root


_VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def natural_key(value):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value))]


def unique_paths(paths: Iterable[Path]) -> List[Path]:
    out, seen = [], set()
    for path in paths:
        try:
            key = str(path.resolve())
        except Exception:
            key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def find_video(json_file: Path, roots: List[Path]) -> Optional[Path]:
    stem = json_file.stem
    candidates: List[Path] = []
    for ext in sorted(_VIDEO_EXTS):
        candidates.append(json_file.parent / f"{stem}{ext}")
    for root in roots:
        trial = root / stem
        for ext in sorted(_VIDEO_EXTS):
            candidates.extend([root / f"{stem}{ext}", trial / f"{stem}{ext}"])
        if trial.is_dir():
            candidates.extend(
                sorted([p for p in trial.iterdir() if p.is_file() and p.suffix.lower() in _VIDEO_EXTS], key=natural_key)
            )
    valid = [p for p in unique_paths(candidates) if p.is_file() and p.stat().st_size > 0]
    return valid[0] if valid else None


def json_duration(json_file: Path, signal_rate: float) -> float:
    try:
        data = json.loads(json_file.read_text(encoding="utf-8"))
        package = data.get("/FullPackage", [])
        return float(len(package) / signal_rate) if signal_rate > 0 else float("nan")
    except Exception:
        return float("nan")


def count_pngs(json_file: Path, roots: List[Path]) -> int:
    stem = json_file.stem
    candidates = [json_file.parent / stem] + [root / stem for root in roots]
    for folder in unique_paths(candidates):
        if folder.is_dir():
            return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)
    return 0


def inspect_video(path: Path, decode_all: bool) -> Dict:
    capture = cv2.VideoCapture(str(path))
    opened = bool(capture.isOpened())
    fps = float(capture.get(cv2.CAP_PROP_FPS)) if opened else 0.0
    reported = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))) if opened else 0
    decoded = 0
    decode_failures = 0
    if opened and decode_all:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                decode_failures += 1
            else:
                decoded += 1
    elif opened:
        decoded = reported
    capture.release()
    duration = decoded / fps if fps > 0 else 0.0
    return {
        "avi_opened": opened,
        "fps": fps,
        "reported_frame_count": reported,
        "decoded_frame_count": decoded,
        "decode_failure_count": decode_failures,
        "video_duration_seconds": duration,
        "reported_minus_decoded": reported - decoded,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check PURE AVI integrity without running rPPG.")
    parser.add_argument("--config", default="configs/pure_avi_integrity.yaml")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    check = config.get("pure_avi_check", {})
    expected_fps = float(check.get("expected_fps", 30.0))
    expected_duration = float(check.get("expected_duration_seconds", 30.0))
    duration_tolerance = float(check.get("duration_tolerance_seconds", 1.0))
    fps_tolerance = float(check.get("fps_tolerance", 0.5))
    signal_rate = float(check.get("signal_sample_rate", 60.0))
    decode_all = bool(check.get("decode_all_frames", True))

    out = ensure_dir(Path(args.output_root) if args.output_root else resolve_mode_root(config, "diagnostics") / "PURE_AVI_CHECK")
    rows = []
    pure_cfg = config["datasets"]["PURE"]
    for source in pure_cfg.get("sources", []):
        label = str(source.get("label", "PURE"))
        motion = str(source.get("motion_type", ""))
        video_root = Path(str(source.get("video_DIR", "")))
        signal_root = Path(str(source.get("BVP_DIR", video_root)))
        roots = unique_paths([video_root, video_root / "PURE", signal_root, signal_root / "PURE"])
        json_files = []
        for root in roots:
            if root.exists():
                json_files.extend(root.rglob("*.json"))
        json_files = sorted(unique_paths(json_files), key=natural_key)
        for json_file in json_files:
            stem = json_file.stem
            if "-" not in stem or not re.match(r"^\d+[-_]\d+$", stem):
                continue
            video = find_video(json_file, roots)
            row = {
                "source_label": label,
                "motion_type": motion,
                "sample_id": stem,
                "sample_uid": f"PURE/{label}/{stem}",
                "json_path": str(json_file),
                "avi_path": str(video) if video else "",
                "avi_exists": bool(video),
                "avi_file_size_bytes": video.stat().st_size if video else 0,
                "png_frame_count": count_pngs(json_file, roots),
                "json_duration_seconds": json_duration(json_file, signal_rate),
            }
            if video:
                row.update(inspect_video(video, decode_all))
            else:
                row.update({
                    "avi_opened": False, "fps": 0.0, "reported_frame_count": 0,
                    "decoded_frame_count": 0, "decode_failure_count": 0,
                    "video_duration_seconds": 0.0, "reported_minus_decoded": 0,
                })
            row["expected_frame_count"] = int(round(expected_fps * expected_duration))
            row["fps_ok"] = bool(row["avi_opened"] and abs(row["fps"] - expected_fps) <= fps_tolerance)
            row["duration_ok"] = bool(row["avi_opened"] and row["video_duration_seconds"] >= expected_duration - duration_tolerance)
            json_duration_value = row["json_duration_seconds"]
            row["duration_difference_vs_json"] = (
                row["video_duration_seconds"] - json_duration_value
                if pd.notna(json_duration_value) else float("nan")
            )
            row["json_duration_match"] = bool(
                pd.notna(json_duration_value)
                and abs(row["duration_difference_vs_json"]) <= duration_tolerance
            )
            issues = []
            if not row["avi_exists"]:
                issues.append("missing_avi")
            elif not row["avi_opened"]:
                issues.append("unreadable_avi")
            else:
                if not row["fps_ok"]:
                    issues.append("unexpected_fps")
                if not row["duration_ok"]:
                    issues.append("insufficient_duration")
                if row["reported_minus_decoded"] != 0:
                    issues.append("reported_decoded_mismatch")
                if row["decode_failure_count"] > 0:
                    issues.append("decode_failures")
                if pd.notna(json_duration_value) and not row["json_duration_match"]:
                    issues.append("json_duration_mismatch")
            row["status"] = "ok" if not issues else "warning"
            row["issues"] = ";".join(issues)
            rows.append(row)
            print(f"[CHECK] {row['sample_uid']} status={row['status']} issues={row['issues'] or '-'}")

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "pure_avi_integrity.csv", index=False, encoding="utf-8-sig")
    subsets = {
        "missing_avi_samples.csv": frame[frame["issues"].str.contains("missing_avi", na=False)],
        "unreadable_avi_samples.csv": frame[frame["issues"].str.contains("unreadable_avi", na=False)],
        "duration_mismatch_samples.csv": frame[frame["issues"].str.contains("duration", na=False)],
    }
    for name, subset in subsets.items():
        subset.to_csv(out / name, index=False, encoding="utf-8-sig")
    summary = {
        "sample_count": int(len(frame)),
        "ok_count": int((frame["status"] == "ok").sum()) if not frame.empty else 0,
        "warning_count": int((frame["status"] != "ok").sum()) if not frame.empty else 0,
        "missing_avi_count": int(frame["issues"].str.contains("missing_avi", na=False).sum()) if not frame.empty else 0,
        "unreadable_avi_count": int(frame["issues"].str.contains("unreadable_avi", na=False).sum()) if not frame.empty else 0,
        "insufficient_duration_count": int(frame["issues"].str.contains("insufficient_duration", na=False).sum()) if not frame.empty else 0,
    }
    (out / "pure_avi_integrity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[INFO] PURE AVI integrity report: {out}")


if __name__ == "__main__":
    main()
