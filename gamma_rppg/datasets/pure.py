from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from pyVHR.BPM.BPM import BVPsignal


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}


def natural_key(path_like):
    text = str(path_like)
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", text)]


class PUREDirectoryDataset:
    """Read configured PURE sources as one logical dataset.

    v5 formal experiments prefer AVI/video files. PNG/JPG directories can be
    enabled only as an explicit fallback for diagnostics or legacy experiments.
    The six PURE motion sources remain one dataset and retain motion metadata.
    """

    name = "PURE"
    signalGT = "BVP"
    frameRate = 30.0
    SIG_SampleRate = 60.0

    def __init__(
        self,
        videodataDIR: Optional[str] = None,
        BVPdataDIR: Optional[str] = None,
        sources: Optional[Sequence[Dict]] = None,
        prefer_avi: bool = True,
        allow_frames_fallback: bool = False,
    ):
        if sources:
            self.sources = [dict(source) for source in sources]
        elif videodataDIR:
            self.sources = [{
                "label": "PURE",
                "motion_type": "",
                "video_DIR": str(videodataDIR),
                "BVP_DIR": str(BVPdataDIR or videodataDIR),
            }]
        else:
            raise ValueError("PUREDirectoryDataset requires sources or videodataDIR.")

        self.prefer_avi = bool(prefer_avi)
        self.allow_frames_fallback = bool(allow_frames_fallback)
        self.videoFilenames: List[str] = []
        self.sigFilenames: List[str] = []
        self.motionTypes: List[str] = []
        self.sourceLabels: List[str] = []
        self.sampleIds: List[str] = []
        self.sampleUids: List[str] = []
        self.subjectIds: List[str] = []
        self.trialIds: List[str] = []
        self.inputTypes: List[str] = []
        self.numVideos = 0
        self.loadFilenames()

    @staticmethod
    def _unique_paths(paths: Sequence[Path]) -> List[Path]:
        output: List[Path] = []
        seen = set()
        for path in paths:
            try:
                key = str(path.resolve())
            except Exception:
                key = str(path)
            if key not in seen:
                seen.add(key)
                output.append(path)
        return output

    @classmethod
    def _candidate_roots(cls, raw_dir: Optional[str]) -> List[Path]:
        if not raw_dir:
            return []
        root = Path(raw_dir)
        return cls._unique_paths([root, root / "PURE"])

    @staticmethod
    def _has_image_frames(frame_dir: Path) -> bool:
        if not frame_dir.is_dir():
            return False
        return any(path.is_file() and path.suffix.lower() in _IMAGE_EXTS for path in frame_dir.iterdir())

    @classmethod
    def _find_signal_root(cls, source: Dict) -> Path:
        candidates = cls._candidate_roots(source.get("BVP_DIR") or source.get("video_DIR"))
        for root in candidates:
            if root.exists() and any(root.glob("*.json")):
                return root
        for root in candidates:
            if root.exists() and any(root.rglob("*.json")):
                return root
        checked = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"Cannot find PURE json files for source '{source.get('label', 'PURE')}'. Checked: {checked}"
        )

    @classmethod
    def _video_roots(cls, source: Dict, signal_root: Path) -> List[Path]:
        roots = cls._candidate_roots(source.get("video_DIR"))
        roots.extend([signal_root, signal_root / "PURE"])
        return cls._unique_paths(roots)

    def _resolve_input(self, json_file: Path, video_roots: Sequence[Path]) -> Optional[Tuple[Path, str]]:
        stem = json_file.stem
        frame_dirs: List[Path] = [json_file.parent / stem]
        video_candidates: List[Path] = []

        for ext in sorted(_VIDEO_EXTS):
            video_candidates.append(json_file.parent / f"{stem}{ext}")
        for root in video_roots:
            trial_dir = root / stem
            frame_dirs.append(trial_dir)
            for ext in sorted(_VIDEO_EXTS):
                video_candidates.extend([root / f"{stem}{ext}", trial_dir / f"{stem}{ext}"])
            if trial_dir.exists():
                video_candidates.extend(
                    sorted(
                        [p for p in trial_dir.iterdir() if p.is_file() and p.suffix.lower() in _VIDEO_EXTS],
                        key=natural_key,
                    )
                )

        valid_videos = [
            path for path in self._unique_paths(video_candidates)
            if path.is_file() and path.stat().st_size > 0
        ]
        valid_frames = [path for path in self._unique_paths(frame_dirs) if self._has_image_frames(path)]

        if self.prefer_avi and valid_videos:
            return valid_videos[0], "video_file"
        if self.allow_frames_fallback and valid_frames:
            return valid_frames[0], "frames_dir"
        if not self.prefer_avi and valid_frames:
            return valid_frames[0], "frames_dir"
        if valid_videos:
            return valid_videos[0], "video_file"
        return None

    @staticmethod
    def _subject_id(stem: str) -> str:
        return stem.split("-", 1)[0] if "-" in stem else stem

    def loadFilenames(self) -> None:
        samples = []
        seen_pairs = set()
        for source_index, source in enumerate(self.sources):
            label = str(source.get("label") or f"PURE_source_{source_index + 1}")
            motion_type = str(source.get("motion_type") or "")
            signal_root = self._find_signal_root(source)
            video_roots = self._video_roots(source, signal_root)
            print(f"[INFO] PURE source '{label}': signal root={signal_root}")

            json_files = sorted(signal_root.rglob("*.json"), key=natural_key)
            source_count = 0
            for json_file in json_files:
                stem = json_file.stem
                if not re.match(r"^\d+[-_]\d+$", stem):
                    continue
                resolved = self._resolve_input(json_file, video_roots)
                if resolved is None:
                    required = "AVI/video" if self.prefer_avi and not self.allow_frames_fallback else "allowed input"
                    print(f"[WARN] PURE trial skipped; no {required} for {json_file}")
                    continue
                video_path, input_type = resolved
                pair_key = (str(video_path.resolve()), str(json_file.resolve()))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                subject_id = self._subject_id(stem)
                sample_uid = f"PURE/{label}/{stem}"
                samples.append({
                    "source_index": source_index,
                    "label": label,
                    "motion_type": motion_type,
                    "sample_id": stem,
                    "sample_uid": sample_uid,
                    "subject_id": subject_id,
                    "trial_id": stem,
                    "input_type": input_type,
                    "video": str(video_path),
                    "signal": str(json_file),
                })
                source_count += 1
            print(f"[INFO] PURE source '{label}': found {source_count} usable samples")

        if not samples:
            configured = ", ".join(str(source.get("video_DIR", "")) for source in self.sources)
            raise FileNotFoundError(
                "No usable PURE samples found under the configured input policy. "
                f"prefer_avi={self.prefer_avi}, allow_frames_fallback={self.allow_frames_fallback}. "
                f"Configured sources: {configured}"
            )

        samples.sort(key=lambda item: (item["source_index"], natural_key(item["sample_id"])))
        self.videoFilenames = [item["video"] for item in samples]
        self.sigFilenames = [item["signal"] for item in samples]
        self.motionTypes = [item["motion_type"] for item in samples]
        self.sourceLabels = [item["label"] for item in samples]
        self.sampleIds = [item["sample_id"] for item in samples]
        self.sampleUids = [item["sample_uid"] for item in samples]
        self.subjectIds = [item["subject_id"] for item in samples]
        self.trialIds = [item["trial_id"] for item in samples]
        self.inputTypes = [item["input_type"] for item in samples]
        self.numVideos = len(samples)

        print(f"[INFO] PURE merged dataset: {self.numVideos} samples from {len(self.sources)} sources")
        for index, item in enumerate(samples[:10]):
            print(
                f"[INFO]   PURE[{index}] source={item['label']} motion={item['motion_type']} "
                f"sample={item['sample_id']} input={item['input_type']} path={item['video']}"
            )

    def getVideoFilename(self, videoIdx: int = 0) -> str:
        return self.videoFilenames[videoIdx]

    def getSigFilename(self, videoIdx: int = 0) -> str:
        return self.sigFilenames[videoIdx]

    def getSampleMetadata(self, videoIdx: int = 0) -> Dict[str, str]:
        return {
            "motion_type": self.motionTypes[videoIdx],
            "source_label": self.sourceLabels[videoIdx],
            "sample_id": self.sampleIds[videoIdx],
            "sample_uid": self.sampleUids[videoIdx],
            "subject_id": self.subjectIds[videoIdx],
            "trial_id": self.trialIds[videoIdx],
            "input_type": self.inputTypes[videoIdx],
            "input_path": self.videoFilenames[videoIdx],
        }

    def readSigfile(self, filename: str):
        with open(filename, "r", encoding="utf-8") as json_file:
            json_data = json.load(json_file)
        bvp = [entry["Value"]["waveform"] for entry in json_data["/FullPackage"]]
        return BVPsignal(np.asarray(bvp, dtype=np.float32), self.SIG_SampleRate)
