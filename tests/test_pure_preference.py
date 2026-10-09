import json
from pathlib import Path

import pytest

pytest.importorskip("pyVHR")
from gamma_rppg.datasets.pure import PUREDirectoryDataset


def test_avi_preferred_over_frames(tmp_path: Path):
    trial = "01-01"
    (tmp_path / f"{trial}.json").write_text(json.dumps({"/FullPackage": []}), encoding="utf-8")
    frame_dir = tmp_path / trial
    frame_dir.mkdir()
    (frame_dir / "Image0001.png").write_bytes(b"frame")
    (tmp_path / f"{trial}.avi").write_bytes(b"avi")
    dataset = PUREDirectoryDataset(
        str(tmp_path), str(tmp_path), prefer_avi=True, allow_frames_fallback=False
    )
    assert Path(dataset.videoFilenames[0]).suffix.lower() == ".avi"
    assert dataset.getSampleMetadata(0)["input_type"] == "video_file"


def test_no_png_fallback_in_formal_mode(tmp_path: Path):
    trial = "01-01"
    (tmp_path / f"{trial}.json").write_text(json.dumps({"/FullPackage": []}), encoding="utf-8")
    frame_dir = tmp_path / trial
    frame_dir.mkdir()
    (frame_dir / "Image0001.png").write_bytes(b"frame")
    with pytest.raises(FileNotFoundError):
        PUREDirectoryDataset(
            str(tmp_path), str(tmp_path), prefer_avi=True, allow_frames_fallback=False
        )
