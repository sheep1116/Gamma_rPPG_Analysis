from pathlib import Path

import cv2
import numpy as np

from gamma_rppg.diagnostics.avi_integrity import inspect_video


def test_inspect_video_counts_decoded_frames(tmp_path: Path):
    path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24))
    for value in range(5):
        writer.write(np.full((24, 32, 3), value * 20, dtype=np.uint8))
    writer.release()
    result = inspect_video(path, decode_all=True)
    assert result["avi_opened"] is True
    assert result["decoded_frame_count"] == 5
    assert result["reported_minus_decoded"] == 0
