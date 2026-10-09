from __future__ import annotations

import argparse
from pathlib import Path

from gamma_rppg.common.config import load_config
from gamma_rppg.datasets.pure import PUREDirectoryDataset


def main():
    parser = argparse.ArgumentParser(description="Inspect the merged PURE input selection.")
    parser.add_argument("--config", default="configs/full_pre_roi_experiment.yaml")
    parser.add_argument("--root", default=None)
    args = parser.parse_args()

    if args.root:
        dataset = PUREDirectoryDataset(args.root, args.root, prefer_avi=True, allow_frames_fallback=False)
    else:
        config = load_config(args.config)
        pure = config["datasets"]["PURE"]
        dataset = PUREDirectoryDataset(
            sources=pure["sources"],
            prefer_avi=bool(pure.get("prefer_avi", True)),
            allow_frames_fallback=bool(pure.get("allow_frames_fallback", False)),
        )

    print(f"numVideos = {dataset.numVideos}")
    counts, input_counts = {}, {}
    for index in range(dataset.numVideos):
        metadata = dataset.getSampleMetadata(index)
        motion = metadata["motion_type"] or "unspecified"
        counts[motion] = counts.get(motion, 0) + 1
        input_type = metadata["input_type"]
        input_counts[input_type] = input_counts.get(input_type, 0) + 1
    print("motion counts:")
    for motion, count in counts.items():
        print(f"  {motion}: {count}")
    print("input-type counts:")
    for input_type, count in input_counts.items():
        print(f"  {input_type}: {count}")

    for index, (video, signal) in enumerate(zip(dataset.videoFilenames[:30], dataset.sigFilenames[:30])):
        metadata = dataset.getSampleMetadata(index)
        detected_kind = "frames_dir" if Path(video).is_dir() else "video_file"
        print(
            f"[{index}] {detected_kind}: {video}\n"
            f"    signal: {signal}\n"
            f"    uid: {metadata['sample_uid']}\n"
            f"    subject: {metadata['subject_id']}\n"
            f"    source: {metadata['source_label']}\n"
            f"    motion: {metadata['motion_type']}\n"
            f"    sample: {metadata['sample_id']}"
        )


if __name__ == "__main__":
    main()
