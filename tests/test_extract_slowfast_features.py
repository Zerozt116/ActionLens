from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.tal.extract_slowfast_features import (
    build_slowfast_pathways,
    extract_features_for_manifest,
    flatten_slowfast_feature,
    frame_to_smoke_feature,
    make_centered_window,
    sample_frame_indices,
)


class ExtractSlowfastFeaturesTests(unittest.TestCase):
    def test_sample_frame_indices_respects_max_frames(self) -> None:
        indices = sample_frame_indices(frame_count=100, source_fps=25.0, sample_fps=5.0, max_frames=8)
        self.assertLessEqual(len(indices), 8)
        self.assertEqual(indices, sorted(indices))
        self.assertEqual(indices[0], 0)
        self.assertLess(indices[-1], 100)

    def test_frame_to_smoke_feature_shape(self) -> None:
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        feature = frame_to_smoke_feature(frame)
        self.assertEqual(feature.shape, (2304,))
        self.assertEqual(feature.dtype, np.float32)

    def test_extract_features_for_manifest_writes_npz_and_manifest_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video = tmp_path / "clip.mp4"
            write_video(video)
            clips_csv = tmp_path / "clips.csv"
            write_clips_csv(clips_csv, video)

            records = extract_features_for_manifest(
                clips_manifest=clips_csv,
                output_dir=tmp_path / "features",
                backend="cv2-smoke",
                sample_fps=2.0,
                max_frames=5,
                offset=0,
                limit=None,
                overwrite=False,
            )

            feature_path = Path(records[0].feature_path)
            with np.load(feature_path) as data:
                feats = data["feats"]

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].status, "extracted")
        self.assertEqual(feats.shape[1], 2304)
        self.assertLessEqual(feats.shape[0], 5)

    def test_extract_features_reuses_existing_npz(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video = tmp_path / "clip.mp4"
            write_video(video)
            clips_csv = tmp_path / "clips.csv"
            write_clips_csv(clips_csv, video)

            first = extract_features_for_manifest(
                clips_manifest=clips_csv,
                output_dir=tmp_path / "features",
                backend="cv2-smoke",
                sample_fps=2.0,
                max_frames=5,
                offset=0,
                limit=None,
                overwrite=False,
            )
            second = extract_features_for_manifest(
                clips_manifest=clips_csv,
                output_dir=tmp_path / "features",
                backend="cv2-smoke",
                sample_fps=2.0,
                max_frames=5,
                offset=0,
                limit=None,
                overwrite=False,
            )

        self.assertEqual(first[0].status, "extracted")
        self.assertEqual(second[0].status, "exists")

    def test_make_centered_window_clamps_edges(self) -> None:
        frames = np.zeros((4, 8, 8, 3), dtype=np.uint8)
        for index in range(4):
            frames[index] = index
        window = make_centered_window(frames, center_index=0, clip_frames=5)
        self.assertEqual(list(window[:, 0, 0, 0]), [0, 0, 0, 1, 2])

    def test_build_slowfast_pathways_shapes(self) -> None:
        window = np.zeros((32, 16, 16, 3), dtype=np.uint8)
        slow, fast = build_slowfast_pathways(window, alpha=4, crop_size=32)
        self.assertEqual(list(slow.shape), [3, 8, 32, 32])
        self.assertEqual(list(fast.shape), [3, 32, 32, 32])

    def test_flatten_slowfast_feature_accepts_projection_input_layout(self) -> None:
        import torch

        feature = torch.ones((1, 1, 1, 1, 2304), dtype=torch.float32)
        vector = flatten_slowfast_feature(feature)
        self.assertEqual(vector.shape, (2304,))


def write_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
    if not writer.isOpened():
        raise RuntimeError("Could not open test video writer")
    try:
        for index in range(12):
            frame = np.full((24, 32, 3), index * 10, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def write_clips_csv(path: Path, video: Path) -> None:
    fieldnames = ["clip_id", "video_id", "action_id", "action_name", "status", "clip_path"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "clip_id": "VID_c106_t0.00_1.00",
                "video_id": "VID",
                "action_id": "c106",
                "action_name": "drink",
                "status": "sliced",
                "clip_path": str(video),
            }
        )


if __name__ == "__main__":
    unittest.main()
