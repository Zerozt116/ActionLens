from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.tal.dataloader_smoke import (
    intersect_clip_annotations,
    seconds_to_feature_grid,
    smoke_load_manifest,
)


class TalDataLoaderSmokeTests(unittest.TestCase):
    def test_intersect_clip_annotations_converts_to_local_segments(self) -> None:
        local, labels = intersect_clip_annotations(
            [
                {"segment": [1.0, 3.0], "label_id": 1},
                {"segment": [8.0, 12.0], "label_id": 2},
                {"segment": [20.0, 22.0], "label_id": 3},
            ],
            clip_start=5.0,
            clip_end=10.0,
        )
        self.assertEqual(local, [[3.0, 5.0]])
        self.assertEqual(labels, [2])

    def test_seconds_to_feature_grid_matches_actionformer_formula(self) -> None:
        grid = seconds_to_feature_grid(
            [[2.0, 4.0]],
            source_fps=30.0,
            feat_stride=15.0,
            feat_offset=0.5 / 15.0,
        )
        self.assertEqual(grid, [[3.9667, 7.9667]])

    def test_smoke_load_manifest_builds_data_dict_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            feature_path = tmp_path / "features" / "VID_c106_t5.00_10.00.npz"
            feature_path.parent.mkdir()
            np.savez_compressed(feature_path, feats=np.ones((10, 2304), dtype=np.float32))
            clips_manifest = tmp_path / "clips.csv"
            write_clips_manifest(clips_manifest, feature_path)
            annotations_path = tmp_path / "annotations.json"
            write_annotations(annotations_path)
            feature_manifest = tmp_path / "features.json"
            write_feature_manifest(feature_manifest, feature_path)

            records = smoke_load_manifest(
                clips_manifest=clips_manifest,
                annotations_path=annotations_path,
                feature_manifest_path=feature_manifest,
                feat_num_frames=1,
                downsample_rate=1,
                offset=0,
                limit=None,
            )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].status, "ok")
        self.assertEqual(records[0].feats_shape, [2304, 10])
        self.assertEqual(records[0].segments_shape, [1, 2])
        self.assertEqual(records[0].labels_shape, [1])
        self.assertEqual(records[0].local_segments_seconds, [[2.0, 4.0]])


def write_clips_manifest(path: Path, feature_path: Path) -> None:
    fieldnames = [
        "clip_id",
        "video_id",
        "action_id",
        "action_name",
        "start_seconds",
        "end_seconds",
        "clip_start_seconds",
        "clip_end_seconds",
        "status",
        "clip_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "clip_id": "VID_c106_t5.00_10.00",
                "video_id": "VID",
                "action_id": "c106",
                "action_name": "drink",
                "start_seconds": "7.0",
                "end_seconds": "9.0",
                "clip_start_seconds": "5.0",
                "clip_end_seconds": "10.0",
                "status": "sliced",
                "clip_path": str(feature_path.with_suffix(".mp4")),
            }
        )


def write_annotations(path: Path) -> None:
    payload = {
        "database": {
            "VID": {
                "duration": 12.0,
                "fps": 30.0,
                "annotations": [{"segment": [7.0, 9.0], "label_id": 4, "label": "drink"}],
            }
        }
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_feature_manifest(path: Path, feature_path: Path) -> None:
    payload = {
        "database": {
            "VID_c106_t5.00_10.00": {
                "feature_path": str(feature_path),
                "source_fps": 30.0,
                "feature_fps": 2.0,
            }
        }
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
