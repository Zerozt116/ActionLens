from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = PROJECT_ROOT / "external" / "actionformer_release"
TAL_DIR = PROJECT_ROOT / "scripts" / "tal"
if str(EXTERNAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_ROOT))
if str(TAL_DIR) not in sys.path:
    sys.path.insert(0, str(TAL_DIR))

import nms_1d_cpu  # noqa: E402

sys.modules["nms_1d_cpu"] = nms_1d_cpu

from libs.datasets import make_dataset  # noqa: E402


class ActionFormerCharadesDatasetTests(unittest.TestCase):
    def test_make_dataset_returns_clip_local_actionformer_item(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            feature_path = tmp_path / "features" / "VID_c106_t5.00_10.00.npz"
            feature_path.parent.mkdir()
            np.savez_compressed(feature_path, feats=np.ones((10, 4), dtype=np.float32))

            clips_manifest = tmp_path / "clips.csv"
            write_clips_manifest(clips_manifest)
            annotations_path = tmp_path / "annotations.json"
            write_annotations(annotations_path)
            feature_manifest = tmp_path / "features.json"
            write_feature_manifest(feature_manifest, feature_path)

            dataset = make_dataset(
                "charades",
                False,
                ["all"],
                feat_folder=str(feature_path.parent),
                json_file=str(annotations_path),
                clips_manifest=str(clips_manifest),
                feature_manifest=str(feature_manifest),
                feat_stride=15,
                num_frames=1,
                default_fps=30,
                downsample_rate=1,
                max_seq_len=16,
                trunc_thresh=0.5,
                crop_ratio=None,
                input_dim=4,
                num_classes=3,
                file_prefix=None,
                file_ext=".npz",
                force_upsampling=False,
            )
            item = dataset[0]

        self.assertEqual(len(dataset), 1)
        self.assertEqual(item["video_id"], "VID_c106_t5.00_10.00")
        self.assertEqual(list(item["feats"].shape), [4, 10])
        self.assertEqual(list(item["segments"].shape), [1, 2])
        self.assertEqual(item["labels"].tolist(), [2])
        self.assertAlmostEqual(float(item["segments"][0][0]), 3.9666667, places=4)
        self.assertAlmostEqual(float(item["segments"][0][1]), 7.9666667, places=4)
        self.assertEqual(dataset.get_attributes()["dataset_name"], "charades")

    def test_make_dataset_filters_by_split_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            feature_dir = tmp_path / "features"
            feature_dir.mkdir()
            first_feature = feature_dir / "VID_c106_t5.00_10.00.npz"
            second_feature = feature_dir / "OTHER_c106_t5.00_10.00.npz"
            np.savez_compressed(first_feature, feats=np.ones((10, 4), dtype=np.float32))
            np.savez_compressed(second_feature, feats=np.ones((10, 4), dtype=np.float32))

            clips_manifest = tmp_path / "clips.csv"
            write_split_clips_manifest(clips_manifest)
            annotations_path = tmp_path / "annotations.json"
            write_split_annotations(annotations_path)
            feature_manifest = tmp_path / "features.json"
            write_split_feature_manifest(feature_manifest, first_feature, second_feature)
            split_folder = tmp_path / "splits"
            split_folder.mkdir()
            (split_folder / "train.txt").write_text("VID\n", encoding="utf-8")

            dataset = make_dataset(
                "charades",
                False,
                ["train"],
                feat_folder=str(feature_dir),
                json_file=str(annotations_path),
                clips_manifest=str(clips_manifest),
                feature_manifest=str(feature_manifest),
                split_folder=str(split_folder),
                feat_stride=15,
                num_frames=1,
                default_fps=30,
                downsample_rate=1,
                max_seq_len=16,
                trunc_thresh=0.5,
                crop_ratio=None,
                input_dim=4,
                num_classes=3,
                file_prefix=None,
                file_ext=".npz",
                force_upsampling=False,
            )
            item = dataset[0]

        self.assertEqual(len(dataset), 1)
        self.assertEqual(item["video_id"], "VID_c106_t5.00_10.00")

    def test_get_eval_json_file_exports_clip_level_ground_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            feature_path = tmp_path / "features" / "VID_c106_t5.00_10.00.npz"
            feature_path.parent.mkdir()
            np.savez_compressed(feature_path, feats=np.ones((10, 4), dtype=np.float32))

            clips_manifest = tmp_path / "clips.csv"
            write_clips_manifest(clips_manifest)
            annotations_path = tmp_path / "annotations.json"
            write_annotations(annotations_path)
            feature_manifest = tmp_path / "features.json"
            write_feature_manifest(feature_manifest, feature_path)

            dataset = make_dataset(
                "charades",
                False,
                ["val"],
                feat_folder=str(feature_path.parent),
                json_file=str(annotations_path),
                clips_manifest=str(clips_manifest),
                feature_manifest=str(feature_manifest),
                split_folder=None,
                feat_stride=15,
                num_frames=1,
                default_fps=30,
                downsample_rate=1,
                max_seq_len=16,
                trunc_thresh=0.5,
                crop_ratio=None,
                input_dim=4,
                num_classes=3,
                file_prefix=None,
                file_ext=".npz",
                force_upsampling=False,
            )
            eval_json = Path(dataset.get_eval_json_file())
            payload = json.loads(eval_json.read_text(encoding="utf-8"))

        self.assertTrue(eval_json.name.endswith("charades_clip_eval_val.json"))
        self.assertIn("VID_c106_t5.00_10.00", payload["database"])
        item = payload["database"]["VID_c106_t5.00_10.00"]
        self.assertEqual(item["subset"], "val")
        self.assertEqual(item["annotations"][0]["label_id"], 2)
        self.assertEqual(item["annotations"][0]["segment"], [2.0, 4.0])


def write_clips_manifest(path: Path) -> None:
    fieldnames = [
        "clip_id",
        "video_id",
        "action_id",
        "action_name",
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
                "clip_start_seconds": "5.0",
                "clip_end_seconds": "10.0",
                "status": "sliced",
                "clip_path": str(path.with_suffix(".mp4")),
            }
        )


def write_annotations(path: Path) -> None:
    payload = {
        "database": {
            "VID": {
                "duration": 12.0,
                "fps": 30.0,
                "annotations": [
                    {"segment": [7.0, 9.0], "label_id": 2, "label": "drink"}
                ],
            }
        },
        "label_dict": {"drink": 2},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_feature_manifest(path: Path, feature_path: Path) -> None:
    payload = {
        "database": {
            "VID_c106_t5.00_10.00": {
                "feature_path": str(feature_path),
                "source_fps": 30.0,
                "feature_fps": 2.0,
                "duration_seconds": 5.0,
            }
        }
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_split_clips_manifest(path: Path) -> None:
    fieldnames = [
        "clip_id",
        "video_id",
        "action_id",
        "action_name",
        "clip_start_seconds",
        "clip_end_seconds",
        "status",
        "clip_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for video_id in ["VID", "OTHER"]:
            writer.writerow(
                {
                    "clip_id": f"{video_id}_c106_t5.00_10.00",
                    "video_id": video_id,
                    "action_id": "c106",
                    "action_name": "drink",
                    "clip_start_seconds": "5.0",
                    "clip_end_seconds": "10.0",
                    "status": "sliced",
                    "clip_path": str(path.with_suffix(".mp4")),
                }
            )


def write_split_annotations(path: Path) -> None:
    payload = {
        "database": {
            video_id: {
                "duration": 12.0,
                "fps": 30.0,
                "annotations": [
                    {"segment": [7.0, 9.0], "label_id": 2, "label": "drink"}
                ],
            }
            for video_id in ["VID", "OTHER"]
        },
        "label_dict": {"drink": 2},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_split_feature_manifest(path: Path, first_feature: Path, second_feature: Path) -> None:
    payload = {
        "database": {
            "VID_c106_t5.00_10.00": {
                "feature_path": str(first_feature),
                "source_fps": 30.0,
                "feature_fps": 2.0,
                "duration_seconds": 5.0,
            },
            "OTHER_c106_t5.00_10.00": {
                "feature_path": str(second_feature),
                "source_fps": 30.0,
                "feature_fps": 2.0,
                "duration_seconds": 5.0,
            },
        }
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
