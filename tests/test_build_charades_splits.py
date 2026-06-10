from __future__ import annotations

import tempfile
import unittest
import csv
from pathlib import Path

from scripts.tal.build_charades_splits import build_splits, read_video_ids_from_clips, write_splits


class BuildCharadesSplitsTests(unittest.TestCase):
    def test_build_splits_is_deterministic_and_keeps_must_test(self) -> None:
        video_ids = [f"V{i:02d}" for i in range(10)] + ["OINMN"]
        first = build_splits(
            video_ids,
            seed=7,
            train_ratio=0.6,
            val_ratio=0.2,
            must_test={"OINMN"},
        )
        second = build_splits(
            video_ids,
            seed=7,
            train_ratio=0.6,
            val_ratio=0.2,
            must_test={"OINMN"},
        )
        self.assertEqual(first, second)
        self.assertIn("OINMN", first["test"])
        self.assertEqual(len(set(first["train"] + first["val"] + first["test"])), len(video_ids))

    def test_write_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            write_splits({"train": ["A"], "val": ["B"], "test": ["C"]}, output_dir)
            train = (output_dir / "train.txt").read_text(encoding="utf-8")

        self.assertEqual(train, "A\n")

    def test_read_video_ids_from_clips_filters_eligible_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clips.csv"
            with path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["video_id", "status"])
                writer.writeheader()
                writer.writerow({"video_id": "A", "status": "sliced"})
                writer.writerow({"video_id": "B", "status": "missing_source"})
                writer.writerow({"video_id": "A", "status": "exists"})
            video_ids = read_video_ids_from_clips(path)

        self.assertEqual(video_ids, ["A"])


if __name__ == "__main__":
    unittest.main()
