from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_ava_alignment import bbox_iou, find_target, parse_clip_name


class EvaluateAvaAlignmentTests(unittest.TestCase):
    def test_parse_clip_name(self) -> None:
        parsed = parse_clip_name(Path("5BDj0ow5hnA_t0997_p114_a15_answer_phone.mp4"))

        self.assertEqual(parsed["video_id"], "5BDj0ow5hnA")
        self.assertEqual(parsed["timestamp"], 997)
        self.assertEqual(parsed["person_id"], 114)
        self.assertEqual(parsed["action_id"], 15)

    def test_bbox_iou(self) -> None:
        self.assertAlmostEqual(bbox_iou([0, 0, 10, 10], [5, 5, 15, 15]), 25 / 175)
        self.assertEqual(bbox_iou([0, 0, 1, 1], [2, 2, 3, 3]), 0.0)

    def test_find_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["video_id", "timestamp", "x1", "y1", "x2", "y2", "action_id", "action", "action_name", "action_type", "person_id"])
                writer.writerow(["abc123", 902, 0.1, 0.2, 0.3, 0.4, 27, "drink", "drink", "OBJECT_MANIPULATION", 5])

            target = find_target(manifest, Path("abc123_t0902_p5_a27_drink.mp4"))

        self.assertEqual(target.video_id, "abc123")
        self.assertEqual(target.action_id, 27)
        self.assertEqual(target.bbox_xyxy_norm, [0.1, 0.2, 0.3, 0.4])


if __name__ == "__main__":
    unittest.main()
