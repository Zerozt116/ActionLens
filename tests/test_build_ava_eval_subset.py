from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.build_ava_eval_subset import build_subset, build_summary


class BuildAvaEvalSubsetTests(unittest.TestCase):
    def test_build_subset_balances_actions_and_limits_video_repeats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["video_id", "timestamp", "x1", "y1", "x2", "y2", "action_id", "action", "action_name", "action_type", "person_id"])
                for timestamp in range(10):
                    writer.writerow(["v1", timestamp, 0, 0, 1, 1, 12, "stand", "stand", "PERSON_MOVEMENT", timestamp])
                for timestamp in range(10):
                    writer.writerow(["v2", timestamp, 0, 0, 1, 1, 12, "stand", "stand", "PERSON_MOVEMENT", timestamp])
                for timestamp in range(10):
                    writer.writerow(["v3", timestamp, 0, 0, 1, 1, 17, "carry_hold", "carry/hold", "OBJECT_MANIPULATION", timestamp])

            rows = build_subset(manifest, {12, 17}, samples_per_action=4, max_per_video_action=2)

        self.assertEqual(len(rows), 6)
        self.assertEqual(sum(row.action_id == 12 for row in rows), 4)
        self.assertEqual(sum(row.action_id == 17 for row in rows), 2)
        self.assertEqual(sum(row.video_id == "v1" and row.action_id == 12 for row in rows), 2)

    def test_build_summary_counts_types_and_videos(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["video_id", "timestamp", "x1", "y1", "x2", "y2", "action_id", "action", "action_name", "action_type", "person_id"])
                writer.writerow(["v1", 1, 0, 0, 1, 1, 12, "stand", "stand", "PERSON_MOVEMENT", 1])
                writer.writerow(["v2", 1, 0, 0, 1, 1, 17, "carry_hold", "carry/hold", "OBJECT_MANIPULATION", 1])
            rows = build_subset(manifest, {12, 17}, samples_per_action=1, max_per_video_action=1)

        summary = build_summary(rows)

        self.assertEqual(summary["num_rows"], 2)
        self.assertEqual(summary["num_videos"], 2)
        self.assertEqual(summary["by_action_type"]["PERSON_MOVEMENT"], 1)


if __name__ == "__main__":
    unittest.main()
