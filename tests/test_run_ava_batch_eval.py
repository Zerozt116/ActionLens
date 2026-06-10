from __future__ import annotations

import unittest

from scripts.run_ava_batch_eval import BatchEvalRecord, summarize_records


class RunAvaBatchEvalTests(unittest.TestCase):
    def test_summarize_records(self) -> None:
        records = [
            BatchEvalRecord("a.mp4", "out/a", "v1", 1, 1, 12, "stand", "stand", "PERSON_MOVEMENT", 0.8, True, True, 0, 0, "ok"),
            BatchEvalRecord("b.mp4", "out/b", "v2", 1, 1, 27, "drink", "drink", "OBJECT_MANIPULATION", 0.2, False, False, 1, 1, "ok"),
            BatchEvalRecord("", "out/c", "v3", 1, 1, 14, "walk", "walk", "PERSON_MOVEMENT", 0.0, False, False, 0, 0, "download_failed"),
        ]

        summary = summarize_records(records)

        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["ok"], 2)
        self.assertEqual(summary["download_failed"], 1)
        self.assertEqual(summary["match_rate"], 0.5)
        self.assertEqual(summary["average_best_iou"], 0.5)
        self.assertEqual(summary["pose_on_matched_rate"], 1.0)
        self.assertEqual(summary["object_detected_rate"], 0.5)
        self.assertEqual(summary["event_detected_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
