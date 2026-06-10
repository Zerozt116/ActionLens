from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_charades_stage2_vlm import canonicalize_action, compare_video, fuse_action, load_charades_evidence, load_stage2_evidence, load_vlm_evidence


class CompareCharadesStage2VlmTests(unittest.TestCase):
    def test_load_charades_evidence_maps_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.csv"
            write_manifest(manifest, "VID01")

            evidence = load_charades_evidence(manifest, "VID01")

        self.assertIn("drinking_water", evidence)
        self.assertIn("holding_phone", evidence)
        self.assertEqual(evidence["drinking_water"].action_ids, ["c106"])

    def test_compare_video_identifies_vlm_recovered_label_and_false_positive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest = tmp_path / "manifest.csv"
            stage2_dir = tmp_path / "stage2"
            stage2_dir.mkdir()
            vlm_review = tmp_path / "vlm_review.json"
            write_manifest(manifest, "VID01")
            (stage2_dir / "events.json").write_text(
                json.dumps(
                    [
                        {
                            "action": "talking_on_phone",
                            "confidence": 0.64,
                            "start_seconds": 1.0,
                            "end_seconds": 2.0,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            vlm_review.write_text(
                json.dumps(
                    {
                        "actions": [
                            {"action": "drinking_water", "present": True, "confidence": 0.9, "evidence": "cup", "supporting_frame_indices": [1]},
                            {"action": "talking_on_phone", "present": False, "confidence": 0.0, "evidence": "no phone call", "supporting_frame_indices": []},
                            {"action": "holding_phone", "present": True, "confidence": 0.8, "evidence": "phone", "supporting_frame_indices": [2]},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            rows = compare_video(manifest, "VID01", stage2_dir, vlm_review)

        by_action = {row.action: row for row in rows}
        self.assertEqual(by_action["drinking_water"].fused_status, "vlm_recovered_label")
        self.assertEqual(by_action["talking_on_phone"].fused_status, "possible_false_positive")
        self.assertEqual(by_action["holding_phone"].fused_status, "vlm_recovered_label")

    def test_fuse_action_cases(self) -> None:
        self.assertEqual(fuse_action(True, True, True, 0.7, 0.9)[0], "confirmed_event")
        self.assertEqual(fuse_action(False, True, False, 0.7, 0.0)[0], "possible_false_positive")
        self.assertEqual(fuse_action(True, False, False, 0.0, 0.0)[0], "missed_label")
        self.assertEqual(fuse_action(False, False, False, 0.0, 0.0)[0], "not_present")

    def test_load_stage2_evidence_preserves_index_and_offsets_clip_times(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            events_path = Path(tmp) / "events.json"
            events_path.write_text(
                json.dumps(
                    [
                        {
                            "action": "drinking_water",
                            "confidence": 0.7,
                            "start_seconds": 1.0,
                            "end_seconds": 2.5,
                            "start_time": "00:00:01.000",
                            "end_time": "00:00:02.500",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            evidence = load_stage2_evidence(events_path, time_offset_seconds=10.0)

        event = evidence["drinking_water"].events[0]
        self.assertEqual(event["stage2_event_index"], 0)
        self.assertEqual(event["clip_start_seconds"], 1.0)
        self.assertEqual(event["clip_end_seconds"], 2.5)
        self.assertEqual(event["start_seconds"], 11.0)
        self.assertEqual(event["end_seconds"], 12.5)
        self.assertEqual(event["time_coordinate"], "source_video")

    def test_canonicalize_extended_charades_actions(self) -> None:
        self.assertEqual(canonicalize_action("c156"), "eating")
        self.assertEqual(canonicalize_action("c052"), "using_laptop")
        self.assertEqual(canonicalize_action("custom_action"), "custom_action")

    def test_load_vlm_evidence_maps_raw_charades_action_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vlm_review = Path(tmp) / "vlm_review.json"
            vlm_review.write_text(
                json.dumps(
                    {
                        "actions": [
                            {
                                "action": "c156",
                                "present": True,
                                "confidence": 0.8,
                                "evidence": "food",
                                "supporting_frame_indices": [0],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            evidence = load_vlm_evidence(vlm_review)

        self.assertIn("eating", evidence)
        self.assertTrue(evidence["eating"].present)


def write_manifest(path: Path, video_id: str) -> None:
    fieldnames = [
        "video_id",
        "action_id",
        "action_name",
        "start_seconds",
        "end_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"video_id": video_id, "action_id": "c106", "action_name": "Drinking from a cup/glass/bottle", "start_seconds": 1, "end_seconds": 5})
        writer.writerow({"video_id": video_id, "action_id": "c015", "action_name": "Holding a phone/camera", "start_seconds": 0, "end_seconds": 6})


if __name__ == "__main__":
    unittest.main()
