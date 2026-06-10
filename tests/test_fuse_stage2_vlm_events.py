from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.fuse_stage2_vlm_events import fuse_payload, load_event_review


class FuseStage2VlmEventsTests(unittest.TestCase):
    def test_event_center_review_confirms_and_rejects_stage2_events(self) -> None:
        payload = fuse_payload(
            comparison_payload(),
            event_reviews=[
                {
                    "event_index": 0,
                    "action": "drinking_water",
                    "present": True,
                    "confidence": 0.7,
                    "evidence": "can to mouth",
                    "supporting_frame_indices": [0, 1],
                },
                {
                    "event_index": 1,
                    "action": "talking_on_phone",
                    "present": False,
                    "confidence": 0.0,
                    "evidence": "not a phone",
                    "supporting_frame_indices": [],
                },
            ],
        )

        self.assertEqual(payload["summary"]["final_events"], 1)
        self.assertEqual(payload["summary"]["rejected_events"], 1)
        self.assertEqual(payload["final_events"][0]["action"], "drinking_water")
        self.assertEqual(payload["final_events"][0]["decision"], "confirmed_event")
        self.assertEqual(payload["rejected_events"][0]["action"], "talking_on_phone")
        self.assertEqual(payload["rejected_events"][0]["decision"], "rejected_event")

    def test_vlm_recovered_label_becomes_semantic_candidate(self) -> None:
        payload = fuse_payload(comparison_payload(), event_reviews=[])

        actions = {item["action"] for item in payload["semantic_candidates"]}
        self.assertIn("holding_phone", actions)
        self.assertIn("looking_at_phone", actions)
        holding = next(item for item in payload["semantic_candidates"] if item["action"] == "holding_phone")
        self.assertEqual(holding["start_seconds"], 0.0)
        self.assertEqual(holding["end_seconds"], 46.0)
        self.assertEqual(holding["evidence_sources"], ["charades", "full_video_vlm"])

    def test_event_review_uses_stable_stage2_event_index(self) -> None:
        comparison = comparison_payload()
        comparison["comparisons"][0]["stage2"]["events"][0]["stage2_event_index"] = 1
        comparison["comparisons"][1]["stage2"]["events"][0]["stage2_event_index"] = 0

        payload = fuse_payload(
            comparison,
            event_reviews=[
                {
                    "event_index": 0,
                    "action": "talking_on_phone",
                    "present": False,
                    "confidence": 0.0,
                    "evidence": "not a phone",
                    "supporting_frame_indices": [],
                },
                {
                    "event_index": 1,
                    "action": "drinking_water",
                    "present": True,
                    "confidence": 0.7,
                    "evidence": "can to mouth",
                    "supporting_frame_indices": [0],
                },
            ],
        )

        self.assertEqual(payload["final_events"][0]["action"], "drinking_water")
        self.assertEqual(payload["final_events"][0]["stage2_event_index"], 1)
        self.assertEqual(payload["rejected_events"][0]["action"], "talking_on_phone")
        self.assertEqual(payload["rejected_events"][0]["stage2_event_index"], 0)

    def test_load_event_review_uses_summary_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            summary_path = tmp_path / "vlm_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "review_window": {
                            "source": "event:3",
                            "start_seconds": 1.0,
                            "end_seconds": 2.0,
                            "event": {"action": "drinking_water"},
                        },
                        "review_json": {
                            "actions": [
                                {
                                    "action": "drinking_water",
                                    "present": True,
                                    "confidence": 0.8,
                                    "evidence": "cup",
                                    "supporting_frame_indices": [1],
                                }
                            ],
                            "visible_objects": ["cup"],
                        },
                    }
                ),
                encoding="utf-8",
            )

            review = load_event_review(tmp_path)

        self.assertEqual(review["event_index"], 3)
        self.assertTrue(review["present"])
        self.assertEqual(review["review_window"]["start_seconds"], 1.0)


def comparison_payload() -> dict:
    return {
        "video_id": "VID01",
        "comparisons": [
            {
                "action": "drinking_water",
                "action_name": "喝水",
                "charades_present": True,
                "charades": {"action_ids": ["c106"], "action_names": ["drink"], "intervals": [{"start_seconds": 1.0, "end_seconds": 5.0}]},
                "stage2_present": True,
                "stage2": {
                    "events": [
                        {
                            "person_id": 1,
                            "action": "drinking_water",
                            "action_name": "喝水",
                            "start_seconds": 2.0,
                            "end_seconds": 3.0,
                            "confidence": 0.72,
                            "evidence": "cup close to head",
                        }
                    ]
                },
                "vlm": {"confidence": 0.0},
                "fused_status": "needs_temporal_review",
            },
            {
                "action": "talking_on_phone",
                "action_name": "打电话",
                "charades_present": False,
                "charades": {"action_ids": [], "action_names": [], "intervals": []},
                "stage2_present": True,
                "stage2": {
                    "events": [
                        {
                            "person_id": 1,
                            "action": "talking_on_phone",
                            "action_name": "打电话",
                            "start_seconds": 3.0,
                            "end_seconds": 4.0,
                            "confidence": 0.64,
                            "evidence": "phone close to head",
                        }
                    ]
                },
                "vlm": {"confidence": 0.0},
                "fused_status": "possible_false_positive",
            },
            {
                "action": "holding_phone",
                "action_name": "拿手机",
                "charades_present": True,
                "charades": {"action_ids": ["c015"], "action_names": ["hold phone"], "intervals": [{"start_seconds": 0.0, "end_seconds": 46.0}]},
                "stage2_present": False,
                "stage2": {"events": []},
                "vlm": {"confidence": 0.95, "evidence": "phone", "supporting_frame_indices": [2]},
                "fused_status": "vlm_recovered_label",
            },
            {
                "action": "looking_at_phone",
                "action_name": "看手机",
                "charades_present": False,
                "charades": {"action_ids": [], "action_names": [], "intervals": []},
                "stage2_present": False,
                "stage2": {"events": []},
                "vlm": {"confidence": 0.9, "evidence": "looking at screen", "supporting_frame_indices": [3]},
                "fused_status": "vlm_candidate_event",
            },
        ],
    }


if __name__ == "__main__":
    unittest.main()
