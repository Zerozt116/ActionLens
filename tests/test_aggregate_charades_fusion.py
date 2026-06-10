from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.aggregate_charades_fusion import (
    aggregate_batch,
    build_status_summary,
    clips_manifest_clip_count,
    read_charades_gt_actions,
    render_markdown,
)


class AggregateCharadesFusionTests(unittest.TestCase):
    def test_build_status_summary_includes_supported_events(self) -> None:
        text = build_status_summary(final=2, semantic=1, pending=0, rejected=1, gt=4)
        self.assertIn("F=2", text)
        self.assertIn("S=1", text)
        self.assertIn("supported_events=3, gt_clips=4", text)

    def test_clips_manifest_clip_count(self) -> None:
        mapping = {"C1": "drinking_water", "C2": "talking_on_phone", "C3": "drinking_water"}
        self.assertEqual(clips_manifest_clip_count(mapping, "drinking_water"), 2)
        self.assertEqual(clips_manifest_clip_count(mapping, "missing"), 0)

    def test_aggregate_batch_sums_buckets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            batch_root = tmp_path / "batch"
            batch_root.mkdir()
            write_fused(batch_root / "V1_c106_t1_2", {
                "final_events": [{"action": "drinking_water"}],
                "semantic_candidates": [],
                "pending_events": [],
                "rejected_events": [],
            })
            write_fused(batch_root / "V1_c106_t3_4", {
                "final_events": [],
                "semantic_candidates": [{"action": "drinking_water"}],
                "pending_events": [],
                "rejected_events": [{"action": "drinking_water"}],
            })
            write_fused(batch_root / "V2_c019_t1_2", {
                "final_events": [],
                "semantic_candidates": [],
                "pending_events": [],
                "rejected_events": [{"action": "talking_on_phone"}],
            })
            payload = aggregate_batch(batch_root, clips_manifest={})
        self.assertEqual(payload["summary"]["total_clips_processed"], 3)
        self.assertEqual(payload["summary"]["unique_actions"], 2)
        self.assertEqual(payload["summary"]["overall_bucket_counts"]["final_events"], 1)
        self.assertEqual(payload["summary"]["overall_bucket_counts"]["semantic_candidates"], 1)
        self.assertEqual(payload["summary"]["overall_bucket_counts"]["rejected_events"], 2)
        per_action = {row["canonical_action"]: row for row in payload["per_action"]}
        self.assertEqual(per_action["drinking_water"]["final"], 1)
        self.assertEqual(per_action["drinking_water"]["rejected"], 1)
        self.assertEqual(per_action["talking_on_phone"]["rejected"], 1)

    def test_aggregate_batch_counts_each_event_action_independently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            batch_root = tmp_path / "batch"
            batch_root.mkdir()
            write_fused(
                batch_root / "V1_c106_t1_2",
                {
                    "final_events": [{"action": "drinking_water"}],
                    "semantic_candidates": [{"action": "holding_phone"}],
                    "pending_events": [],
                    "rejected_events": [{"action": "talking_on_phone"}],
                },
            )

            payload = aggregate_batch(batch_root, clips_manifest={})

        per_action = {row["canonical_action"]: row for row in payload["per_action"]}
        self.assertEqual(per_action["drinking_water"]["final"], 1)
        self.assertEqual(per_action["holding_phone"]["semantic"], 1)
        self.assertEqual(per_action["talking_on_phone"]["rejected"], 1)

    def test_aggregate_batch_counts_charades_gt_from_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            batch_root = tmp_path / "batch"
            clip_dir = batch_root / "V1_c109_t1_2"
            write_fused(
                clip_dir,
                {
                    "final_events": [{"action": "drinking_water"}],
                    "semantic_candidates": [],
                    "pending_events": [],
                    "rejected_events": [],
                },
            )
            write_comparison(
                clip_dir,
                [
                    {"action": "drinking_water", "charades_present": True},
                    {"action": "putting_drink_container", "charades_present": True},
                ],
            )

            payload = aggregate_batch(batch_root, clips_manifest={"V1_c109_t1_2": "putting_drink_container"})

        per_action = {row["canonical_action"]: row for row in payload["per_action"]}
        self.assertEqual(per_action["putting_drink_container"]["clips_total"], 1)
        self.assertEqual(per_action["putting_drink_container"]["charades_groundtruth"], 1)
        self.assertEqual(per_action["drinking_water"]["clips_total"], 0)
        self.assertEqual(per_action["drinking_water"]["charades_groundtruth"], 1)

    def test_render_markdown_includes_section_headers(self) -> None:
        payload = {
            "summary": {
                "total_clips_processed": 1,
                "total_fused_events": 1,
                "unique_videos": 1,
                "unique_actions": 1,
                "unique_actions_with_outputs": 1,
                "overall_bucket_counts": {"final_events": 1, "semantic_candidates": 0, "pending_events": 0, "rejected_events": 0},
            },
            "per_action": [
                {
                    "canonical_action": "drinking_water",
                    "clips_total": 0,
                    "charades_groundtruth": 0,
                    "final": 1,
                    "semantic": 0,
                    "pending": 0,
                    "rejected": 0,
                    "stage2_hits": 1,
                    "vlm_present": 1,
                    "status_summary": "F=1 S=0 P=0 R=0",
                }
            ],
            "per_video": [{"video_id": "V1", "final_events": 1, "semantic_candidates": 0, "pending_events": 0, "rejected_events": 0}],
        }
        text = render_markdown(payload)
        self.assertIn("# Charades 1 Clip Fusion Aggregation Report", text)
        self.assertIn("| Action | Selected clips | Charades GT clips |", text)
        self.assertIn("drinking_water", text)
        self.assertIn("| V1 |", text)

    def test_read_charades_gt_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "comparison.json"
            write_json(
                path,
                {
                    "comparisons": [
                        {"action": "drinking_water", "charades_present": True},
                        {"action": "talking_on_phone", "charades_present": False},
                    ]
                },
            )

            actions = read_charades_gt_actions(path)

        self.assertEqual(actions, ["drinking_water"])


def write_fused(path: Path, payload: dict) -> None:
    path.mkdir(parents=True)
    (path / "fused_events.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_comparison(path: Path, rows: list[dict]) -> None:
    write_json(path / "comparison.json", {"comparisons": rows})


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
