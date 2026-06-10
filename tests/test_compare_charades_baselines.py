from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_charades_baselines import BaselineInput, compare_baselines, render_markdown


class CompareCharadesBaselinesTests(unittest.TestCase):
    def test_compare_baselines_builds_metric_and_action_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            left = tmp_path / "left"
            right = tmp_path / "right"
            write_batch(left, total_clips=2, stage2_hits=1, vlm_hits=1)
            write_batch(right, total_clips=5, stage2_hits=3, vlm_hits=4)
            write_aggregation(
                left,
                final=1,
                semantic=1,
                pending=0,
                rejected=2,
                per_action=[
                    action_row("drinking_water", final=1, semantic=0, rejected=2, stage2_hits=3, gt=2),
                    action_row("running", final=0, semantic=1, rejected=0, stage2_hits=0, gt=1),
                ],
            )
            write_aggregation(
                right,
                final=3,
                semantic=4,
                pending=0,
                rejected=8,
                per_action=[
                    action_row("drinking_water", final=3, semantic=0, rejected=8, stage2_hits=11, gt=4),
                    action_row("running", final=0, semantic=2, rejected=0, stage2_hits=0, gt=3),
                    action_row("standing_up", final=0, semantic=0, rejected=0, stage2_hits=0, gt=2),
                ],
            )

            payload = compare_baselines(BaselineInput("batch2", left), BaselineInput("batch5", right))

        self.assertEqual(payload["summary"]["final_events_delta"], 2)
        self.assertEqual(payload["summary"]["semantic_candidates_delta"], 3)
        self.assertEqual(payload["summary"]["unique_actions_with_outputs_delta"], 0)
        per_action = {row["action"]: row for row in payload["per_action"]}
        self.assertEqual(per_action["drinking_water"]["supported_delta"], 2)
        self.assertEqual(per_action["running"]["supported_delta"], 1)
        self.assertEqual(payload["bottlenecks"]["semantic_only_actions"][0]["action"], "running")
        self.assertEqual(payload["bottlenecks"]["no_output_actions"][0]["action"], "standing_up")
        self.assertEqual(payload["bottlenecks"]["high_rejection_actions"][0]["action"], "drinking_water")

    def test_render_markdown_includes_sections(self) -> None:
        payload = {
            "summary": {"left_label": "batch20", "right_label": "batch50"},
            "metrics": [{"metric": "final_events", "left": 6, "right": 20, "delta": 14}],
            "per_action": [
                {
                    "action": "drinking_water",
                    "left_supported_events": 7,
                    "right_supported_events": 16,
                    "supported_delta": 9,
                    "right_charades_gt_clips": 11,
                    "right_stage2_hits": 59,
                    "right_rejected": 44,
                }
            ],
            "bottlenecks": {
                "semantic_only_actions": [],
                "no_output_actions": [],
                "high_rejection_actions": [
                    {
                        "action": "drinking_water",
                        "charades_gt_clips": 11,
                        "supported_events": 16,
                        "final": 15,
                        "semantic": 1,
                        "stage2_hits": 59,
                        "rejected": 44,
                    }
                ],
            },
        }

        text = render_markdown(payload)

        self.assertIn("# Charades Baseline Comparison: batch20 vs batch50", text)
        self.assertIn("## Overall Metrics", text)
        self.assertIn("## Per Action", text)
        self.assertIn("### High-rejection actions", text)
        self.assertIn("drinking_water", text)


def write_batch(path: Path, *, total_clips: int, stage2_hits: int, vlm_hits: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": {
            "total_clips": total_clips,
            "stage2_event_hits": stage2_hits,
            "stage2_event_rate": round(stage2_hits / total_clips, 4),
            "vlm_present_hits": vlm_hits,
            "vlm_present_rate": round(vlm_hits / total_clips, 4),
            "fused_ok": total_clips,
        }
    }
    (path / "batch_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_aggregation(
    path: Path,
    *,
    final: int,
    semantic: int,
    pending: int,
    rejected: int,
    per_action: list[dict],
) -> None:
    payload = {
        "summary": {
            "total_clips_processed": 0,
            "total_fused_events": final + semantic + pending + rejected,
            "unique_videos": 0,
            "unique_actions": len(per_action),
            "unique_actions_with_outputs": sum(
                1
                for row in per_action
                if row["final"] or row["semantic"] or row["pending"] or row["rejected"]
            ),
            "overall_bucket_counts": {
                "final_events": final,
                "semantic_candidates": semantic,
                "pending_events": pending,
                "rejected_events": rejected,
            },
        },
        "per_action": per_action,
    }
    (path / "aggregation_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def action_row(
    action: str,
    *,
    final: int,
    semantic: int,
    rejected: int,
    stage2_hits: int,
    gt: int,
) -> dict:
    return {
        "canonical_action": action,
        "clips_total": gt,
        "charades_groundtruth": gt,
        "final": final,
        "semantic": semantic,
        "pending": 0,
        "rejected": rejected,
        "stage2_hits": stage2_hits,
        "vlm_present": 0,
        "status_summary": "",
    }


if __name__ == "__main__":
    unittest.main()
