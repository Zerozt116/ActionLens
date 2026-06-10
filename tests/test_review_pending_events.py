from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.review_pending_events import review_pending_events


class ReviewPendingEventsTests(unittest.TestCase):
    def test_review_pending_events_runs_review_and_refuses_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            batch_root = tmp_path / "batch"
            clip_dir = batch_root / "VID_c106_t0_10"
            stage2_dir = clip_dir / "stage2"
            stage2_dir.mkdir(parents=True)
            write_json(
                batch_root / "batch_summary.json",
                {
                    "records": [
                        {
                            "clip_id": "VID_c106_t0_10",
                            "clip_path": "/tmp/clip.mp4",
                        }
                    ]
                },
            )
            write_json(
                stage2_dir / "events.json",
                [
                    {
                        "action": "drinking_water",
                        "start_seconds": 1.0,
                        "end_seconds": 2.0,
                    }
                ],
            )
            write_json(
                clip_dir / "comparison.json",
                {
                    "video_id": "VID",
                    "comparisons": [
                        {
                            "action": "drinking_water",
                            "action_name": "喝水",
                            "charades_present": True,
                            "charades": {"action_ids": ["c106"], "action_names": ["drink"], "intervals": [{"start_seconds": 1.0, "end_seconds": 2.0}]},
                            "stage2": {
                                "events": [
                                    {
                                        "stage2_event_index": 0,
                                        "action": "drinking_water",
                                        "confidence": 0.7,
                                        "start_seconds": 1.0,
                                        "end_seconds": 2.0,
                                    }
                                ]
                            },
                            "fused_status": "needs_temporal_review",
                        }
                    ],
                },
            )
            write_json(
                clip_dir / "fused_events.json",
                {
                    "pending_events": [
                        {
                            "stage2_event_index": 0,
                            "action": "drinking_water",
                        }
                    ]
                },
            )

            def fake_run(command, check, capture_output, text):
                review_dir = Path(command[command.index("-o") + 1])
                write_json(
                    review_dir / "vlm_summary.json",
                    {
                        "review_window": {"source": "event:0"},
                        "review_json": {
                            "actions": [
                                {
                                    "action": "drinking_water",
                                    "present": True,
                                    "confidence": 0.8,
                                    "evidence": "cup",
                                }
                            ]
                        },
                    },
                )
                return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

            with patch("scripts.review_pending_events.subprocess.run", side_effect=fake_run):
                records = review_pending_events(
                    batch_root=batch_root,
                    frame_count=2,
                    event_context_seconds=1.0,
                    model="model",
                    base_url="https://example.test",
                    api_key_env="KEY",
                    env_file=Path(".env"),
                    dry_run=False,
                )

            fused = json.loads((clip_dir / "fused_events.json").read_text(encoding="utf-8"))

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].status, "completed")
        self.assertTrue(records[0].present)
        self.assertEqual(fused["summary"]["final_events"], 1)
        self.assertEqual(fused["summary"]["pending_events"], 0)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
