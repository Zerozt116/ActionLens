from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_charades_clip_batch import (
    ClipRecord,
    build_tool_env,
    combine_errors,
    find_event_review_paths,
    read_clips,
    run_vlm,
    summarize_records,
)


class RunCharadesClipBatchTests(unittest.TestCase):
    def test_read_clips_skips_non_sliced_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "clips.csv"
            write_clips_manifest(
                manifest,
                rows=[
                    ("C1", "V1", "c106", "sliced"),
                    ("C2", "V1", "c019", "missing_source"),
                    ("C3", "V2", "c015", "sliced"),
                    ("C4", "V2", "c051", "sliced"),
                ],
            )
            rows = read_clips(manifest, offset=0, limit=10)
        self.assertEqual(len(rows), 3)
        self.assertEqual([row["clip_id"] for row in rows], ["C1", "C3", "C4"])

    def test_read_clips_respects_offset_and_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "clips.csv"
            write_clips_manifest(
                manifest,
                rows=[(f"C{i}", "V1", "c106", "sliced") for i in range(5)],
            )
            rows = read_clips(manifest, offset=1, limit=2)
        self.assertEqual([row["clip_id"] for row in rows], ["C1", "C2"])

    def test_summarize_records_computes_rates(self) -> None:
        records = [
            make_record(clip_id="C1", canonical="drinking_water", stage2_status="ok", stage2_events=1, vlm_status="completed", vlm_present=True, fused="ok"),
            make_record(clip_id="C2", canonical="talking_on_phone", stage2_status="ok", stage2_events=0, vlm_status="completed", vlm_present=False, fused="ok"),
            make_record(clip_id="C3", canonical="drinking_water", stage2_status="stage2_failed", stage2_events=0, vlm_status="skipped", vlm_present=False, fused="skipped"),
            make_record(clip_id="C4", canonical="drinking_water", stage2_status="exists", stage2_events=2, vlm_status="exists", vlm_present=True, fused="ok"),
        ]
        summary = summarize_records(records)
        self.assertEqual(summary["total_clips"], 4)
        self.assertEqual(summary["stage2_ok"], 3)
        self.assertEqual(summary["stage2_event_hits"], 2)
        self.assertEqual(summary["vlm_ok"], 3)
        self.assertEqual(summary["vlm_present_hits"], 2)
        self.assertEqual(summary["fused_ok"], 3)
        self.assertEqual(summary["by_canonical_action"]["drinking_water"], 3)
        self.assertEqual(summary["stage2_event_rate"], round(2 / 3, 4))
        self.assertEqual(summary["vlm_present_rate"], round(2 / 3, 4))

    def test_build_tool_env_adds_tool_dirs(self) -> None:
        env = build_tool_env(Path("/a/bin/ffmpeg"), Path("/b/bin/video-analyst"))
        self.assertIn("/a/bin", env["PATH"])
        self.assertIn("/b/bin", env["PATH"])
        self.assertIn(os.environ.get("PATH", ""), env["PATH"])

    def test_run_vlm_uses_custom_api_key_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            vlm_dir = tmp_path / "vlm"
            captured: dict[str, list[str]] = {}

            def fake_run(command, check, env, capture_output, text):
                captured["command"] = command
                vlm_dir.mkdir(parents=True, exist_ok=True)
                (vlm_dir / "vlm_summary.json").write_text(
                    json.dumps(
                        {
                            "status": "completed",
                            "review_json": {
                                "actions": [
                                    {
                                        "action": "drinking_water",
                                        "present": True,
                                    }
                                ]
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                return type("Completed", (), {"returncode": 0, "stderr": "", "stdout": ""})()

            with patch("scripts.run_charades_clip_batch.subprocess.run", side_effect=fake_run), patch("builtins.print"):
                status, present, error = run_vlm(
                    clip={"clip_path": "/tmp/clip.mp4"},
                    vlm_dir=vlm_dir,
                    vlm_model="model",
                    vlm_base_url="https://example.test",
                    api_key_env="CUSTOM_KEY",
                    has_api_key=True,
                    python=Path("/tmp/python"),
                    env={"CUSTOM_KEY": "secret"},
                    actions=["drinking_water"],
                    frame_count=2,
                )

        command = captured["command"]
        self.assertEqual(status, "completed")
        self.assertTrue(present)
        self.assertEqual(error, "")
        self.assertIn("--api-key-env", command)
        self.assertEqual(command[command.index("--api-key-env") + 1], "CUSTOM_KEY")

    def test_combine_errors_omits_empty_entries(self) -> None:
        self.assertEqual(combine_errors("", "vlm failed", "", "fuse failed"), "vlm failed; fuse failed")

    def test_find_event_review_paths_only_reads_event_reviews_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clip_dir = Path(tmp)
            (clip_dir / "vlm_review").mkdir()
            event_dir = clip_dir / "event_reviews" / "event_000_drinking_water"
            event_dir.mkdir(parents=True)

            paths = find_event_review_paths(clip_dir)

        self.assertEqual(paths, [event_dir])


def make_record(
    clip_id: str,
    canonical: str,
    stage2_status: str,
    stage2_events: int,
    vlm_status: str,
    vlm_present: bool,
    fused: str,
) -> ClipRecord:
    return ClipRecord(
        clip_id=clip_id,
        video_id="VID",
        action_id="c106",
        action_name="action",
        canonical_action=canonical,
        clip_path="/tmp/clip.mp4",
        stage2_dir="/tmp/stage2",
        vlm_dir=None,
        comparison_path=None,
        fused_path=None,
        stage2_status=stage2_status,
        vlm_status=vlm_status,
        fused_status=fused,
        stage2_event_count=stage2_events,
        vlm_present=vlm_present,
    )


def write_clips_manifest(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    fieldnames = ["clip_id", "video_id", "action_id", "action_name", "status", "clip_path"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for clip_id, video_id, action_id, status in rows:
            writer.writerow(
                {
                    "clip_id": clip_id,
                    "video_id": video_id,
                    "action_id": action_id,
                    "action_name": "action",
                    "status": status,
                    "clip_path": f"/tmp/{clip_id}.mp4",
                }
            )


if __name__ == "__main__":
    unittest.main()
