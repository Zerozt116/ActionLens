from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.slice_charades_clips import (
    SliceRecord,
    build_summary,
    read_manifest_rows,
    slice_one,
    write_clips_csv,
)


class SliceCharadesClipsTests(unittest.TestCase):
    def test_read_manifest_rows_respects_offset_and_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.csv"
            write_manifest(manifest, rows=5)
            rows = read_manifest_rows(manifest, offset=1, limit=2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["video_id"], "V0002")
        self.assertEqual(rows[1]["video_id"], "V0003")

    def test_read_manifest_rows_rejects_missing_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "bad.csv"
            with manifest.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["video_id", "action_id"])
                writer.writeheader()
                writer.writerow({"video_id": "V1", "action_id": "c106"})
            with self.assertRaises(ValueError):
                read_manifest_rows(manifest)

    def test_slice_one_returns_exists_when_clip_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            clips_dir = tmp_path / "clips"
            clips_dir.mkdir()
            existing_clip = clips_dir / "V1_c106_t1.00_5.00.mp4"
            existing_clip.write_bytes(b"fake")
            record = slice_one(
                row={
                    "video_id": "V1",
                    "action_id": "c106",
                    "action_name": "drinking",
                    "start_seconds": "2.0",
                    "end_seconds": "4.0",
                    "clip_start_seconds": "1.0",
                    "clip_end_seconds": "5.0",
                },
                videos_dir=tmp_path / "videos",
                clips_dir=clips_dir,
                overwrite=False,
                ffmpeg=Path("/usr/bin/ffmpeg"),
            )
        self.assertEqual(record.status, "exists")
        self.assertEqual(record.clip_duration_seconds, 4.0)
        self.assertEqual(record.start_seconds, 2.0)

    def test_slice_one_reports_missing_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            record = slice_one(
                row={
                    "video_id": "MISSING",
                    "action_id": "c106",
                    "action_name": "drinking",
                    "start_seconds": "0.0",
                    "end_seconds": "5.0",
                    "clip_start_seconds": "0.0",
                    "clip_end_seconds": "5.0",
                },
                videos_dir=tmp_path / "videos",
                clips_dir=tmp_path / "clips",
                overwrite=False,
                ffmpeg=Path("/usr/bin/ffmpeg"),
            )
        self.assertEqual(record.status, "missing_source")
        self.assertIn("Source video not found", record.error)

    def test_build_summary_aggregates_status_counts(self) -> None:
        records = [
            SliceRecord(
                clip_id="V1",
                video_id="V1",
                action_id="c106",
                action_name="drinking",
                start_seconds=0.0,
                end_seconds=2.0,
                clip_start_seconds=0.0,
                clip_end_seconds=2.0,
                clip_duration_seconds=2.0,
                source_video="",
                clip_path="",
                status="sliced",
            ),
            SliceRecord(
                clip_id="V2",
                video_id="V2",
                action_id="c106",
                action_name="drinking",
                start_seconds=0.0,
                end_seconds=2.0,
                clip_start_seconds=0.0,
                clip_end_seconds=2.0,
                clip_duration_seconds=2.0,
                source_video="",
                clip_path="",
                status="missing_source",
            ),
        ]
        summary = build_summary(records)
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["status_counts"]["sliced"], 1)
        self.assertEqual(summary["status_counts"]["missing_source"], 1)
        self.assertEqual(summary["unique_videos"], 1)
        self.assertEqual(summary["average_clip_duration_seconds"], 2.0)


def write_manifest(path: Path, rows: int) -> None:
    fieldnames = [
        "video_id",
        "action_id",
        "action_name",
        "start_seconds",
        "end_seconds",
        "clip_start_seconds",
        "clip_end_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(1, rows + 1):
            writer.writerow(
                {
                    "video_id": f"V{index:04d}",
                    "action_id": "c106",
                    "action_name": "drinking",
                    "start_seconds": 0.0,
                    "end_seconds": 5.0,
                    "clip_start_seconds": 0.0,
                    "clip_end_seconds": 5.0,
                }
            )


if __name__ == "__main__":
    unittest.main()
