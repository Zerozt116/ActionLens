from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.download_ava_clips import ClipRequest, build_download_command, read_clip_requests


class DownloadAvaClipsTests(unittest.TestCase):
    def test_read_clip_requests_deduplicates_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "subset.csv"
            with manifest.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["video_id", "timestamp", "x1", "y1", "x2", "y2", "action_id", "action", "action_name", "action_type", "person_id"])
                writer.writerow(["abc123", 902, 0, 0, 1, 1, 12, "stand", "stand", "PERSON_MOVEMENT", 1])
                writer.writerow(["abc123", 902, 0, 0, 1, 1, 12, "stand", "stand", "PERSON_MOVEMENT", 1])
                writer.writerow(["def456", 905, 0, 0, 1, 1, 14, "walk", "walk", "PERSON_MOVEMENT", 2])

            requests = read_clip_requests(manifest, limit=10)

        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0].video_id, "abc123")

    def test_build_download_command_uses_timestamp_window(self) -> None:
        request = ClipRequest(video_id="abc123", timestamp=902, action_id=12, action="stand", person_id=1)

        command = build_download_command(request, Path("clips"), seconds_before=2.0, seconds_after=3.0)

        self.assertIn("*900.000-905.000", command)
        self.assertIn("https://www.youtube.com/watch?v=abc123", command)


if __name__ == "__main__":
    unittest.main()
