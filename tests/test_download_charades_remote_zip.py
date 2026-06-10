from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.download_charades_remote_zip import find_members_for_video_ids, read_video_ids


class DownloadCharadesRemoteZipTests(unittest.TestCase):
    def test_read_video_ids_keeps_manifest_order_and_uniqueness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "subset.csv"
            with manifest.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["video_id", "action_id"])
                writer.writeheader()
                writer.writerow({"video_id": "VID01", "action_id": "c106"})
                writer.writerow({"video_id": "VID02", "action_id": "c019"})
                writer.writerow({"video_id": "VID01", "action_id": "c015"})

            video_ids = read_video_ids(manifest)

        self.assertEqual(video_ids, ["VID01", "VID02"])

    def test_find_members_for_video_ids(self) -> None:
        members = find_members_for_video_ids(
            [
                "Charades_v1_480/",
                "Charades_v1_480/VID01.mp4",
                "Charades_v1_480/VID02.mp4",
                "Charades_v1_480/notes.txt",
            ],
            ["VID02", "MISSING"],
        )

        self.assertEqual(members, {"VID02": "Charades_v1_480/VID02.mp4"})


if __name__ == "__main__":
    unittest.main()
