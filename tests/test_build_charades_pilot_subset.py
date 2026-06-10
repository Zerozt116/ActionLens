from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.build_charades_pilot_subset import build_subset, build_summary, normalize_action_id


FIELDNAMES = [
    "video_id",
    "subject",
    "scene",
    "quality",
    "relevance",
    "verified",
    "video_length",
    "action_id",
    "action_index",
    "action",
    "action_name",
    "object_id",
    "object_name",
    "verb_id",
    "verb_name",
    "start_seconds",
    "end_seconds",
    "duration_seconds",
    "objects",
    "descriptions",
]


class BuildCharadesPilotSubsetTests(unittest.TestCase):
    def test_build_subset_balances_actions_and_keeps_unique_videos(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
                writer.writeheader()
                for index in range(4):
                    writer.writerow(make_row(f"drink{index}", "Kitchen", "c106", "Drinking", quality=7, start=4.0, end=8.0))
                for index in range(4):
                    writer.writerow(make_row(f"phone{index}", "Bedroom", "c019", "Talking phone", quality=6, start=2.0, end=6.0))
                writer.writerow(make_row("short", "Kitchen", "c106", "Drinking", quality=7, start=1.0, end=1.5))

            rows = build_subset(
                manifest_path=manifest,
                action_ids=["c106", "c019"],
                max_videos=4,
                samples_per_action=2,
                max_per_scene=3,
                min_duration=2.0,
                context_seconds=1.0,
            )

        self.assertEqual(len(rows), 4)
        self.assertEqual(len({row.video_id for row in rows}), 4)
        self.assertEqual(sum(row.action_id == "c106" for row in rows), 2)
        self.assertEqual(sum(row.action_id == "c019" for row in rows), 2)
        self.assertTrue(all(row.video_id != "short" for row in rows))
        self.assertEqual(rows[0].clip_start_seconds, 3.0)
        self.assertEqual(rows[0].clip_end_seconds, 9.0)

    def test_scene_limit_is_applied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
                writer.writeheader()
                for index in range(4):
                    writer.writerow(make_row(f"v{index}", "Kitchen", "c106", "Drinking", quality=7))

            rows = build_subset(
                manifest_path=manifest,
                action_ids=["c106"],
                max_videos=4,
                samples_per_action=4,
                max_per_scene=2,
                min_duration=2.0,
                context_seconds=1.0,
            )

        self.assertEqual(len(rows), 2)

    def test_build_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
                writer.writeheader()
                writer.writerow(make_row("v1", "Kitchen", "c106", "Drinking"))

            rows = build_subset(manifest, ["c106"], max_videos=1, samples_per_action=1, max_per_scene=1, min_duration=2.0, context_seconds=2.0)
            summary = build_summary(rows)

        self.assertEqual(summary["selected_videos"], 1)
        self.assertEqual(summary["actions"], 1)
        self.assertEqual(summary["by_action"][0]["action_id"], "c106")

    def test_normalize_action_id(self) -> None:
        self.assertEqual(normalize_action_id("106"), "c106")
        self.assertEqual(normalize_action_id("c19"), "c019")


def make_row(
    video_id: str,
    scene: str,
    action_id: str,
    action_name: str,
    quality: int = 6,
    start: float = 1.0,
    end: float = 5.0,
) -> dict[str, object]:
    return {
        "video_id": video_id,
        "subject": "subject",
        "scene": scene,
        "quality": quality,
        "relevance": 7,
        "verified": "Yes",
        "video_length": 12.0,
        "action_id": action_id,
        "action_index": int(action_id[1:]),
        "action": action_name.lower().replace(" ", "_"),
        "action_name": action_name,
        "object_id": "o010",
        "object_name": "cup/glass/bottle",
        "verb_id": "v004",
        "verb_name": "drink",
        "start_seconds": start,
        "end_seconds": end,
        "duration_seconds": end - start,
        "objects": "cup",
        "descriptions": "description",
    }


if __name__ == "__main__":
    unittest.main()
