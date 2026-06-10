from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_charades import build_manifest, load_classes, load_id_labels, load_mapping, normalize_action_id, parse_actions, slugify, summarize


class PrepareCharadesTests(unittest.TestCase):
    def test_parse_actions(self) -> None:
        actions = parse_actions("c015 0.00 4.30;c019 4.30 8.00")

        self.assertEqual(actions, [("c015", 0.0, 4.3), ("c019", 4.3, 8.0)])

    def test_build_manifest_filters_and_maps_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "Charades_v1_train.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "id",
                        "subject",
                        "scene",
                        "quality",
                        "relevance",
                        "verified",
                        "script",
                        "objects",
                        "descriptions",
                        "actions",
                        "length",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "id": "VID01",
                        "subject": "SUB01",
                        "scene": "Kitchen",
                        "quality": "6",
                        "relevance": "7",
                        "verified": "Yes",
                        "script": "A person drinks while holding a phone.",
                        "objects": "cup;phone",
                        "descriptions": "A person drinks from a cup.",
                        "actions": "c015 0.00 4.30;c106 4.30 8.00",
                        "length": "12.5",
                    }
                )
                writer.writerow(
                    {
                        "id": "VID02",
                        "subject": "SUB02",
                        "scene": "Bedroom",
                        "quality": "4",
                        "relevance": "5",
                        "verified": "No",
                        "script": "",
                        "objects": "door",
                        "descriptions": "",
                        "actions": "c006 0.00 3.00",
                        "length": "5.0",
                    }
                )

            manifest = build_manifest(
                csv_path=csv_path,
                class_labels={"c015": "Holding a phone/camera", "c106": "Drinking from a cup/glass/bottle"},
                mapping={"c015": ("o025", "v008"), "c106": ("o010", "v004")},
                object_labels={"o025": "phone/camera", "o010": "cup/glass/bottle"},
                verb_labels={"v008": "hold", "v004": "drink"},
                action_ids={"c015", "c106"},
                verified_only=True,
                min_quality=5,
            )

        self.assertEqual(len(manifest), 2)
        self.assertEqual(manifest[0].video_id, "VID01")
        self.assertEqual(manifest[0].action, "holding_a_phone_camera")
        self.assertEqual(manifest[0].object_name, "phone/camera")
        self.assertEqual(manifest[1].verb_name, "drink")
        self.assertEqual(manifest[1].duration_seconds, 3.7)

    def test_load_label_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            classes = tmp_path / "classes.txt"
            objects = tmp_path / "objects.txt"
            verbs = tmp_path / "verbs.txt"
            mapping = tmp_path / "mapping.txt"
            classes.write_text("c015 Holding a phone/camera\n", encoding="utf-8")
            objects.write_text("o025 phone/camera\n", encoding="utf-8")
            verbs.write_text("v008 hold\n", encoding="utf-8")
            mapping.write_text("c015 o025 v008\n", encoding="utf-8")

            self.assertEqual(load_classes(classes)["c015"], "Holding a phone/camera")
            self.assertEqual(load_id_labels(objects)["o025"], "phone/camera")
            self.assertEqual(load_id_labels(verbs)["v008"], "hold")
            self.assertEqual(load_mapping(mapping)["c015"], ("o025", "v008"))

    def test_normalize_and_slugify(self) -> None:
        self.assertEqual(normalize_action_id("15"), "c015")
        self.assertEqual(normalize_action_id("c15"), "c015")
        self.assertEqual(slugify("Drinking from a cup/glass/bottle"), "drinking_from_a_cup_glass_bottle")

    def test_summarize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "charades.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "id",
                        "subject",
                        "scene",
                        "quality",
                        "relevance",
                        "verified",
                        "script",
                        "objects",
                        "descriptions",
                        "actions",
                        "length",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "id": "VID01",
                        "subject": "SUB01",
                        "scene": "Kitchen",
                        "quality": "6",
                        "relevance": "7",
                        "verified": "Yes",
                        "script": "",
                        "objects": "cup",
                        "descriptions": "",
                        "actions": "c106 1.00 3.50",
                        "length": "4.0",
                    }
                )

            manifest = build_manifest(
                csv_path,
                {"c106": "Drinking from a cup/glass/bottle"},
                {"c106": ("o010", "v004")},
                {"o010": "cup/glass/bottle"},
                {"v004": "drink"},
            )
            summary = summarize(manifest)

        self.assertEqual(summary["actions"], 1)
        self.assertEqual(summary["videos"], 1)
        self.assertEqual(summary["top_verbs"][0], ("drink", 1))


if __name__ == "__main__":
    unittest.main()
