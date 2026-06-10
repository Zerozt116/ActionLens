from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_ava import build_manifest, load_action_labels, slugify


class PrepareAvaTests(unittest.TestCase):
    def test_build_manifest_filters_target_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "ava.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["abc123", "902", "0.1", "0.2", "0.3", "0.4", "27", "1"])
                writer.writerow(["abc123", "903", "0.2", "0.3", "0.4", "0.5", "57", "1"])
                writer.writerow(["abc123", "904", "0.2", "0.3", "0.4", "0.5", "80", "1"])

            annotations = build_manifest(csv_path, {27, 57})

        self.assertEqual(len(annotations), 2)
        self.assertEqual(annotations[0].action, "drink")
        self.assertEqual(annotations[1].action, "text_on_look_at_cellphone")
        self.assertEqual(annotations[0].timestamp, 902)

    def test_build_manifest_can_include_all_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "ava.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["abc123", "902", "0.1", "0.2", "0.3", "0.4", "27", "1"])
                writer.writerow(["abc123", "904", "0.2", "0.3", "0.4", "0.5", "80", "1"])

            annotations = build_manifest(csv_path, None)

        self.assertEqual(len(annotations), 2)
        self.assertEqual(annotations[1].action, "ava_action_80")

    def test_load_action_labels_from_pbtxt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            action_list = Path(tmp) / "actions.pbtxt"
            action_list.write_text(
                'label { name: "text on/look at a cellphone" label_id: 57 label_type: OBJECT_MANIPULATION }',
                encoding="utf-8",
            )

            labels = load_action_labels(action_list)

        self.assertEqual(labels[57], ("text_on_look_at_a_cellphone", "text on/look at a cellphone", "OBJECT_MANIPULATION"))

    def test_slugify_action_label(self) -> None:
        self.assertEqual(slugify("talk to (e.g., self, a person, a group)"), "talk_to")


if __name__ == "__main__":
    unittest.main()
