from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.prepare_ntu_rgbd import build_manifest, parse_ntu_filename


class PrepareNtuRgbdTests(unittest.TestCase):
    def test_parse_ntu_filename(self) -> None:
        parsed = parse_ntu_filename("S001C002P003R004A028_rgb.avi")

        self.assertEqual(
            parsed,
            {
                "setup_id": 1,
                "camera_id": 2,
                "person_id": 3,
                "replication_id": 4,
                "action_id": 28,
            },
        )

    def test_build_manifest_filters_target_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "S001C001P001R001A001_rgb.avi").touch()
            (root / "S001C001P001R001A028_rgb.avi").touch()
            (root / "S001C001P001R001A050_rgb.avi").touch()

            samples = build_manifest(root, {1, 28}, {".avi"})

        self.assertEqual([sample.action_id for sample in samples], [1, 28])
        self.assertEqual(samples[0].action, "drink_water")
        self.assertEqual(samples[1].action, "phone_call")


if __name__ == "__main__":
    unittest.main()
