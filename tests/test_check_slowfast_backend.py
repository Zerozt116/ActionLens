from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.tal.check_slowfast_backend import build_base_summary, build_synthetic_window, write_summary


class CheckSlowFastBackendTests(unittest.TestCase):
    def test_build_synthetic_window_shape(self) -> None:
        window = build_synthetic_window(32)
        self.assertEqual(list(window.shape), [32, 240, 320, 3])
        self.assertEqual(window.dtype.name, "uint8")

    def test_build_base_summary_contains_requested_device(self) -> None:
        summary = build_base_summary("cpu", 32, 4, 256)
        self.assertEqual(summary["requested_device"], "cpu")
        self.assertEqual(summary["clip_frames"], 32)

    def test_write_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            write_summary(path, {"status": "ok"})
            text = path.read_text(encoding="utf-8")

        self.assertIn('"status": "ok"', text)


if __name__ == "__main__":
    unittest.main()
