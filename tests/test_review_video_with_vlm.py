from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from scripts.review_video_with_vlm import build_payload, load_env_file, parse_review_response, resolve_actions, resolve_review_window, sample_timestamps, strip_code_fence
from scripts.review_video_with_vlm import ExtractedFrame


class ReviewVideoWithVlmTests(unittest.TestCase):
    def test_sample_timestamps_excludes_boundaries(self) -> None:
        self.assertEqual(sample_timestamps(0.0, 8.0, 3), [2.0, 4.0, 6.0])

    def test_sample_single_timestamp_uses_midpoint(self) -> None:
        self.assertEqual(sample_timestamps(2.0, 10.0, 1), [6.0])

    def test_build_payload_embeds_frames_and_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            frame_path = Path(tmp) / "frame.jpg"
            frame_path.write_bytes(b"fakejpeg")
            payload = build_payload(
                model="Qwen/Qwen3-VL-8B-Instruct",
                video_path=Path("video.mp4"),
                metadata={"duration_seconds": 4.0},
                frames=[ExtractedFrame(index=0, timestamp_seconds=1.0, frame_number=30, path=str(frame_path))],
                actions=["drinking_water"],
            )

        self.assertEqual(payload["model"], "Qwen/Qwen3-VL-8B-Instruct")
        content = payload["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertIn("drinking_water", content[0]["text"])
        self.assertIn("It does not need to be literally water", content[0]["text"])
        self.assertEqual(content[2]["type"], "image_url")
        self.assertTrue(content[2]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_parse_review_response_from_json_content(self) -> None:
        review = {"overall_summary": "ok", "actions": [], "visible_objects": [], "risk_notes": []}
        response = {"choices": [{"message": {"content": json.dumps(review)}}]}

        self.assertEqual(parse_review_response(response), review)

    def test_strip_code_fence(self) -> None:
        self.assertEqual(strip_code_fence("```json\n{\"ok\": true}\n```"), "{\"ok\": true}")

    def test_load_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("VLM_TEST_KEY='abc123'\n", encoding="utf-8")
            os.environ.pop("VLM_TEST_KEY", None)
            try:
                load_env_file(env_path)

                self.assertEqual(os.environ["VLM_TEST_KEY"], "abc123")
            finally:
                os.environ.pop("VLM_TEST_KEY", None)

    def test_resolve_review_window_from_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            events_path = Path(tmp) / "events.json"
            events_path.write_text(
                json.dumps(
                    [
                        {
                            "action": "drinking_water",
                            "start_seconds": 11.833,
                            "end_seconds": 12.667,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            window = resolve_review_window(
                explicit_start=None,
                explicit_end=None,
                events_json=events_path,
                event_index=0,
                event_context_seconds=1.0,
                video_duration_seconds=45.5,
            )

        self.assertEqual(window.start_seconds, 10.833)
        self.assertEqual(window.end_seconds, 13.667)
        self.assertEqual(window.source, "event:0")
        self.assertEqual(window.event["action"], "drinking_water")

    def test_resolve_actions_prefers_explicit_actions(self) -> None:
        self.assertEqual(resolve_actions(["talking_on_phone"], {"action": "drinking_water"}), ["talking_on_phone"])
        self.assertEqual(resolve_actions(None, {"action": "drinking_water"}), ["drinking_water"])


if __name__ == "__main__":
    unittest.main()
