from __future__ import annotations

import unittest

from video_analyst.stage2 import FrameActionScore, _merge_frame_scores


class Stage2RuleTests(unittest.TestCase):
    def test_merge_consecutive_scores_into_event(self) -> None:
        scores = [
            FrameActionScore(
                frame_id=frame_id,
                timestamp=f"00:00:0{frame_id}.000",
                seconds=float(frame_id),
                person_id=1,
                action="drinking_water",
                action_name="喝水",
                confidence=0.8,
                evidence="cup close to head and wrist",
            )
            for frame_id in range(3)
        ]

        events = _merge_frame_scores(scores, min_event_frames=3, min_event_seconds=0.0, max_gap_frames=0)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].person_id, 1)
        self.assertEqual(events[0].action, "drinking_water")
        self.assertEqual(events[0].start_time, "00:00:00.000")
        self.assertEqual(events[0].end_time, "00:00:02.000")
        self.assertEqual(events[0].frame_count, 3)

    def test_short_scores_are_filtered(self) -> None:
        scores = [
            FrameActionScore(
                frame_id=0,
                timestamp="00:00:00.000",
                seconds=0.0,
                person_id=1,
                action="talking_on_phone",
                action_name="打电话",
                confidence=0.9,
                evidence="cell phone close to head",
            )
        ]

        events = _merge_frame_scores(scores, min_event_frames=3, min_event_seconds=0.0, max_gap_frames=1)

        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
