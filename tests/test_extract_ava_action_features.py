from __future__ import annotations

import unittest

from scripts.extract_ava_action_features import (
    bbox_shape_features,
    count_visible_keypoints,
    nearest_person_distance_norm,
    wrist_head_distances,
)


class ExtractAvaActionFeaturesTests(unittest.TestCase):
    def test_bbox_shape_features(self) -> None:
        area, aspect, center_x, center_y = bbox_shape_features([10, 20, 30, 60], width=100, height=100)

        self.assertEqual(area, 0.08)
        self.assertEqual(aspect, 0.5)
        self.assertEqual(center_x, 0.2)
        self.assertEqual(center_y, 0.4)

    def test_wrist_head_distances(self) -> None:
        keypoints = [[0.0, 0.0, 0.0] for _ in range(17)]
        keypoints[0] = [0.0, 0.0, 0.9]
        keypoints[9] = [3.0, 4.0, 0.9]
        keypoints[10] = [0.0, 10.0, 0.9]
        pose = {"keypoints": keypoints}

        distances = wrist_head_distances(pose, scale=10.0)

        self.assertEqual(distances, [0.5, 1.0])

    def test_count_visible_keypoints(self) -> None:
        pose = {"keypoints": [[0, 0, 0.1], [0, 0, 0.2], [0, 0, 0.9]]}

        self.assertEqual(count_visible_keypoints(pose), 2)

    def test_nearest_person_distance_norm(self) -> None:
        people = [
            {"person_id": 1, "bbox_xyxy": [0, 0, 10, 10]},
            {"person_id": 2, "bbox_xyxy": [10, 0, 20, 10]},
            {"person_id": 3, "bbox_xyxy": [30, 0, 40, 10]},
        ]

        distance = nearest_person_distance_norm(people, person_id=1, bbox=[0, 0, 10, 10], scale=10.0)

        self.assertEqual(distance, 1.0)


if __name__ == "__main__":
    unittest.main()
