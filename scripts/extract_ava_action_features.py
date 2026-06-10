from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


NOSE = 0
LEFT_WRIST = 9
RIGHT_WRIST = 10
KEYPOINT_CONFIDENCE = 0.2


@dataclass(frozen=True)
class ActionFeatureRow:
    clip_path: str
    stage2_output_dir: str
    video_id: str
    timestamp: int
    ava_person_id: int
    action_id: int
    action: str
    action_name: str
    action_type: str
    matched_iou: float
    matched_stage2_person_id: int | None
    matched_frame_id: int | None
    has_pose: bool
    visible_keypoints: int
    bbox_area_norm: float
    bbox_aspect_ratio: float
    bbox_center_x_norm: float
    bbox_center_y_norm: float
    min_wrist_head_distance_norm: float | None
    avg_wrist_head_distance_norm: float | None
    person_count_in_frame: int
    nearest_person_distance_norm: float | None
    object_count_in_window: int
    nearest_object_class: str | None
    nearest_object_distance_norm: float | None
    event_count: int


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract general action features from AVA alignment and stage2 outputs.")
    parser.add_argument("--batch-summary", type=Path, required=True, help="Batch summary JSON from run_ava_batch_eval.py.")
    parser.add_argument("--output", type=Path, default=Path("outputs/ava_action_features.csv"))
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--frame-window", type=int, default=5)
    args = parser.parse_args()

    rows = extract_batch_features(args.batch_summary, frame_window=args.frame_window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, rows)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps([asdict(row) for row in rows], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} feature rows to {args.output}")


def extract_batch_features(batch_summary_path: Path, frame_window: int = 5) -> list[ActionFeatureRow]:
    payload = read_json(batch_summary_path)
    rows: list[ActionFeatureRow] = []
    for record in payload.get("records", []):
        if record.get("status") != "ok":
            continue
        stage2_output_dir = Path(record["stage2_output_dir"])
        alignment_path = stage2_output_dir / "ava_alignment.json"
        if not alignment_path.exists():
            continue
        rows.append(extract_features_for_alignment(alignment_path, frame_window=frame_window))
    return rows


def extract_features_for_alignment(alignment_path: Path, frame_window: int = 5) -> ActionFeatureRow:
    alignment = read_json(alignment_path)
    stage2_output_dir = Path(alignment["stage2_output_dir"])
    metadata = read_json(stage2_output_dir / "metadata.json")
    person_boxes = read_json(stage2_output_dir / "person_boxes.json")
    poses = read_json(stage2_output_dir / "poses.json")
    objects = read_json(stage2_output_dir / "objects.json")
    events = read_json(stage2_output_dir / "events.json")

    width = float(metadata["width"])
    height = float(metadata["height"])
    scale = math.hypot(width, height)
    frame_id = alignment["best_frame_id"]
    person_id = alignment["best_stage2_person_id"]
    bbox = alignment["best_stage2_bbox_xyxy"]
    target = alignment["target"]

    if frame_id is None or person_id is None or bbox is None:
        return empty_feature_row(alignment, len(events))

    same_frame_people = person_boxes.get(str(frame_id), [])
    pose = find_pose(poses, frame_id, person_id)
    visible_keypoints = count_visible_keypoints(pose)
    wrist_distances = wrist_head_distances(pose, scale)
    nearest_person_distance = nearest_person_distance_norm(same_frame_people, person_id, bbox, scale)
    window_objects = [obj for obj in objects if abs(int(obj["frame_id"]) - int(frame_id)) <= frame_window]
    nearest_object_class, nearest_object_distance = nearest_object(window_objects, bbox, scale)

    area, aspect_ratio, center_x, center_y = bbox_shape_features(bbox, width, height)
    return ActionFeatureRow(
        clip_path=alignment["clip_path"],
        stage2_output_dir=alignment["stage2_output_dir"],
        video_id=target["video_id"],
        timestamp=int(target["timestamp"]),
        ava_person_id=int(target["person_id"]),
        action_id=int(target["action_id"]),
        action=target["action"],
        action_name=target["action_name"],
        action_type=target["action_type"],
        matched_iou=float(alignment["best_iou"]),
        matched_stage2_person_id=int(person_id),
        matched_frame_id=int(frame_id),
        has_pose=pose is not None,
        visible_keypoints=visible_keypoints,
        bbox_area_norm=round(area, 6),
        bbox_aspect_ratio=round(aspect_ratio, 6),
        bbox_center_x_norm=round(center_x, 6),
        bbox_center_y_norm=round(center_y, 6),
        min_wrist_head_distance_norm=round(min(wrist_distances), 6) if wrist_distances else None,
        avg_wrist_head_distance_norm=round(sum(wrist_distances) / len(wrist_distances), 6) if wrist_distances else None,
        person_count_in_frame=len(same_frame_people),
        nearest_person_distance_norm=round(nearest_person_distance, 6) if nearest_person_distance is not None else None,
        object_count_in_window=len(window_objects),
        nearest_object_class=nearest_object_class,
        nearest_object_distance_norm=round(nearest_object_distance, 6) if nearest_object_distance is not None else None,
        event_count=len(events),
    )


def empty_feature_row(alignment: dict[str, Any], event_count: int) -> ActionFeatureRow:
    target = alignment["target"]
    return ActionFeatureRow(
        clip_path=alignment["clip_path"],
        stage2_output_dir=alignment["stage2_output_dir"],
        video_id=target["video_id"],
        timestamp=int(target["timestamp"]),
        ava_person_id=int(target["person_id"]),
        action_id=int(target["action_id"]),
        action=target["action"],
        action_name=target["action_name"],
        action_type=target["action_type"],
        matched_iou=float(alignment["best_iou"]),
        matched_stage2_person_id=None,
        matched_frame_id=None,
        has_pose=False,
        visible_keypoints=0,
        bbox_area_norm=0.0,
        bbox_aspect_ratio=0.0,
        bbox_center_x_norm=0.0,
        bbox_center_y_norm=0.0,
        min_wrist_head_distance_norm=None,
        avg_wrist_head_distance_norm=None,
        person_count_in_frame=0,
        nearest_person_distance_norm=None,
        object_count_in_window=0,
        nearest_object_class=None,
        nearest_object_distance_norm=None,
        event_count=event_count,
    )


def bbox_shape_features(bbox: list[float], width: float, height: float) -> tuple[float, float, float, float]:
    box_width = max(0.0, bbox[2] - bbox[0])
    box_height = max(0.0, bbox[3] - bbox[1])
    area = (box_width * box_height) / max(1.0, width * height)
    aspect_ratio = box_width / box_height if box_height else 0.0
    center_x = ((bbox[0] + bbox[2]) / 2.0) / width if width else 0.0
    center_y = ((bbox[1] + bbox[3]) / 2.0) / height if height else 0.0
    return area, aspect_ratio, center_x, center_y


def find_pose(poses: list[dict[str, Any]], frame_id: int, person_id: int) -> dict[str, Any] | None:
    for pose in poses:
        if int(pose["frame_id"]) == int(frame_id) and int(pose["person_id"]) == int(person_id):
            return pose
    return None


def count_visible_keypoints(pose: dict[str, Any] | None) -> int:
    if pose is None:
        return 0
    return sum(1 for _x, _y, conf in pose["keypoints"] if float(conf) >= KEYPOINT_CONFIDENCE)


def wrist_head_distances(pose: dict[str, Any] | None, scale: float) -> list[float]:
    if pose is None or scale <= 0:
        return []
    keypoints = pose["keypoints"]
    if len(keypoints) <= RIGHT_WRIST or float(keypoints[NOSE][2]) < KEYPOINT_CONFIDENCE:
        return []
    head = (float(keypoints[NOSE][0]), float(keypoints[NOSE][1]))
    distances: list[float] = []
    for index in (LEFT_WRIST, RIGHT_WRIST):
        x, y, conf = keypoints[index]
        if float(conf) >= KEYPOINT_CONFIDENCE:
            distances.append(math.dist(head, (float(x), float(y))) / scale)
    return distances


def nearest_person_distance_norm(
    people: list[dict[str, Any]],
    person_id: int,
    bbox: list[float],
    scale: float,
) -> float | None:
    if scale <= 0:
        return None
    center = bbox_center(bbox)
    distances = [
        math.dist(center, bbox_center(person["bbox_xyxy"])) / scale
        for person in people
        if int(person["person_id"]) != int(person_id)
    ]
    return min(distances) if distances else None


def nearest_object(
    objects: list[dict[str, Any]],
    bbox: list[float],
    scale: float,
) -> tuple[str | None, float | None]:
    if not objects or scale <= 0:
        return None, None
    center = bbox_center(bbox)
    best = min(objects, key=lambda obj: math.dist(center, bbox_center(obj["bbox_xyxy"])))
    return best["class_name"], math.dist(center, bbox_center(best["bbox_xyxy"])) / scale


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def write_csv(path: Path, rows: list[ActionFeatureRow]) -> None:
    fieldnames = list(ActionFeatureRow.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
