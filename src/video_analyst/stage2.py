from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2

from .pipeline import PERSON_CLASS_ID
from .video_io import VideoMetadata, probe_video, seconds_to_timestamp


TARGET_OBJECT_NAMES = {"bottle", "cup", "cell phone"}
DRINKING_OBJECT_NAMES = {"bottle", "cup"}
PHONE_OBJECT_NAMES = {"cell phone"}
KEYPOINT_CONFIDENCE = 0.2
DRINKING_HEAD_DISTANCE_THRESHOLD = 0.5
DRINKING_WRIST_DISTANCE_THRESHOLD = 0.4
PHONE_HEAD_DISTANCE_THRESHOLD = 0.34


@dataclass(frozen=True)
class ObjectRecord:
    frame_id: int
    timestamp: str
    seconds: float
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]


@dataclass(frozen=True)
class PoseRecord:
    frame_id: int
    timestamp: str
    seconds: float
    person_id: int
    confidence: float
    bbox_xyxy: list[float]
    keypoints: list[list[float]]


@dataclass(frozen=True)
class FrameActionScore:
    frame_id: int
    timestamp: str
    seconds: float
    person_id: int
    action: str
    action_name: str
    confidence: float
    evidence: str


@dataclass(frozen=True)
class EventRecord:
    person_id: int
    action: str
    action_name: str
    start_time: str
    end_time: str
    start_seconds: float
    end_seconds: float
    frame_count: int
    confidence: float
    evidence: str


@dataclass(frozen=True)
class Stage2Result:
    video: VideoMetadata
    objects: list[ObjectRecord]
    poses: list[PoseRecord]
    frame_scores: list[FrameActionScore]
    events: list[EventRecord]
    annotated_video: str | None

    def to_summary(self) -> dict[str, Any]:
        return {
            "video": self.video.to_dict(),
            "num_objects": len(self.objects),
            "num_poses": len(self.poses),
            "num_frame_scores": len(self.frame_scores),
            "num_events": len(self.events),
            "events": [asdict(event) for event in self.events],
            "annotated_video": self.annotated_video,
        }


@dataclass(frozen=True)
class _PersonBox:
    frame_id: int
    timestamp: str
    seconds: float
    person_id: int
    bbox_xyxy: list[float]
    confidence: float


def run_stage2(
    video_path: str | Path,
    output_dir: str | Path,
    detection_model_name: str = "yolo11n.pt",
    pose_model_name: str = "yolo11n-pose.pt",
    confidence: float = 0.25,
    iou: float = 0.5,
    image_size: int = 640,
    device: str | None = None,
    min_event_frames: int = 3,
    min_event_seconds: float = 0.3,
    max_gap_frames: int = 6,
    write_annotated_video: bool = True,
) -> Stage2Result:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Run: ./.conda/bin/python -m pip install -e ."
        ) from exc

    input_path = Path(video_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = probe_video(input_path)
    detection_model = YOLO(detection_model_name)
    pose_model = YOLO(pose_model_name)
    target_class_ids = _target_class_ids(detection_model.names)

    objects: list[ObjectRecord] = []
    poses: list[PoseRecord] = []
    frame_scores: list[FrameActionScore] = []
    person_boxes_by_frame: dict[int, list[_PersonBox]] = defaultdict(list)

    annotated_path = out_dir / "stage2_annotated.mp4" if write_annotated_video else None
    writer: cv2.VideoWriter | None = None

    track_kwargs: dict[str, Any] = {
        "source": str(input_path),
        "stream": True,
        "persist": True,
        "tracker": "bytetrack.yaml",
        "classes": [PERSON_CLASS_ID, *sorted(target_class_ids)],
        "conf": confidence,
        "iou": iou,
        "imgsz": image_size,
        "verbose": False,
    }
    pose_kwargs: dict[str, Any] = {
        "conf": confidence,
        "iou": iou,
        "imgsz": image_size,
        "verbose": False,
    }
    if device:
        track_kwargs["device"] = device
        pose_kwargs["device"] = device

    try:
        for frame_id, detection_result in enumerate(detection_model.track(**track_kwargs)):
            seconds = frame_id / metadata.fps if metadata.fps > 0 else 0.0
            timestamp = seconds_to_timestamp(seconds)
            frame = detection_result.orig_img
            person_boxes, object_records = _extract_detections(
                detection_result=detection_result,
                frame_id=frame_id,
                seconds=seconds,
                timestamp=timestamp,
                model_names=detection_model.names,
                target_class_ids=target_class_ids,
            )
            person_boxes_by_frame[frame_id].extend(person_boxes)
            objects.extend(object_records)

            pose_result = pose_model.predict(frame, **pose_kwargs)[0]
            pose_records = _extract_poses(
                pose_result=pose_result,
                frame_id=frame_id,
                seconds=seconds,
                timestamp=timestamp,
                person_boxes=person_boxes,
            )
            poses.extend(pose_records)

            by_person_pose = {pose.person_id: pose for pose in pose_records}
            current_scores = _score_frame_actions(person_boxes, object_records, by_person_pose)
            frame_scores.extend(current_scores)

            if annotated_path is not None:
                if writer is None:
                    height, width = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(annotated_path), fourcc, metadata.fps or 25.0, (width, height))
                writer.write(_draw_stage2_frame(detection_result.plot(), current_scores))
    finally:
        if writer is not None:
            writer.release()

    events = _merge_frame_scores(
        frame_scores,
        min_event_frames=min_event_frames,
        min_event_seconds=min_event_seconds,
        max_gap_frames=max_gap_frames,
    )
    result = Stage2Result(
        video=metadata,
        objects=objects,
        poses=poses,
        frame_scores=frame_scores,
        events=events,
        annotated_video=str(annotated_path) if annotated_path is not None else None,
    )
    _write_outputs(out_dir, result, person_boxes_by_frame)
    return result


def _target_class_ids(model_names: dict[int, str] | list[str]) -> set[int]:
    if isinstance(model_names, dict):
        iterable = model_names.items()
    else:
        iterable = enumerate(model_names)
    return {int(class_id) for class_id, name in iterable if str(name) in TARGET_OBJECT_NAMES}


def _extract_detections(
    detection_result: Any,
    frame_id: int,
    seconds: float,
    timestamp: str,
    model_names: dict[int, str] | list[str],
    target_class_ids: set[int],
) -> tuple[list[_PersonBox], list[ObjectRecord]]:
    boxes = detection_result.boxes
    if boxes is None or len(boxes) == 0:
        return [], []

    ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
    xyxy = boxes.xyxy.cpu().tolist()
    confs = boxes.conf.cpu().tolist()
    classes = boxes.cls.int().cpu().tolist()

    person_boxes: list[_PersonBox] = []
    object_records: list[ObjectRecord] = []
    for track_id, bbox, conf, class_id in zip(ids, xyxy, confs, classes, strict=True):
        clean_bbox = [round(float(value), 3) for value in bbox]
        conf_value = round(float(conf), 4)
        if class_id == PERSON_CLASS_ID and track_id is not None:
            person_boxes.append(
                _PersonBox(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    seconds=round(seconds, 3),
                    person_id=int(track_id),
                    bbox_xyxy=clean_bbox,
                    confidence=conf_value,
                )
            )
        elif class_id in target_class_ids:
            object_records.append(
                ObjectRecord(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    seconds=round(seconds, 3),
                    class_id=class_id,
                    class_name=str(model_names[class_id]),
                    confidence=conf_value,
                    bbox_xyxy=clean_bbox,
                )
            )
    return person_boxes, object_records


def _extract_poses(
    pose_result: Any,
    frame_id: int,
    seconds: float,
    timestamp: str,
    person_boxes: list[_PersonBox],
) -> list[PoseRecord]:
    boxes = pose_result.boxes
    keypoints = pose_result.keypoints
    if boxes is None or keypoints is None or len(boxes) == 0 or len(person_boxes) == 0:
        return []

    pose_boxes = boxes.xyxy.cpu().tolist()
    pose_confs = boxes.conf.cpu().tolist()
    pose_keypoints = keypoints.data.cpu().tolist()

    matches: list[tuple[float, int, int]] = []
    for pose_idx, pose_bbox in enumerate(pose_boxes):
        for person_idx, person_box in enumerate(person_boxes):
            matches.append((_bbox_iou(pose_bbox, person_box.bbox_xyxy), pose_idx, person_idx))

    matched_poses: set[int] = set()
    matched_persons: set[int] = set()
    records: list[PoseRecord] = []
    for overlap, pose_idx, person_idx in sorted(matches, reverse=True):
        if overlap < 0.25 or pose_idx in matched_poses or person_idx in matched_persons:
            continue
        matched_poses.add(pose_idx)
        matched_persons.add(person_idx)
        person_box = person_boxes[person_idx]
        records.append(
            PoseRecord(
                frame_id=frame_id,
                timestamp=timestamp,
                seconds=round(seconds, 3),
                person_id=person_box.person_id,
                confidence=round(float(pose_confs[pose_idx]), 4),
                bbox_xyxy=[round(float(value), 3) for value in pose_boxes[pose_idx]],
                keypoints=[
                    [round(float(x), 3), round(float(y), 3), round(float(conf), 4)]
                    for x, y, conf in pose_keypoints[pose_idx]
                ],
            )
        )
    return records


def _score_frame_actions(
    person_boxes: list[_PersonBox],
    objects: list[ObjectRecord],
    poses_by_person: dict[int, PoseRecord],
) -> list[FrameActionScore]:
    scores: list[FrameActionScore] = []
    for person in person_boxes:
        pose = poses_by_person.get(person.person_id)
        drinking = _score_drinking(person, objects, pose)
        if drinking is not None:
            scores.append(drinking)

        phone_call = _score_phone_call(person, objects, pose)
        if phone_call is not None:
            scores.append(phone_call)
    return scores


def _score_drinking(
    person: _PersonBox,
    objects: list[ObjectRecord],
    pose: PoseRecord | None,
) -> FrameActionScore | None:
    candidates = [obj for obj in objects if obj.class_name in DRINKING_OBJECT_NAMES and _bbox_intersects(person.bbox_xyxy, obj.bbox_xyxy)]
    if not candidates:
        return None

    head = _head_point(person.bbox_xyxy, pose)
    wrists = _wrist_points(pose)
    person_scale = _bbox_diag(person.bbox_xyxy)

    best: tuple[float, ObjectRecord, str] | None = None
    for obj in candidates:
        center = _bbox_center(obj.bbox_xyxy)
        head_distance = _distance(center, head) / person_scale
        wrist_distance = min((_distance(center, wrist) / person_scale for wrist in wrists), default=0.35)
        near_head = head_distance <= DRINKING_HEAD_DISTANCE_THRESHOLD
        near_hand = wrist_distance <= DRINKING_WRIST_DISTANCE_THRESHOLD if wrists else _bbox_contains(person.bbox_xyxy, center)
        if not (near_head and near_hand):
            continue
        score = _clamp(
            0.25
            + obj.confidence * 0.45
            + max(0.0, DRINKING_HEAD_DISTANCE_THRESHOLD - head_distance) * 0.5
            + max(0.0, DRINKING_WRIST_DISTANCE_THRESHOLD - wrist_distance) * 0.5,
            0.0,
            0.95,
        )
        evidence = f"{obj.class_name} close to head and wrist; head_dist={head_distance:.2f}, wrist_dist={wrist_distance:.2f}"
        if best is None or score > best[0]:
            best = (score, obj, evidence)

    if best is None:
        return None

    return FrameActionScore(
        frame_id=person.frame_id,
        timestamp=person.timestamp,
        seconds=person.seconds,
        person_id=person.person_id,
        action="drinking_water",
        action_name="喝水",
        confidence=round(best[0], 4),
        evidence=best[2],
    )


def _score_phone_call(
    person: _PersonBox,
    objects: list[ObjectRecord],
    pose: PoseRecord | None,
) -> FrameActionScore | None:
    candidates = [obj for obj in objects if obj.class_name in PHONE_OBJECT_NAMES and _bbox_intersects(person.bbox_xyxy, obj.bbox_xyxy)]
    if not candidates:
        return None

    head = _head_point(person.bbox_xyxy, pose)
    person_scale = _bbox_diag(person.bbox_xyxy)
    best: tuple[float, ObjectRecord, str] | None = None
    for obj in candidates:
        center = _bbox_center(obj.bbox_xyxy)
        head_distance = _distance(center, head) / person_scale
        if head_distance > PHONE_HEAD_DISTANCE_THRESHOLD:
            continue
        score = _clamp(0.35 + obj.confidence * 0.5 + max(0.0, PHONE_HEAD_DISTANCE_THRESHOLD - head_distance), 0.0, 0.95)
        evidence = f"cell phone close to head; head_dist={head_distance:.2f}"
        if best is None or score > best[0]:
            best = (score, obj, evidence)

    if best is None:
        return None

    return FrameActionScore(
        frame_id=person.frame_id,
        timestamp=person.timestamp,
        seconds=person.seconds,
        person_id=person.person_id,
        action="talking_on_phone",
        action_name="打电话",
        confidence=round(best[0], 4),
        evidence=best[2],
    )


def _merge_frame_scores(
    frame_scores: list[FrameActionScore],
    min_event_frames: int,
    min_event_seconds: float,
    max_gap_frames: int,
) -> list[EventRecord]:
    grouped: dict[tuple[int, str], list[FrameActionScore]] = defaultdict(list)
    for score in frame_scores:
        grouped[(score.person_id, score.action)].append(score)

    events: list[EventRecord] = []
    for (_person_id, _action), scores in sorted(grouped.items()):
        ordered = sorted(scores, key=lambda item: item.frame_id)
        current: list[FrameActionScore] = []
        for score in ordered:
            if current and score.frame_id - current[-1].frame_id > max_gap_frames + 1:
                events.extend(_scores_to_event(current, min_event_frames, min_event_seconds))
                current = []
            current.append(score)
        events.extend(_scores_to_event(current, min_event_frames, min_event_seconds))

    return sorted(events, key=lambda event: (event.start_seconds, event.person_id, event.action))


def _scores_to_event(scores: list[FrameActionScore], min_event_frames: int, min_event_seconds: float) -> list[EventRecord]:
    if len(scores) < min_event_frames:
        return []
    first = scores[0]
    last = scores[-1]
    if last.seconds - first.seconds < min_event_seconds:
        return []
    confidence = sum(score.confidence for score in scores) / len(scores)
    evidence_counts: dict[str, int] = defaultdict(int)
    for score in scores:
        evidence_counts[score.evidence] += 1
    evidence = max(evidence_counts.items(), key=lambda item: item[1])[0]
    return [
        EventRecord(
            person_id=first.person_id,
            action=first.action,
            action_name=first.action_name,
            start_time=first.timestamp,
            end_time=last.timestamp,
            start_seconds=first.seconds,
            end_seconds=last.seconds,
            frame_count=len(scores),
            confidence=round(confidence, 4),
            evidence=evidence,
        )
    ]


def _draw_stage2_frame(frame: Any, scores: list[FrameActionScore]) -> Any:
    y = 32
    for score in scores:
        label = f"person {score.person_id}: {score.action_name} {score.confidence:.2f}"
        cv2.putText(frame, label, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 240, 20), 2, cv2.LINE_AA)
        y += 32
    return frame


def _write_outputs(output_dir: Path, result: Stage2Result, person_boxes_by_frame: dict[int, list[_PersonBox]]) -> None:
    _write_json(output_dir / "metadata.json", result.video.to_dict())
    _write_json(output_dir / "person_boxes.json", {str(frame): [asdict(box) for box in boxes] for frame, boxes in person_boxes_by_frame.items()})
    _write_json(output_dir / "objects.json", [asdict(record) for record in result.objects])
    _write_json(output_dir / "poses.json", [asdict(record) for record in result.poses])
    _write_json(output_dir / "frame_action_scores.json", [asdict(record) for record in result.frame_scores])
    _write_json(output_dir / "events.json", [asdict(record) for record in result.events])
    _write_json(output_dir / "summary.json", result.to_summary())
    _write_events_csv(output_dir / "events.csv", result.events)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _write_events_csv(path: Path, events: list[EventRecord]) -> None:
    fieldnames = [
        "person_id",
        "action",
        "action_name",
        "start_time",
        "end_time",
        "start_seconds",
        "end_seconds",
        "frame_count",
        "confidence",
        "evidence",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            writer.writerow(asdict(event))


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _bbox_diag(bbox: list[float]) -> float:
    width = max(1.0, bbox[2] - bbox[0])
    height = max(1.0, bbox[3] - bbox[1])
    return math.sqrt(width * width + height * height)


def _bbox_iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def _bbox_intersects(a: list[float], b: list[float]) -> bool:
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


def _bbox_contains(bbox: list[float], point: tuple[float, float]) -> bool:
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.dist(a, b)


def _head_point(person_bbox: list[float], pose: PoseRecord | None) -> tuple[float, float]:
    if pose and len(pose.keypoints) > 0 and pose.keypoints[0][2] >= KEYPOINT_CONFIDENCE:
        return (pose.keypoints[0][0], pose.keypoints[0][1])
    return ((person_bbox[0] + person_bbox[2]) / 2.0, person_bbox[1] + (person_bbox[3] - person_bbox[1]) * 0.16)


def _wrist_points(pose: PoseRecord | None) -> list[tuple[float, float]]:
    if pose is None or len(pose.keypoints) <= 10:
        return []
    wrists = []
    for index in (9, 10):
        x, y, conf = pose.keypoints[index]
        if conf >= KEYPOINT_CONFIDENCE:
            wrists.append((x, y))
    return wrists


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))
