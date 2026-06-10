from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2

from .video_io import VideoMetadata, probe_video, seconds_to_timestamp


PERSON_CLASS_ID = 0


@dataclass(frozen=True)
class Observation:
    frame_id: int
    timestamp: str
    seconds: float
    bbox_xyxy: list[float]
    confidence: float


@dataclass(frozen=True)
class DetectionRecord:
    frame_id: int
    timestamp: str
    seconds: float
    track_id: int | None
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]


@dataclass(frozen=True)
class TrackRecord:
    person_id: int
    start_time: str
    end_time: str
    start_seconds: float
    end_seconds: float
    frame_count: int
    average_confidence: float
    observations: list[Observation]


@dataclass(frozen=True)
class Stage1Result:
    video: VideoMetadata
    detections: list[DetectionRecord]
    tracks: list[TrackRecord]
    annotated_video: str | None

    def to_summary(self) -> dict[str, Any]:
        return {
            "video": self.video.to_dict(),
            "num_detections": len(self.detections),
            "num_tracks": len(self.tracks),
            "tracks": [
                {
                    "person_id": track.person_id,
                    "start_time": track.start_time,
                    "end_time": track.end_time,
                    "frame_count": track.frame_count,
                    "average_confidence": track.average_confidence,
                }
                for track in self.tracks
            ],
            "annotated_video": self.annotated_video,
        }


def run_stage1(
    video_path: str | Path,
    output_dir: str | Path,
    model_name: str = "yolo11n.pt",
    confidence: float = 0.25,
    iou: float = 0.5,
    image_size: int = 640,
    device: str | None = None,
    write_annotated_video: bool = True,
) -> Stage1Result:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Run: ./.conda/bin/python -m pip install -r requirements.txt"
        ) from exc

    input_path = Path(video_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = probe_video(input_path)
    model = YOLO(model_name)

    detections: list[DetectionRecord] = []
    observations_by_track: dict[int, list[Observation]] = defaultdict(list)
    annotated_path = out_dir / "annotated.mp4" if write_annotated_video else None
    writer: cv2.VideoWriter | None = None

    track_kwargs: dict[str, Any] = {
        "source": str(input_path),
        "stream": True,
        "persist": True,
        "tracker": "bytetrack.yaml",
        "classes": [PERSON_CLASS_ID],
        "conf": confidence,
        "iou": iou,
        "imgsz": image_size,
        "verbose": False,
    }
    if device:
        track_kwargs["device"] = device

    try:
        for frame_id, result in enumerate(model.track(**track_kwargs)):
            seconds = frame_id / metadata.fps if metadata.fps > 0 else 0.0
            timestamp = seconds_to_timestamp(seconds)

            if writer is None and annotated_path is not None:
                frame = result.orig_img
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(annotated_path), fourcc, metadata.fps or 25.0, (width, height))

            if annotated_path is not None and writer is not None:
                writer.write(result.plot())

            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
            xyxy = boxes.xyxy.cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            classes = boxes.cls.int().cpu().tolist()

            for track_id, bbox, conf, class_id in zip(ids, xyxy, confs, classes, strict=True):
                if class_id != PERSON_CLASS_ID:
                    continue

                clean_bbox = [round(float(value), 3) for value in bbox]
                confidence_value = round(float(conf), 4)
                record = DetectionRecord(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    seconds=round(seconds, 3),
                    track_id=int(track_id) if track_id is not None else None,
                    class_id=class_id,
                    class_name="person",
                    confidence=confidence_value,
                    bbox_xyxy=clean_bbox,
                )
                detections.append(record)

                if track_id is not None:
                    observations_by_track[int(track_id)].append(
                        Observation(
                            frame_id=frame_id,
                            timestamp=timestamp,
                            seconds=round(seconds, 3),
                            bbox_xyxy=clean_bbox,
                            confidence=confidence_value,
                        )
                    )
    finally:
        if writer is not None:
            writer.release()

    tracks = _build_tracks(observations_by_track)
    result = Stage1Result(
        video=metadata,
        detections=detections,
        tracks=tracks,
        annotated_video=str(annotated_path) if annotated_path is not None else None,
    )
    _write_outputs(out_dir, result)
    return result


def _build_tracks(observations_by_track: dict[int, list[Observation]]) -> list[TrackRecord]:
    tracks: list[TrackRecord] = []
    for track_id, observations in sorted(observations_by_track.items()):
        ordered = sorted(observations, key=lambda obs: obs.frame_id)
        first = ordered[0]
        last = ordered[-1]
        avg_conf = sum(obs.confidence for obs in ordered) / len(ordered)
        tracks.append(
            TrackRecord(
                person_id=track_id,
                start_time=first.timestamp,
                end_time=last.timestamp,
                start_seconds=first.seconds,
                end_seconds=last.seconds,
                frame_count=len(ordered),
                average_confidence=round(avg_conf, 4),
                observations=ordered,
            )
        )
    return tracks


def _write_outputs(output_dir: Path, result: Stage1Result) -> None:
    _write_json(output_dir / "metadata.json", result.video.to_dict())
    _write_json(output_dir / "detections.json", [asdict(record) for record in result.detections])
    _write_json(output_dir / "tracks.json", [asdict(track) for track in result.tracks])
    _write_json(output_dir / "summary.json", result.to_summary())


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
