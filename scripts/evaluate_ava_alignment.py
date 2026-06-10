from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CLIP_RE = re.compile(
    r"(?P<video_id>.+)_t(?P<timestamp>\d+)_p(?P<person_id>\d+)_a(?P<action_id>\d+)_(?P<action>.+)"
)


@dataclass(frozen=True)
class AvaTarget:
    video_id: str
    timestamp: int
    person_id: int
    action_id: int
    action: str
    action_name: str
    action_type: str
    bbox_xyxy_norm: list[float]


@dataclass(frozen=True)
class AlignmentResult:
    clip_path: str
    stage2_output_dir: str
    target: AvaTarget
    center_frame: int
    frame_window: int
    best_iou: float
    best_frame_id: int | None
    best_stage2_person_id: int | None
    best_stage2_bbox_xyxy: list[float] | None
    ava_bbox_xyxy_pixels: list[float]
    has_pose_for_best_person: bool
    object_count_in_window: int
    event_count: int
    notes: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Align an AVA annotation row with stage2 detection outputs.")
    parser.add_argument("manifest", type=Path, help="AVA subset manifest CSV.")
    parser.add_argument("clip_path", type=Path, help="Downloaded AVA clip path.")
    parser.add_argument("stage2_output_dir", type=Path, help="Output directory produced by video-analyst stage2.")
    parser.add_argument("--seconds-before", type=float, default=2.0, help="Seconds before AVA timestamp used when creating the clip.")
    parser.add_argument("--frame-window", type=int, default=5, help="Frames around the center frame to search.")
    parser.add_argument("--output", type=Path, default=None, help="JSON output path.")
    args = parser.parse_args()

    result = evaluate_alignment(
        manifest=args.manifest,
        clip_path=args.clip_path,
        stage2_output_dir=args.stage2_output_dir,
        seconds_before=args.seconds_before,
        frame_window=args.frame_window,
    )
    payload = asdict(result)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    output = args.output or args.stage2_output_dir / "ava_alignment.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate_alignment(
    manifest: Path,
    clip_path: Path,
    stage2_output_dir: Path,
    seconds_before: float,
    frame_window: int,
) -> AlignmentResult:
    metadata = read_json(stage2_output_dir / "metadata.json")
    person_boxes = read_json(stage2_output_dir / "person_boxes.json")
    poses = read_json(stage2_output_dir / "poses.json")
    objects = read_json(stage2_output_dir / "objects.json")
    events = read_json(stage2_output_dir / "events.json")

    target = find_target(manifest, clip_path)
    width = float(metadata["width"])
    height = float(metadata["height"])
    fps = float(metadata["fps"])
    center_frame = int(round(seconds_before * fps))
    ava_bbox_pixels = [
        round(target.bbox_xyxy_norm[0] * width, 3),
        round(target.bbox_xyxy_norm[1] * height, 3),
        round(target.bbox_xyxy_norm[2] * width, 3),
        round(target.bbox_xyxy_norm[3] * height, 3),
    ]

    search_frames = range(max(0, center_frame - frame_window), center_frame + frame_window + 1)
    best_iou = 0.0
    best_frame_id: int | None = None
    best_person_id: int | None = None
    best_bbox: list[float] | None = None
    for frame_id in search_frames:
        for box in person_boxes.get(str(frame_id), []):
            score = bbox_iou(ava_bbox_pixels, box["bbox_xyxy"])
            if score > best_iou:
                best_iou = score
                best_frame_id = frame_id
                best_person_id = int(box["person_id"])
                best_bbox = box["bbox_xyxy"]

    pose_keys = {(int(pose["frame_id"]), int(pose["person_id"])) for pose in poses}
    has_pose = best_frame_id is not None and best_person_id is not None and (best_frame_id, best_person_id) in pose_keys
    object_count = sum(1 for obj in objects if abs(int(obj["frame_id"]) - center_frame) <= frame_window)
    notes = "matched" if best_iou >= 0.3 else "low_iou_or_missing_person_match"

    return AlignmentResult(
        clip_path=str(clip_path),
        stage2_output_dir=str(stage2_output_dir),
        target=target,
        center_frame=center_frame,
        frame_window=frame_window,
        best_iou=round(best_iou, 4),
        best_frame_id=best_frame_id,
        best_stage2_person_id=best_person_id,
        best_stage2_bbox_xyxy=best_bbox,
        ava_bbox_xyxy_pixels=ava_bbox_pixels,
        has_pose_for_best_person=has_pose,
        object_count_in_window=object_count,
        event_count=len(events),
        notes=notes,
    )


def find_target(manifest: Path, clip_path: Path) -> AvaTarget:
    parsed = parse_clip_name(clip_path)
    with manifest.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if (
                row["video_id"] == parsed["video_id"]
                and int(float(row["timestamp"])) == parsed["timestamp"]
                and int(row["person_id"]) == parsed["person_id"]
                and int(row["action_id"]) == parsed["action_id"]
            ):
                return AvaTarget(
                    video_id=row["video_id"],
                    timestamp=int(float(row["timestamp"])),
                    person_id=int(row["person_id"]),
                    action_id=int(row["action_id"]),
                    action=row["action"],
                    action_name=row["action_name"],
                    action_type=row["action_type"],
                    bbox_xyxy_norm=[float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])],
                )
    raise ValueError(f"No AVA manifest row matches clip: {clip_path.name}")


def parse_clip_name(clip_path: Path) -> dict[str, int | str]:
    match = CLIP_RE.match(clip_path.stem)
    if match is None:
        raise ValueError(f"Clip filename does not match AVA naming convention: {clip_path.name}")
    return {
        "video_id": match.group("video_id"),
        "timestamp": int(match.group("timestamp")),
        "person_id": int(match.group("person_id")),
        "action_id": int(match.group("action_id")),
        "action": match.group("action"),
    }


def bbox_iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
