from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import run_stage1
from .stage2 import run_stage2
from .video_io import probe_video


def main() -> None:
    parser = argparse.ArgumentParser(prog="video-analyst")
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe_parser = subparsers.add_parser("probe", help="Read video metadata.")
    probe_parser.add_argument("video", type=Path)

    analyze_parser = subparsers.add_parser("stage1", help="Run person detection and ByteTrack tracking.")
    analyze_parser.add_argument("video", type=Path)
    analyze_parser.add_argument("-o", "--output-dir", type=Path, default=Path("outputs/stage1"))
    analyze_parser.add_argument("--model", default="yolo11n.pt")
    analyze_parser.add_argument("--conf", type=float, default=0.25)
    analyze_parser.add_argument("--iou", type=float, default=0.5)
    analyze_parser.add_argument("--imgsz", type=int, default=640)
    analyze_parser.add_argument("--device", default=None)
    analyze_parser.add_argument("--no-annotated-video", action="store_true")

    stage2_parser = subparsers.add_parser("stage2", help="Run object detection, pose estimation, and rule-based behavior events.")
    stage2_parser.add_argument("video", type=Path)
    stage2_parser.add_argument("-o", "--output-dir", type=Path, default=Path("outputs/stage2"))
    stage2_parser.add_argument("--det-model", default="yolo11n.pt")
    stage2_parser.add_argument("--pose-model", default="yolo11n-pose.pt")
    stage2_parser.add_argument("--conf", type=float, default=0.25)
    stage2_parser.add_argument("--iou", type=float, default=0.5)
    stage2_parser.add_argument("--imgsz", type=int, default=640)
    stage2_parser.add_argument("--device", default=None)
    stage2_parser.add_argument("--min-event-frames", type=int, default=3)
    stage2_parser.add_argument("--min-event-seconds", type=float, default=0.3)
    stage2_parser.add_argument("--max-gap-frames", type=int, default=6)
    stage2_parser.add_argument("--no-annotated-video", action="store_true")

    args = parser.parse_args()

    if args.command == "probe":
        metadata = probe_video(args.video)
        print(json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2))
        return

    if args.command == "stage2":
        result = run_stage2(
            video_path=args.video,
            output_dir=args.output_dir,
            detection_model_name=args.det_model,
            pose_model_name=args.pose_model,
            confidence=args.conf,
            iou=args.iou,
            image_size=args.imgsz,
            device=args.device,
            min_event_frames=args.min_event_frames,
            min_event_seconds=args.min_event_seconds,
            max_gap_frames=args.max_gap_frames,
            write_annotated_video=not args.no_annotated_video,
        )
        print(json.dumps(result.to_summary(), ensure_ascii=False, indent=2))
        return

    result = run_stage1(
        video_path=args.video,
        output_dir=args.output_dir,
        model_name=args.model,
        confidence=args.conf,
        iou=args.iou,
        image_size=args.imgsz,
        device=args.device,
        write_annotated_video=not args.no_annotated_video,
    )
    print(json.dumps(result.to_summary(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
