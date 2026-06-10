from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.download_ava_clips import build_download_command, read_clip_requests, require_tool
from scripts.evaluate_ava_alignment import evaluate_alignment


@dataclass(frozen=True)
class BatchEvalRecord:
    clip_path: str
    stage2_output_dir: str
    video_id: str
    timestamp: int
    ava_person_id: int
    action_id: int
    action: str
    action_name: str
    action_type: str
    best_iou: float
    matched: bool
    has_pose_for_best_person: bool
    object_count_in_window: int
    event_count: int
    status: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Download AVA clips, run stage2, and summarize AVA alignment metrics.")
    parser.add_argument("manifest", type=Path, help="AVA subset manifest CSV.")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--clips-dir", type=Path, default=Path("data/ava/clips"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/ava_batch_eval"))
    parser.add_argument("--seconds-before", type=float, default=2.0)
    parser.add_argument("--seconds-after", type=float, default=2.0)
    parser.add_argument("--frame-window", type=int, default=5)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-stage2", action="store_true")
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args()

    records = run_batch_eval(
        manifest=args.manifest,
        limit=args.limit,
        clips_dir=args.clips_dir,
        output_root=args.output_root,
        seconds_before=args.seconds_before,
        seconds_after=args.seconds_after,
        frame_window=args.frame_window,
        skip_download=args.skip_download,
        skip_stage2=args.skip_stage2,
    )
    summary = summarize_records(records)
    payload = {"summary": summary, "records": [asdict(record) for record in records]}

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_json = args.summary_json or args.output_root / "batch_summary.json"
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_csv(args.output_root / "batch_summary.csv", records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_batch_eval(
    manifest: Path,
    limit: int,
    clips_dir: Path,
    output_root: Path,
    seconds_before: float,
    seconds_after: float,
    frame_window: int,
    skip_download: bool,
    skip_stage2: bool,
) -> list[BatchEvalRecord]:
    requests = read_clip_requests(manifest, limit)
    clips_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    yt_dlp = require_tool("yt-dlp")
    ffmpeg = require_tool("ffmpeg")
    video_analyst = require_tool("video-analyst")
    env = build_tool_env(yt_dlp, ffmpeg, video_analyst)

    records: list[BatchEvalRecord] = []
    for request in requests:
        clip_path = find_existing_clip(clips_dir, request.file_stem)
        if clip_path is None and not skip_download:
            command = build_download_command(request, clips_dir, seconds_before, seconds_after)
            command[0] = str(yt_dlp)
            print(" ".join(command))
            subprocess.run(command, check=False, env=env)
            clip_path = find_existing_clip(clips_dir, request.file_stem)

        stage2_dir = output_root / request.file_stem
        status = "ok"
        if clip_path is None:
            status = "download_failed"
            records.append(empty_record(request, stage2_dir, status))
            continue

        if not skip_stage2 or not (stage2_dir / "summary.json").exists():
            command = [
                str(video_analyst),
                "stage2",
                str(clip_path),
                "-o",
                str(stage2_dir),
                "--det-model",
                "yolo11n.pt",
                "--pose-model",
                "yolo11n-pose.pt",
                "--conf",
                "0.25",
                "--imgsz",
                "640",
            ]
            print(" ".join(command))
            completed = subprocess.run(command, check=False, env=env)
            if completed.returncode != 0:
                status = "stage2_failed"
                records.append(empty_record(request, stage2_dir, status, clip_path))
                continue

        try:
            alignment = evaluate_alignment(
                manifest=manifest,
                clip_path=clip_path,
                stage2_output_dir=stage2_dir,
                seconds_before=seconds_before,
                frame_window=frame_window,
            )
            (stage2_dir / "ava_alignment.json").write_text(
                json.dumps(asdict(alignment), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            records.append(
                BatchEvalRecord(
                    clip_path=str(clip_path),
                    stage2_output_dir=str(stage2_dir),
                    video_id=alignment.target.video_id,
                    timestamp=alignment.target.timestamp,
                    ava_person_id=alignment.target.person_id,
                    action_id=alignment.target.action_id,
                    action=alignment.target.action,
                    action_name=alignment.target.action_name,
                    action_type=alignment.target.action_type,
                    best_iou=alignment.best_iou,
                    matched=alignment.best_iou >= 0.3,
                    has_pose_for_best_person=alignment.has_pose_for_best_person,
                    object_count_in_window=alignment.object_count_in_window,
                    event_count=alignment.event_count,
                    status=status,
                )
            )
        except Exception:
            records.append(empty_record(request, stage2_dir, "alignment_failed", clip_path))

    return records


def summarize_records(records: list[BatchEvalRecord]) -> dict[str, float | int]:
    total = len(records)
    ok_records = [record for record in records if record.status == "ok"]
    matched = [record for record in ok_records if record.matched]
    with_pose = [record for record in matched if record.has_pose_for_best_person]
    with_objects = [record for record in ok_records if record.object_count_in_window > 0]
    with_events = [record for record in ok_records if record.event_count > 0]
    return {
        "total": total,
        "ok": len(ok_records),
        "download_failed": sum(record.status == "download_failed" for record in records),
        "stage2_failed": sum(record.status == "stage2_failed" for record in records),
        "alignment_failed": sum(record.status == "alignment_failed" for record in records),
        "matched": len(matched),
        "match_rate": round(len(matched) / len(ok_records), 4) if ok_records else 0.0,
        "average_best_iou": round(sum(record.best_iou for record in ok_records) / len(ok_records), 4) if ok_records else 0.0,
        "pose_on_matched_rate": round(len(with_pose) / len(matched), 4) if matched else 0.0,
        "object_detected_rate": round(len(with_objects) / len(ok_records), 4) if ok_records else 0.0,
        "event_detected_rate": round(len(with_events) / len(ok_records), 4) if ok_records else 0.0,
    }


def write_summary_csv(path: Path, records: list[BatchEvalRecord]) -> None:
    import csv

    fieldnames = list(BatchEvalRecord.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def empty_record(request, stage2_dir: Path, status: str, clip_path: Path | None = None) -> BatchEvalRecord:
    return BatchEvalRecord(
        clip_path=str(clip_path) if clip_path is not None else "",
        stage2_output_dir=str(stage2_dir),
        video_id=request.video_id,
        timestamp=request.timestamp,
        ava_person_id=request.person_id,
        action_id=request.action_id,
        action=request.action,
        action_name="",
        action_type="",
        best_iou=0.0,
        matched=False,
        has_pose_for_best_person=False,
        object_count_in_window=0,
        event_count=0,
        status=status,
    )


def find_existing_clip(clips_dir: Path, file_stem: str) -> Path | None:
    matches = sorted(clips_dir.glob(f"{file_stem}.*"))
    return matches[0] if matches else None


def build_tool_env(*tools: Path) -> dict[str, str]:
    env = os.environ.copy()
    tool_dirs = [str(tool.parent) for tool in tools]
    env["PATH"] = os.pathsep.join(tool_dirs + [env.get("PATH", "")])
    return env


def require_tool(name: str) -> Path:
    resolved = shutil.which(name)
    if resolved is not None:
        return Path(resolved)

    candidate = Path(sys.executable).parent / name
    if candidate.exists():
        return candidate

    raise RuntimeError(f"Required command not found: {name}")


if __name__ == "__main__":
    main()
