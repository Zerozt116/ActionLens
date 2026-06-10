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


from scripts.compare_charades_stage2_vlm import CHARADES_TO_CANONICAL
from scripts.fuse_stage2_vlm_events import fuse_from_files


@dataclass(frozen=True)
class ClipRecord:
    clip_id: str
    video_id: str
    action_id: str
    action_name: str
    canonical_action: str
    clip_path: str
    stage2_dir: str
    vlm_dir: str | None
    comparison_path: str | None
    fused_path: str | None
    stage2_status: str
    vlm_status: str
    fused_status: str
    stage2_event_count: int
    vlm_present: bool
    error: str = ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stage2 + VLM + comparison + fusion on a batch of sliced Charades clips."
    )
    parser.add_argument("clips_manifest", type=Path, help="Output CSV from slice_charades_clips.py.")
    parser.add_argument(
        "--charades-manifest",
        type=Path,
        default=Path("data/charades/charades_train_pilot_actions.csv"),
        help="Charades actions manifest (used by compare script to look up intervals).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/charades_clip_batch"),
        help="Root directory for per-clip subdirectories.",
    )
    parser.add_argument(
        "--vlm-model",
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="SiliconFlow VLM model name.",
    )
    parser.add_argument(
        "--vlm-base-url",
        default="https://api.siliconflow.cn/v1/chat/completions",
        help="SiliconFlow chat completions URL.",
    )
    parser.add_argument(
        "--api-key-env",
        default="SILICONFLOW_API_KEY",
        help="Environment variable holding the SiliconFlow API key.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Optional .env file to load before reading the API key.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum clips to process.")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N rows.")
    parser.add_argument("--skip-stage2", action="store_true", help="Reuse existing stage2 outputs.")
    parser.add_argument("--skip-vlm", action="store_true", help="Skip the VLM review step.")
    parser.add_argument("--skip-fuse", action="store_true", help="Skip the fusion step.")
    parser.add_argument("--frame-count", type=int, default=6, help="Frames sampled per VLM review.")
    parser.add_argument(
        "--actions",
        nargs="+",
        default=None,
        help="Override the list of actions to send to the VLM (defaults to the clip's canonical action).",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional explicit path for the batch summary JSON.",
    )
    args = parser.parse_args()

    records = run_batch(
        clips_manifest=args.clips_manifest,
        charades_manifest=args.charades_manifest,
        output_root=args.output_root,
        vlm_model=args.vlm_model,
        vlm_base_url=args.vlm_base_url,
        api_key_env=args.api_key_env,
        env_file=args.env_file,
        limit=args.limit,
        offset=args.offset,
        skip_stage2=args.skip_stage2,
        skip_vlm=args.skip_vlm,
        skip_fuse=args.skip_fuse,
        frame_count=args.frame_count,
        actions_override=args.actions,
    )

    summary = summarize_records(records)
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_json = args.summary_json or args.output_root / "batch_summary.json"
    summary_json.write_text(
        json.dumps({"summary": summary, "records": [asdict(record) for record in records]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_records_csv(args.output_root / "batch_summary.csv", records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_batch(
    clips_manifest: Path,
    charades_manifest: Path,
    output_root: Path,
    vlm_model: str,
    vlm_base_url: str,
    api_key_env: str,
    env_file: Path,
    limit: int | None,
    offset: int,
    skip_stage2: bool,
    skip_vlm: bool,
    skip_fuse: bool,
    frame_count: int,
    actions_override: list[str] | None,
) -> list[ClipRecord]:
    clips = read_clips(clips_manifest, offset=offset, limit=limit)
    output_root.mkdir(parents=True, exist_ok=True)

    ffmpeg = require_tool("ffmpeg")
    video_analyst = require_tool("video-analyst")
    python = Path(sys.executable)

    env = build_tool_env(ffmpeg, video_analyst, python)
    load_env_file(env_file, env)

    has_api_key = bool(env.get(api_key_env))

    records: list[ClipRecord] = []
    for clip in clips:
        clip_dir = output_root / clip["clip_id"]
        stage2_dir = clip_dir / "stage2"
        vlm_dir = clip_dir / "vlm_review"
        comparison_path = clip_dir / "comparison.json"
        fused_path = clip_dir / "fused_events.json"

        canonical = CHARADES_TO_CANONICAL.get(clip["action_id"], clip["action_id"])
        actions = actions_override or [canonical]

        stage2_status, stage2_event_count, stage2_error = run_stage2(
            clip=clip,
            stage2_dir=stage2_dir,
            video_analyst=video_analyst,
            env=env,
            skip=skip_stage2,
        )

        vlm_status, vlm_present, vlm_error = "skipped", False, ""
        comparison_path_str: str | None = None
        fused_path_str: str | None = None
        fused_status = "skipped"

        if not skip_vlm and stage2_status in {"ok", "exists"} and not stage2_error:
            vlm_status, vlm_present, vlm_error = run_vlm(
                clip=clip,
                vlm_dir=vlm_dir,
                vlm_model=vlm_model,
                vlm_base_url=vlm_base_url,
                api_key_env=api_key_env,
                has_api_key=has_api_key,
                python=python,
                env=env,
                actions=actions,
                frame_count=frame_count,
            )

            if vlm_status in {"completed", "dry_run", "exists"}:
                try:
                    run_compare(
                        charades_manifest=charades_manifest,
                        video_id=clip["video_id"],
                        stage2_dir=stage2_dir,
                        vlm_dir=vlm_dir,
                        comparison_path=comparison_path,
                        stage2_time_offset_seconds=float(clip.get("clip_start_seconds", 0.0) or 0.0),
                        python=python,
                        env=env,
                    )
                    comparison_path_str = str(comparison_path)
                except Exception as exc:
                    vlm_error = (vlm_error + "; compare failed: " + str(exc)).strip("; ")

        if not skip_fuse and comparison_path is not None and comparison_path.exists():
            try:
                payload = fuse_from_files(
                    comparison_path=comparison_path,
                    event_review_paths=find_event_review_paths(clip_dir),
                )
                fused_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                fused_path_str = str(fused_path)
                fused_status = "ok"
            except Exception as exc:
                fused_status = "fuse_failed"
                stage2_error = (stage2_error + "; fuse failed: " + str(exc)).strip("; ")

        records.append(
            ClipRecord(
                clip_id=clip["clip_id"],
                video_id=clip["video_id"],
                action_id=clip["action_id"],
                action_name=clip["action_name"],
                canonical_action=canonical,
                clip_path=clip["clip_path"],
                stage2_dir=str(stage2_dir),
                vlm_dir=str(vlm_dir) if vlm_dir.exists() else None,
                comparison_path=comparison_path_str,
                fused_path=fused_path_str,
                stage2_status=stage2_status,
                vlm_status=vlm_status,
                fused_status=fused_status,
                stage2_event_count=stage2_event_count,
                vlm_present=vlm_present,
                error=combine_errors(stage2_error, vlm_error),
            )
        )

    return records


def run_stage2(
    clip: dict[str, str],
    stage2_dir: Path,
    video_analyst: Path,
    env: dict[str, str],
    skip: bool,
) -> tuple[str, int, str]:
    summary_path = stage2_dir / "summary.json"
    events_path = stage2_dir / "events.json"
    if skip and summary_path.exists() and events_path.exists():
        try:
            event_count = len(json.loads(events_path.read_text(encoding="utf-8")))
            return "exists", event_count, ""
        except Exception:
            pass

    stage2_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(video_analyst),
        "stage2",
        clip["clip_path"],
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
        "--no-annotated-video",
    ]
    print(" ".join(command))
    completed = subprocess.run(command, check=False, env=env, capture_output=True, text=True)
    if completed.returncode != 0:
        return "stage2_failed", 0, completed.stderr.strip() or completed.stdout.strip()

    if not events_path.exists():
        return "ok", 0, ""

    try:
        event_count = len(json.loads(events_path.read_text(encoding="utf-8")))
    except Exception as exc:
        return "ok", 0, f"events.json unreadable: {exc}"

    return "ok", event_count, ""


def run_vlm(
    clip: dict[str, str],
    vlm_dir: Path,
    vlm_model: str,
    vlm_base_url: str,
    api_key_env: str,
    has_api_key: bool,
    python: Path,
    env: dict[str, str],
    actions: list[str],
    frame_count: int,
) -> tuple[str, bool, str]:
    review_json = vlm_dir / "vlm_review.json"
    if review_json.exists():
        try:
            payload = json.loads(review_json.read_text(encoding="utf-8"))
            present = any(action.get("present") for action in payload.get("actions", []))
            return "exists", present, ""
        except Exception:
            pass

    vlm_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(python),
        "scripts/review_video_with_vlm.py",
        clip["clip_path"],
        "-o",
        str(vlm_dir),
        "--model",
        vlm_model,
        "--base-url",
        vlm_base_url,
        "--api-key-env",
        api_key_env,
        "--frame-count",
        str(frame_count),
        "--actions",
        *actions,
    ]
    if not has_api_key:
        command.append("--dry-run")

    print(" ".join(command))
    completed = subprocess.run(command, check=False, env=env, capture_output=True, text=True)
    summary_path = vlm_dir / "vlm_summary.json"
    if not summary_path.exists():
        return "vlm_failed", False, completed.stderr.strip() or completed.stdout.strip()

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return "vlm_failed", False, f"summary unreadable: {exc}"

    status = summary.get("status", "unknown")
    review = summary.get("review_json", {})
    present = any(action.get("present") for action in review.get("actions", []))
    if status == "failed" and completed.returncode != 0:
        return "vlm_failed", present, summary.get("body", "") or summary.get("reason", "")
    return status or "completed", present, ""


def run_compare(
    charades_manifest: Path,
    video_id: str,
    stage2_dir: Path,
    vlm_dir: Path,
    comparison_path: Path,
    stage2_time_offset_seconds: float,
    python: Path,
    env: dict[str, str],
) -> None:
    vlm_review = vlm_dir / "vlm_review.json"
    if not vlm_review.exists():
        return
    command = [
        str(python),
        "scripts/compare_charades_stage2_vlm.py",
        "--manifest",
        str(charades_manifest),
        "--video-id",
        video_id,
        "--stage2-dir",
        str(stage2_dir),
        "--vlm-review",
        str(vlm_review),
        "--output",
        str(comparison_path),
        "--stage2-time-offset-seconds",
        str(stage2_time_offset_seconds),
    ]
    print(" ".join(command))
    subprocess.run(command, check=True, env=env, capture_output=True, text=True)


def read_clips(path: Path, offset: int, limit: int | None) -> list[dict[str, str]]:
    import csv

    if not path.exists():
        raise FileNotFoundError(f"Clips manifest not found: {path}")
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for index, row in enumerate(reader):
            if index < offset:
                continue
            if row.get("status") not in {"sliced", "exists"}:
                continue
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def summarize_records(records: list[ClipRecord]) -> dict[str, float | int]:
    total = len(records)
    by_canonical: dict[str, int] = {}
    for record in records:
        by_canonical[record.canonical_action] = by_canonical.get(record.canonical_action, 0) + 1

    stage2_ok = [r for r in records if r.stage2_status in {"ok", "exists"}]
    vlm_ok = [r for r in records if r.vlm_status in {"completed", "exists", "dry_run"}]
    fused_ok = [r for r in records if r.fused_status == "ok"]
    stage2_event_hits = sum(1 for r in stage2_ok if r.stage2_event_count > 0)
    vlm_present_hits = sum(1 for r in vlm_ok if r.vlm_present)
    return {
        "total_clips": total,
        "stage2_ok": len(stage2_ok),
        "stage2_event_hits": stage2_event_hits,
        "vlm_ok": len(vlm_ok),
        "vlm_present_hits": vlm_present_hits,
        "fused_ok": len(fused_ok),
        "by_canonical_action": by_canonical,
        "stage2_event_rate": round(stage2_event_hits / len(stage2_ok), 4) if stage2_ok else 0.0,
        "vlm_present_rate": round(vlm_present_hits / len(vlm_ok), 4) if vlm_ok else 0.0,
    }


def combine_errors(*errors: str) -> str:
    return "; ".join(error for error in errors if error)


def find_event_review_paths(clip_dir: Path) -> list[Path]:
    reviews_dir = clip_dir / "event_reviews"
    if not reviews_dir.exists():
        return []
    return sorted(path for path in reviews_dir.iterdir() if path.is_dir() or path.name == "vlm_summary.json")


def write_records_csv(path: Path, records: list[ClipRecord]) -> None:
    import csv

    fieldnames = list(ClipRecord.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def build_tool_env(*tools: Path) -> dict[str, str]:
    env = os.environ.copy()
    tool_dirs = {str(tool.parent) for tool in tools}
    env["PATH"] = os.pathsep.join([*tool_dirs, env.get("PATH", "")])
    return env


def load_env_file(path: Path, env: dict[str, str]) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in env:
            env[key] = value


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
