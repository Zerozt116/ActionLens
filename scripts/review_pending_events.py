from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.fuse_stage2_vlm_events import fuse_from_files
from scripts.run_charades_clip_batch import find_event_review_paths


@dataclass(frozen=True)
class PendingReviewRecord:
    clip_id: str
    action: str
    stage2_event_index: int
    review_dir: str
    status: str
    present: bool
    fused_status: str
    error: str = ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run event-centered VLM reviews for pending events in a Charades clip batch.")
    parser.add_argument("--batch-root", type=Path, required=True, help="Batch root containing <clip_id>/fused_events.json.")
    parser.add_argument("--frame-count", type=int, default=6, help="Frames sampled per event review.")
    parser.add_argument("--event-context-seconds", type=float, default=1.0, help="Context around each pending event.")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct", help="VLM model.")
    parser.add_argument("--base-url", default="https://api.siliconflow.cn/v1/chat/completions", help="VLM chat completions URL.")
    parser.add_argument("--api-key-env", default="SILICONFLOW_API_KEY", help="Environment variable for the API key.")
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="Dotenv file passed to the VLM review script.")
    parser.add_argument("--dry-run", action="store_true", help="Only write review payloads, do not call VLM.")
    parser.add_argument("--summary-output", type=Path, default=None, help="Optional summary JSON output.")
    args = parser.parse_args()

    records = review_pending_events(
        batch_root=args.batch_root,
        frame_count=args.frame_count,
        event_context_seconds=args.event_context_seconds,
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        env_file=args.env_file,
        dry_run=args.dry_run,
    )
    summary = summarize(records)
    output = args.summary_output or args.batch_root / "pending_event_review_summary.json"
    output.write_text(
        json.dumps({"summary": summary, "records": [asdict(record) for record in records]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {output}")


def review_pending_events(
    batch_root: Path,
    frame_count: int,
    event_context_seconds: float,
    model: str,
    base_url: str,
    api_key_env: str,
    env_file: Path,
    dry_run: bool,
) -> list[PendingReviewRecord]:
    records: list[PendingReviewRecord] = []
    for fused_path in sorted(batch_root.glob("*/fused_events.json")):
        clip_dir = fused_path.parent
        fused = read_json(fused_path)
        pending_events = fused.get("pending_events", [])
        for event in pending_events:
            record = review_one_pending(
                clip_dir=clip_dir,
                event=event,
                frame_count=frame_count,
                event_context_seconds=event_context_seconds,
                model=model,
                base_url=base_url,
                api_key_env=api_key_env,
                env_file=env_file,
                dry_run=dry_run,
            )
            records.append(record)
    return records


def review_one_pending(
    clip_dir: Path,
    event: dict,
    frame_count: int,
    event_context_seconds: float,
    model: str,
    base_url: str,
    api_key_env: str,
    env_file: Path,
    dry_run: bool,
) -> PendingReviewRecord:
    event_index = int(event["stage2_event_index"])
    action = str(event["action"])
    review_dir = clip_dir / "event_reviews" / f"event_{event_index:03d}_{action}"
    review_dir.mkdir(parents=True, exist_ok=True)

    if not (review_dir / "vlm_summary.json").exists():
        command = [
            str(Path(sys.executable)),
            "scripts/review_video_with_vlm.py",
            clip_path_for(clip_dir),
            "-o",
            str(review_dir),
            "--model",
            model,
            "--base-url",
            base_url,
            "--api-key-env",
            api_key_env,
            "--env-file",
            str(env_file),
            "--frame-count",
            str(frame_count),
            "--events-json",
            str(clip_dir / "stage2" / "events.json"),
            "--event-index",
            str(event_index),
            "--event-context-seconds",
            str(event_context_seconds),
        ]
        if dry_run:
            command.append("--dry-run")
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            return PendingReviewRecord(
                clip_id=clip_dir.name,
                action=action,
                stage2_event_index=event_index,
                review_dir=str(review_dir),
                status="review_failed",
                present=False,
                fused_status="skipped",
                error=completed.stderr.strip() or completed.stdout.strip(),
            )

    present = read_review_present(review_dir / "vlm_summary.json")
    try:
        payload = fuse_from_files(
            comparison_path=clip_dir / "comparison.json",
            event_review_paths=find_event_review_paths(clip_dir),
        )
        (clip_dir / "fused_events.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        fused_status = "ok"
        error = ""
    except Exception as exc:
        fused_status = "fuse_failed"
        error = str(exc)

    return PendingReviewRecord(
        clip_id=clip_dir.name,
        action=action,
        stage2_event_index=event_index,
        review_dir=str(review_dir),
        status="completed",
        present=present,
        fused_status=fused_status,
        error=error,
    )


def clip_path_for(clip_dir: Path) -> str:
    summary = read_json(clip_dir.parent / "batch_summary.json")
    for record in summary.get("records", []):
        if record.get("clip_id") == clip_dir.name:
            return str(record["clip_path"])
    raise KeyError(f"clip path not found for {clip_dir.name}")


def read_review_present(summary_path: Path) -> bool:
    summary = read_json(summary_path)
    review = summary.get("review_json", {})
    return any(action.get("present") for action in review.get("actions", []))


def summarize(records: list[PendingReviewRecord]) -> dict[str, int]:
    return {
        "pending_events_found": len(records),
        "completed": sum(record.status == "completed" for record in records),
        "review_failed": sum(record.status == "review_failed" for record in records),
        "fused_ok": sum(record.fused_status == "ok" for record in records),
        "vlm_present": sum(record.present for record in records),
    }


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
