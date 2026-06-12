"""Run VLM reviews for selected TAL proposal windows."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


DEFAULT_OUTCOMES = ("gt_hit_by_tal_only", "tal_only_non_gt")
DEFAULT_TOP_K_PER_CLIP = 3
DEFAULT_MIN_SCORE = 0.04
DEFAULT_MIN_DURATION = 0.5
DEFAULT_MAX_DURATION = 30.0

CHARADES_TO_CANONICAL = {
    "c015": "holding_phone",
    "c016": "looking_at_phone",
    "c017": "putting_phone",
    "c018": "taking_phone",
    "c019": "talking_on_phone",
    "c051": "watching_laptop",
    "c052": "using_laptop",
    "c059": "sitting_on_chair",
    "c065": "eating",
    "c097": "walking_through_doorway",
    "c106": "drinking_water",
    "c107": "holding_drink_container",
    "c108": "pouring_drink_container",
    "c109": "putting_drink_container",
    "c110": "taking_drink_container",
    "c123": "sitting_on_sofa",
    "c125": "sitting_on_floor",
    "c129": "taking_medicine",
    "c132": "watching_tv",
    "c147": "cooking",
    "c150": "running",
    "c151": "sitting_down",
    "c154": "standing_up",
    "c156": "eating",
}


@dataclass(frozen=True)
class TalReviewRecord:
    clip_id: str
    video_id: str
    action: str
    action_name: str
    outcome: str
    tal_score: float
    tal_iou: float
    start_seconds: float
    end_seconds: float
    review_dir: str
    status: str
    present: bool
    confidence: float
    decision: str
    evidence: str
    error: str = ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Review selected TAL proposals with the existing VLM reviewer.")
    parser.add_argument(
        "--ab-report",
        type=Path,
        default=Path("outputs/tal_stage2_vlm_ab_epoch034.json"),
        help="A/B report JSON produced by compare_tal_stage2_vlm.py.",
    )
    parser.add_argument(
        "--batch-summary",
        type=Path,
        default=Path("outputs/charades_clip_batch_50/batch_summary.json"),
        help="Batch summary containing clip_id -> clip_path records.",
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=Path("outputs/actionformer_epoch034_val_predictions.csv"),
        help="Full TAL prediction CSV used to compute rank_in_clip.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/tal_vlm_review_epoch034"),
        help="Root output directory for TAL proposal VLM reviews.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/tal_vlm_review_epoch034_summary.json"),
        help="Summary JSON output.",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=Path("outputs/tal_vlm_review_epoch034_summary.md"),
        help="Markdown summary output.",
    )
    parser.add_argument(
        "--fusion-output",
        type=Path,
        default=Path("outputs/tal_vlm_fusion_epoch034.json"),
        help="Output fused TAL+VLM events JSON.",
    )
    parser.add_argument(
        "--fusion-report-md",
        type=Path,
        default=Path("outputs/tal_vlm_fusion_epoch034.md"),
        help="Output fused TAL+VLM events Markdown.",
    )
    parser.add_argument(
        "--outcomes",
        nargs="+",
        default=list(DEFAULT_OUTCOMES),
        help="A/B row outcomes to select for review.",
    )
    parser.add_argument("--limit", type=int, default=5, help="Maximum proposals to review.")
    parser.add_argument(
        "--top-k-per-clip",
        type=int,
        default=DEFAULT_TOP_K_PER_CLIP,
        help="Keep only proposals whose full TAL rank within the clip is <= this value. Use 0 to disable.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help="Minimum TAL score to review.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=DEFAULT_MIN_DURATION,
        help="Minimum TAL segment duration in seconds.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=DEFAULT_MAX_DURATION,
        help="Maximum TAL segment duration in seconds.",
    )
    parser.add_argument("--context-seconds", type=float, default=1.0, help="Context to add around each TAL segment.")
    parser.add_argument("--frame-count", type=int, default=6, help="Frames sampled per review.")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct", help="VLM model.")
    parser.add_argument("--base-url", default="https://api.siliconflow.cn/v1/chat/completions", help="VLM endpoint.")
    parser.add_argument("--api-key-env", default="SILICONFLOW_API_KEY", help="API key environment variable.")
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="Dotenv file passed to VLM reviewer.")
    parser.add_argument("--dry-run", action="store_true", help="Extract frames and payloads without calling VLM.")
    parser.add_argument("--overwrite", action="store_true", help="Re-run existing reviews.")
    args = parser.parse_args()

    clip_paths = load_clip_paths(args.batch_summary)
    candidates, selection_summary = select_candidates(
        args.ab_report,
        predictions_csv=args.predictions_csv,
        outcomes=set(args.outcomes),
        limit=args.limit,
        top_k_per_clip=args.top_k_per_clip,
        min_score=args.min_score,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )
    records = [
        review_candidate(
            row=candidate,
            clip_paths=clip_paths,
            output_root=args.output_root,
            context_seconds=args.context_seconds,
            frame_count=args.frame_count,
            model=args.model,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            env_file=args.env_file,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
        for candidate in candidates
    ]
    payload = {
        "summary": {**selection_summary, **summarize(records)},
        "records": [asdict(record) for record in records],
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report_md.write_text(render_markdown(payload), encoding="utf-8")
    fusion_payload = build_fusion_payload(records)
    args.fusion_output.write_text(json.dumps(fusion_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.fusion_report_md.write_text(render_fusion_markdown(fusion_payload), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {args.summary_output}")
    print(f"wrote {args.report_md}")
    print(f"wrote {args.fusion_output}")
    print(f"wrote {args.fusion_report_md}")


def load_clip_paths(batch_summary: Path) -> dict[str, str]:
    payload = json.loads(batch_summary.read_text(encoding="utf-8"))
    return {
        record["clip_id"]: record["clip_path"]
        for record in payload.get("records", [])
        if record.get("clip_id") and record.get("clip_path")
    }


def select_candidates(
    path: Path,
    *,
    predictions_csv: Path,
    outcomes: set[str],
    limit: int,
    top_k_per_clip: int,
    min_score: float,
    min_duration: float,
    max_duration: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ranks = load_ranks(predictions_csv)
    rows = [
        add_proposal_metrics(row, ranks)
        for row in payload.get("rows", [])
        if row.get("outcome") in outcomes and row.get("tal_present") and row.get("tal_best_segment")
    ]
    candidates_before_limit = [row for row in rows if passes_filters(row, top_k_per_clip, min_score, min_duration, max_duration)]
    rows.sort(
        key=lambda row: (
            row.get("outcome") != "gt_hit_by_tal_only",
            -float(row.get("tal_best_iou", 0.0)),
            -float(row.get("tal_best_score", 0.0)),
        )
    )
    candidates_before_limit.sort(
        key=lambda row: (
            row.get("outcome") != "gt_hit_by_tal_only",
            -float(row.get("tal_best_iou", 0.0)),
            -float(row.get("tal_best_score", 0.0)),
        )
    )
    selected = candidates_before_limit[:limit]
    selection_summary = summarize_selection(
        all_candidates=rows,
        filtered_candidates=candidates_before_limit,
        selected=selected,
        outcomes=outcomes,
        top_k_per_clip=top_k_per_clip,
        min_score=min_score,
        min_duration=min_duration,
        max_duration=max_duration,
        limit=limit,
    )
    return selected, selection_summary


def load_ranks(path: Path) -> dict[tuple[str, str, float], int]:
    by_clip: dict[str, list[dict[str, str]]] = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            by_clip.setdefault(row["clip_id"], []).append(row)

    ranks: dict[tuple[str, str, float], int] = {}
    for clip_id, rows in by_clip.items():
        rows.sort(key=lambda row: float(row["score"]), reverse=True)
        for index, row in enumerate(rows, start=1):
            action = CHARADES_TO_CANONICAL.get(row["charades_action_id"], row["charades_action_id"])
            ranks[(clip_id, action, round(float(row["score"]), 8))] = index
    return ranks


def add_proposal_metrics(row: dict[str, Any], ranks: dict[tuple[str, str, float], int]) -> dict[str, Any]:
    proposal = dict(row)
    segment = proposal.get("tal_best_segment", [])
    duration = 0.0
    if len(segment) == 2:
        duration = max(0.0, float(segment[1]) - float(segment[0]))
    score = round(float(proposal.get("tal_best_score", 0.0)), 8)
    proposal["tal_duration"] = round(duration, 4)
    proposal["rank_in_clip"] = ranks.get((proposal["clip_id"], proposal["action"], score))
    return proposal


def passes_filters(
    row: dict[str, Any],
    top_k_per_clip: int,
    min_score: float,
    min_duration: float,
    max_duration: float,
) -> bool:
    rank = row.get("rank_in_clip")
    if top_k_per_clip > 0 and (rank is None or int(rank) > top_k_per_clip):
        return False
    score = float(row.get("tal_best_score", 0.0))
    duration = float(row.get("tal_duration", 0.0))
    return score >= min_score and min_duration <= duration <= max_duration


def summarize_selection(
    *,
    all_candidates: list[dict[str, Any]],
    filtered_candidates: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    outcomes: set[str],
    top_k_per_clip: int,
    min_score: float,
    min_duration: float,
    max_duration: float,
    limit: int,
) -> dict[str, Any]:
    return {
        "selection_outcomes": sorted(outcomes),
        "filter_top_k_per_clip": top_k_per_clip,
        "filter_min_score": min_score,
        "filter_min_duration": min_duration,
        "filter_max_duration": max_duration,
        "limit": limit,
        "candidates_before_filter": len(all_candidates),
        "candidates_after_filter": len(filtered_candidates),
        "selected_for_review": len(selected),
        "filtered_total": len(all_candidates) - len(filtered_candidates),
        "filtered_by_rank": sum(
            1
            for row in all_candidates
            if top_k_per_clip > 0 and (row.get("rank_in_clip") is None or int(row["rank_in_clip"]) > top_k_per_clip)
        ),
        "filtered_by_score": sum(float(row.get("tal_best_score", 0.0)) < min_score for row in all_candidates),
        "filtered_by_duration": sum(
            not (min_duration <= float(row.get("tal_duration", 0.0)) <= max_duration)
            for row in all_candidates
        ),
    }


def review_candidate(
    *,
    row: dict[str, Any],
    clip_paths: dict[str, str],
    output_root: Path,
    context_seconds: float,
    frame_count: int,
    model: str,
    base_url: str,
    api_key_env: str,
    env_file: Path,
    dry_run: bool,
    overwrite: bool,
) -> TalReviewRecord:
    clip_id = row["clip_id"]
    action = row["action"]
    segment = row["tal_best_segment"]
    start = max(0.0, float(segment[0]) - context_seconds)
    end = max(start + 0.05, float(segment[1]) + context_seconds)
    review_dir = output_root / clip_id / safe_name(f"{action}_{start:.2f}_{end:.2f}")
    review_dir.mkdir(parents=True, exist_ok=True)
    proposal_path = review_dir / "tal_proposal.json"
    proposal_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_path = review_dir / "vlm_summary.json"
    status = "exists"
    error = ""
    if overwrite or not summary_path.exists():
        command = [
            str(Path(sys.executable)),
            "scripts/review_video_with_vlm.py",
            clip_paths[clip_id],
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
            "--start",
            f"{start:.4f}",
            "--end",
            f"{end:.4f}",
            "--actions",
            action,
        ]
        if dry_run:
            command.append("--dry-run")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        completed = subprocess.run(command, check=False, capture_output=True, text=True, env=env)
        if completed.returncode != 0:
            status = "review_failed"
            error = completed.stderr.strip() or completed.stdout.strip()
        else:
            status = "completed"

    present = False
    confidence = 0.0
    evidence = ""
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        status = summary.get("status", status)
        action_review = first_action_review(summary.get("review_json", {}))
        present = bool(action_review.get("present", False))
        confidence = float(action_review.get("confidence", 0.0) or 0.0)
        evidence = str(action_review.get("evidence", ""))
        if status == "failed":
            error = summary.get("body") or summary.get("reason") or error

    return TalReviewRecord(
        clip_id=clip_id,
        video_id=row.get("video_id", clip_id.split("_")[0]),
        action=action,
        action_name=row.get("action_name", action),
        outcome=row.get("outcome", ""),
        tal_score=round(float(row.get("tal_best_score", 0.0)), 8),
        tal_iou=round(float(row.get("tal_best_iou", 0.0)), 6),
        start_seconds=round(start, 4),
        end_seconds=round(end, 4),
        review_dir=str(review_dir),
        status=status,
        present=present,
        confidence=round(confidence, 4),
        decision=decision_for(status, present, confidence),
        evidence=evidence,
        error=error,
    )


def first_action_review(review_json: dict[str, Any]) -> dict[str, Any]:
    actions = review_json.get("actions", []) if isinstance(review_json, dict) else []
    return actions[0] if actions else {}


def decision_for(status: str, present: bool, confidence: float) -> str:
    if status == "dry_run":
        return "dry_run"
    if status != "completed" and status != "exists":
        return "review_failed"
    if present and confidence >= 0.5:
        return "vlm_confirmed"
    if present:
        return "vlm_weak_confirmed"
    return "vlm_rejected"


def summarize(records: list[TalReviewRecord]) -> dict[str, int]:
    return {
        "proposals_selected": len(records),
        "completed": sum(record.status == "completed" or record.status == "exists" for record in records),
        "failed": sum(record.decision == "review_failed" for record in records),
        "vlm_confirmed": sum(record.decision == "vlm_confirmed" for record in records),
        "vlm_weak_confirmed": sum(record.decision == "vlm_weak_confirmed" for record in records),
        "vlm_rejected": sum(record.decision == "vlm_rejected" for record in records),
        "dry_run": sum(record.decision == "dry_run" for record in records),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# TAL Proposal VLM Review",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in summary.items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Reviews",
            "",
            "| Clip | Action | Outcome | TAL score | TAL IoU | VLM decision | Confidence |",
            "|---|---|---|---:|---:|---|---:|",
        ]
    )
    for record in payload["records"]:
        lines.append(
            f"| `{record['clip_id']}` | {record['action']} | {record['outcome']} | "
            f"{record['tal_score']:.4f} | {record['tal_iou']:.3f} | "
            f"{record['decision']} | {record['confidence']:.2f} |"
        )
    lines.extend(["", "## Evidence", ""])
    for record in payload["records"]:
        lines.append(f"### `{record['clip_id']}` / {record['action']}")
        lines.append("")
        lines.append(f"- Window: {record['start_seconds']:.2f}s - {record['end_seconds']:.2f}s")
        lines.append(f"- Review dir: `{record['review_dir']}`")
        if record["evidence"]:
            lines.append(f"- Evidence: {record['evidence']}")
        if record["error"]:
            lines.append(f"- Error: {record['error']}")
        lines.append("")
    return "\n".join(lines)


def build_fusion_payload(records: list[TalReviewRecord]) -> dict[str, Any]:
    final_events = [build_fused_event(record, "confirmed_tal_vlm") for record in records if record.decision == "vlm_confirmed"]
    weak_events = [build_fused_event(record, "weak_tal_vlm") for record in records if record.decision == "vlm_weak_confirmed"]
    rejected_events = [build_fused_event(record, "rejected_tal_vlm") for record in records if record.decision == "vlm_rejected"]
    failed_events = [build_fused_event(record, "review_failed") for record in records if record.decision in {"review_failed", "dry_run"}]
    return {
        "summary": {
            "reviewed_proposals": len(records),
            "final_events": len(final_events),
            "weak_events": len(weak_events),
            "rejected_events": len(rejected_events),
            "failed_events": len(failed_events),
        },
        "final_events": final_events,
        "weak_events": weak_events,
        "rejected_events": rejected_events,
        "failed_events": failed_events,
    }


def build_fused_event(record: TalReviewRecord, decision: str) -> dict[str, Any]:
    return {
        "source": "tal_proposal",
        "decision": decision,
        "clip_id": record.clip_id,
        "video_id": record.video_id,
        "action": record.action,
        "action_name": record.action_name,
        "start_seconds": record.start_seconds,
        "end_seconds": record.end_seconds,
        "confidence": fused_confidence(record),
        "evidence_sources": ["tal", "event_vlm"],
        "fusion_reason": fusion_reason(record),
        "tal": {
            "score": record.tal_score,
            "same_label_iou": record.tal_iou,
            "ab_outcome": record.outcome,
        },
        "event_vlm": {
            "present": record.present,
            "confidence": record.confidence,
            "evidence": record.evidence,
            "review_dir": record.review_dir,
        },
    }


def fused_confidence(record: TalReviewRecord) -> float:
    if record.decision == "vlm_confirmed":
        return round((record.confidence + min(1.0, record.tal_score * 10.0)) / 2.0, 4)
    if record.decision == "vlm_weak_confirmed":
        return round(record.confidence * 0.7, 4)
    return round(record.confidence * 0.25, 4)


def fusion_reason(record: TalReviewRecord) -> str:
    if record.decision == "vlm_confirmed":
        return "TAL proposed the temporal action segment and event-centered VLM confirmed the action."
    if record.decision == "vlm_weak_confirmed":
        return "TAL proposed the segment and VLM weakly supported it; keep for review."
    if record.decision == "vlm_rejected":
        return "TAL proposed the segment but event-centered VLM rejected the action."
    return "TAL proposal could not be reviewed successfully."


def render_fusion_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# TAL + VLM Fusion Events",
        "",
        "## Summary",
        "",
        "| Bucket | Count |",
        "|---|---:|",
        f"| final_events | {summary['final_events']} |",
        f"| weak_events | {summary['weak_events']} |",
        f"| rejected_events | {summary['rejected_events']} |",
        f"| failed_events | {summary['failed_events']} |",
        "",
        "## Final Events",
        "",
    ]
    append_event_table(lines, payload["final_events"])
    lines.extend(["", "## Rejected Events", ""])
    append_event_table(lines, payload["rejected_events"])
    return "\n".join(lines)


def append_event_table(lines: list[str], events: list[dict[str, Any]]) -> None:
    if not events:
        lines.append("- None")
        return
    lines.append("| Clip | Action | Time | Confidence | Reason |")
    lines.append("|---|---|---|---:|---|")
    for event in events:
        lines.append(
            f"| `{event['clip_id']}` | {event['action']} | "
            f"{event['start_seconds']:.2f}-{event['end_seconds']:.2f}s | "
            f"{event['confidence']:.4f} | {event['fusion_reason']} |"
        )


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in value)


if __name__ == "__main__":
    main()
