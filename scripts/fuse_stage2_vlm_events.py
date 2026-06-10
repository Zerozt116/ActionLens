from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse Stage2 events, Charades labels, and VLM reviews into final events.")
    parser.add_argument("--comparison", type=Path, required=True, help="Comparison JSON produced by compare_charades_stage2_vlm.py.")
    parser.add_argument("--output", type=Path, required=True, help="Output fused_events.json path.")
    parser.add_argument(
        "--event-review",
        type=Path,
        action="append",
        default=[],
        help="Optional event-centered VLM review path, summary path, or review output directory. Can be repeated.",
    )
    parser.add_argument(
        "--min-vlm-candidate-confidence",
        type=float,
        default=0.8,
        help="Minimum confidence for unlabeled full-video VLM candidates.",
    )
    args = parser.parse_args()

    payload = fuse_from_files(
        comparison_path=args.comparison,
        event_review_paths=args.event_review,
        min_vlm_candidate_confidence=args.min_vlm_candidate_confidence,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


def fuse_from_files(
    comparison_path: Path,
    event_review_paths: list[Path] | None = None,
    min_vlm_candidate_confidence: float = 0.8,
) -> dict[str, Any]:
    comparison = read_json(comparison_path)
    event_reviews = [load_event_review(path) for path in event_review_paths or []]
    return fuse_payload(comparison, event_reviews, min_vlm_candidate_confidence=min_vlm_candidate_confidence)


def fuse_payload(
    comparison: dict[str, Any],
    event_reviews: list[dict[str, Any]] | None = None,
    min_vlm_candidate_confidence: float = 0.8,
) -> dict[str, Any]:
    event_reviews_by_index = {review["event_index"]: review for review in event_reviews or [] if review.get("event_index") is not None}
    final_events: list[dict[str, Any]] = []
    semantic_candidates: list[dict[str, Any]] = []
    rejected_events: list[dict[str, Any]] = []
    pending_events: list[dict[str, Any]] = []

    global_event_index = 0
    for row in comparison.get("comparisons", []):
        stage2_events = row.get("stage2", {}).get("events", [])
        for event in stage2_events:
            event_index = int(event.get("stage2_event_index", global_event_index))
            review = event_reviews_by_index.get(event_index)
            fused = fuse_stage2_event(row, event, event_index, review)
            if fused["decision"] == "confirmed_event":
                final_events.append(fused)
            elif fused["decision"] == "rejected_event":
                rejected_events.append(fused)
            else:
                pending_events.append(fused)
            global_event_index += 1

        if not stage2_events:
            semantic_candidates.extend(
                build_semantic_candidates(
                    row,
                    min_vlm_candidate_confidence=min_vlm_candidate_confidence,
                )
            )

    fused_events = [*final_events, *semantic_candidates, *pending_events, *rejected_events]
    summary = {
        "video_id": comparison.get("video_id", ""),
        "final_events": len(final_events),
        "semantic_candidates": len(semantic_candidates),
        "pending_events": len(pending_events),
        "rejected_events": len(rejected_events),
        "fused_events": len(fused_events),
        "event_center_reviews": len(event_reviews_by_index),
    }
    return {
        "video_id": comparison.get("video_id", ""),
        "summary": summary,
        "final_events": final_events,
        "semantic_candidates": semantic_candidates,
        "pending_events": pending_events,
        "rejected_events": rejected_events,
        "fused_events": fused_events,
    }


def fuse_stage2_event(row: dict[str, Any], event: dict[str, Any], event_index: int, review: dict[str, Any] | None) -> dict[str, Any]:
    action = event.get("action", row.get("action", ""))
    stage2_confidence = float(event.get("confidence", 0.0))
    evidence_sources = ["stage2"]
    if row.get("charades_present"):
        evidence_sources.append("charades")

    if review is not None:
        evidence_sources.append("event_vlm")
        if review["present"]:
            decision = "confirmed_event"
            confidence = average_confidence(stage2_confidence, float(review.get("confidence", 0.0)))
            reason = "Stage2 event confirmed by event-centered VLM review."
        else:
            decision = "rejected_event"
            confidence = round(stage2_confidence * 0.25, 4)
            reason = "Stage2 event rejected by event-centered VLM review."
        return {
            **base_stage2_event(row, event, event_index, action),
            "decision": decision,
            "confidence": confidence,
            "evidence_sources": evidence_sources,
            "fusion_reason": reason,
            "stage2_confidence": stage2_confidence,
            "event_vlm": strip_event_review(review),
        }

    fused_status = row.get("fused_status", "")
    if fused_status == "confirmed_event":
        decision = "confirmed_event"
        confidence = average_confidence(stage2_confidence, float(row.get("vlm", {}).get("confidence", 0.0)))
        evidence_sources.append("full_video_vlm")
        reason = "Stage2 and full-video VLM both support the action."
    elif fused_status in {"possible_false_positive"}:
        decision = "rejected_event"
        confidence = round(stage2_confidence * 0.35, 4)
        evidence_sources.append("full_video_vlm")
        reason = "Stage2 event was denied by full-video VLM review."
    else:
        decision = "pending_event"
        confidence = stage2_confidence
        reason = "Stage2 event needs event-centered VLM review before final confirmation."

    return {
        **base_stage2_event(row, event, event_index, action),
        "decision": decision,
        "confidence": confidence,
        "evidence_sources": evidence_sources,
        "fusion_reason": reason,
        "stage2_confidence": stage2_confidence,
        "comparison_status": fused_status,
    }


def base_stage2_event(row: dict[str, Any], event: dict[str, Any], event_index: int, action: str) -> dict[str, Any]:
    return {
        "source": "stage2_event",
        "stage2_event_index": event_index,
        "action": action,
        "action_name": row.get("action_name", event.get("action_name", action)),
        "person_id": event.get("person_id"),
        "start_seconds": event.get("start_seconds"),
        "end_seconds": event.get("end_seconds"),
        "start_time": event.get("start_time"),
        "end_time": event.get("end_time"),
        "clip_start_seconds": event.get("clip_start_seconds"),
        "clip_end_seconds": event.get("clip_end_seconds"),
        "clip_start_time": event.get("clip_start_time"),
        "clip_end_time": event.get("clip_end_time"),
        "time_offset_seconds": event.get("time_offset_seconds"),
        "time_coordinate": event.get("time_coordinate", "video"),
        "stage2_evidence": event.get("evidence", ""),
        "charades": row.get("charades", {}),
    }


def build_semantic_candidates(row: dict[str, Any], min_vlm_candidate_confidence: float) -> list[dict[str, Any]]:
    status = row.get("fused_status", "")
    vlm = row.get("vlm", {})
    vlm_confidence = float(vlm.get("confidence", 0.0))
    if status == "vlm_recovered_label":
        decision = "semantic_candidate"
        reason = "Stage2 missed the action, but Charades and full-video VLM agree."
    elif status == "vlm_candidate_event" and vlm_confidence >= min_vlm_candidate_confidence:
        decision = "semantic_candidate"
        reason = "Full-video VLM found an unlabeled high-confidence action candidate."
    else:
        return []

    intervals = row.get("charades", {}).get("intervals") or [{"start_seconds": None, "end_seconds": None}]
    candidates = []
    for candidate_index, interval in enumerate(intervals):
        candidates.append(
            {
                "source": status,
                "candidate_index": candidate_index,
                "decision": decision,
                "action": row.get("action", ""),
                "action_name": row.get("action_name", row.get("action", "")),
                "start_seconds": interval.get("start_seconds"),
                "end_seconds": interval.get("end_seconds"),
                "confidence": round(vlm_confidence, 4),
                "evidence_sources": semantic_sources(row, status),
                "fusion_reason": reason,
                "vlm": {
                    "confidence": vlm_confidence,
                    "evidence": vlm.get("evidence", ""),
                    "supporting_frame_indices": vlm.get("supporting_frame_indices", []),
                },
                "charades": row.get("charades", {}),
            }
        )
    return candidates


def semantic_sources(row: dict[str, Any], status: str) -> list[str]:
    sources = ["full_video_vlm"]
    if row.get("charades_present"):
        sources.insert(0, "charades")
    if status == "vlm_candidate_event":
        sources.append("unlabeled_candidate")
    return sources


def load_event_review(path: Path) -> dict[str, Any]:
    summary_path = resolve_summary_path(path)
    if summary_path.exists():
        summary = read_json(summary_path)
        review_json = summary.get("review_json") or read_json(Path(summary["review"]))
        review_window = summary.get("review_window", {})
        event_index = parse_event_index(str(review_window.get("source", "")))
        return normalize_event_review(review_json, review_window, event_index)

    review_json = read_json(path)
    return normalize_event_review(review_json, {}, None)


def resolve_summary_path(path: Path) -> Path:
    if path.is_dir():
        return path / "vlm_summary.json"
    if path.name == "vlm_summary.json":
        return path
    if path.name == "vlm_review.json":
        return path.parent / "vlm_summary.json"
    return path.with_name("vlm_summary.json")


def normalize_event_review(review_json: dict[str, Any], review_window: dict[str, Any], event_index: int | None) -> dict[str, Any]:
    actions = review_json.get("actions", [])
    action_review = actions[0] if actions else {}
    return {
        "event_index": event_index,
        "action": action_review.get("action") or (review_window.get("event") or {}).get("action", ""),
        "present": bool(action_review.get("present", False)),
        "confidence": float(action_review.get("confidence", 0.0)),
        "evidence": action_review.get("evidence", ""),
        "supporting_frame_indices": action_review.get("supporting_frame_indices", []),
        "review_window": {
            "start_seconds": review_window.get("start_seconds"),
            "end_seconds": review_window.get("end_seconds"),
            "source": review_window.get("source", ""),
        },
        "visible_objects": review_json.get("visible_objects", []),
        "risk_notes": review_json.get("risk_notes", []),
    }


def strip_event_review(review: dict[str, Any]) -> dict[str, Any]:
    return {
        "present": review.get("present", False),
        "confidence": review.get("confidence", 0.0),
        "evidence": review.get("evidence", ""),
        "supporting_frame_indices": review.get("supporting_frame_indices", []),
        "review_window": review.get("review_window", {}),
        "visible_objects": review.get("visible_objects", []),
        "risk_notes": review.get("risk_notes", []),
    }


def parse_event_index(source: str) -> int | None:
    match = re.fullmatch(r"event:(\d+)", source)
    return int(match.group(1)) if match else None


def average_confidence(left: float, right: float) -> float:
    if left <= 0:
        return round(right, 4)
    if right <= 0:
        return round(left, 4)
    return round((left + right) / 2.0, 4)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
