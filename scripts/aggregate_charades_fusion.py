from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


BUCKET_KEYS = ("final_events", "semantic_candidates", "pending_events", "rejected_events")
BUCKET_HEADERS = ("Final", "Semantic", "Pending", "Rejected")


@dataclass(frozen=True)
class ActionRow:
    canonical_action: str
    clips_total: int
    charades_groundtruth: int
    final: int
    semantic: int
    pending: int
    rejected: int
    stage2_hits: int
    vlm_present: int
    status_summary: str


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate fused_events.json across a Charades clip batch into a per-action report."
    )
    parser.add_argument(
        "--batch-root",
        type=Path,
        required=True,
        help="Root output directory produced by run_charades_clip_batch.py (contains <clip_id>/fused_events.json).",
    )
    parser.add_argument(
        "--clips-manifest",
        type=Path,
        default=None,
        help="Optional clips_manifest.csv for ground-truth reference (clips_total per action).",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Output JSON report path (default: <batch_root>/aggregation_report.json).",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=None,
        help="Output Markdown report path (default: <batch_root>/aggregation_report.md).",
    )
    args = parser.parse_args()

    clips = read_clips_manifest(args.clips_manifest) if args.clips_manifest else {}
    payload = aggregate_batch(args.batch_root, clips)
    report_json = args.report_json or args.batch_root / "aggregation_report.json"
    report_md = args.report_md or args.batch_root / "aggregation_report.md"
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {report_json}")
    print(f"wrote {report_md}")


def aggregate_batch(batch_root: Path, clips_manifest: dict[str, str]) -> dict[str, object]:
    fused_paths = sorted(batch_root.glob("*/fused_events.json"))
    bucket_counts_by_action: dict[str, dict[str, int]] = defaultdict(lambda: {key: 0 for key in BUCKET_KEYS})
    bucket_counts_by_video: dict[str, dict[str, int]] = defaultdict(lambda: {key: 0 for key in BUCKET_KEYS})
    bucket_counts_overall: Counter[str] = Counter()
    stage2_hits_by_action: Counter[str] = Counter()
    vlm_present_by_action: Counter[str] = Counter()
    fused_status_by_action: Counter[str] = Counter()
    charades_gt_by_action: Counter[str] = Counter()
    total_clips = 0

    for fused_path in fused_paths:
        try:
            payload = json.loads(fused_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        clip_id = fused_path.parent.name
        video_id = clip_id.split("_")[0] if "_" in clip_id else clip_id
        total_clips += 1
        comparison_path = fused_path.parent / "comparison.json"
        if comparison_path.exists():
            for action in read_charades_gt_actions(comparison_path):
                charades_gt_by_action[action] += 1
        for bucket in BUCKET_KEYS:
            events = payload.get(bucket, [])
            bucket_counts_by_video[video_id][bucket] += len(events)
            bucket_counts_overall[bucket] += len(events)
            for event in events:
                action = event.get("action")
                if action:
                    bucket_counts_by_action[action][bucket] += 1
        for event in iter_fused_events(payload):
            action = event.get("action")
            if not action:
                continue
            stage2 = event.get("stage2_confidence")
            if stage2 and stage2 > 0:
                stage2_hits_by_action[action] += 1
            vlm_block = event.get("event_vlm") or event.get("vlm") or {}
            if vlm_block.get("present"):
                vlm_present_by_action[action] += 1
            decision = event.get("decision", "unknown")
            fused_status_by_action[action] += 1
        # We don't have a per-clip "stage2_hits" boolean from the batch summary here, so skip.

    selected_by_action = Counter(manifest_action for manifest_action in clips_manifest.values() if manifest_action)

    actions = sorted(set(bucket_counts_by_action) | set(selected_by_action) | set(charades_gt_by_action))
    rows: list[ActionRow] = []
    for action in actions:
        buckets = bucket_counts_by_action.get(action, {key: 0 for key in BUCKET_KEYS})
        selected = selected_by_action.get(action, 0)
        gt = charades_gt_by_action.get(action, 0)
        status_summary = build_status_summary(
            final=buckets["final_events"],
            semantic=buckets["semantic_candidates"],
            pending=buckets["pending_events"],
            rejected=buckets["rejected_events"],
            gt=gt,
        )
        rows.append(
            ActionRow(
                canonical_action=action,
                clips_total=selected,
                charades_groundtruth=gt,
                final=buckets["final_events"],
                semantic=buckets["semantic_candidates"],
                pending=buckets["pending_events"],
                rejected=buckets["rejected_events"],
                stage2_hits=stage2_hits_by_action.get(action, 0),
                vlm_present=vlm_present_by_action.get(action, 0),
                status_summary=status_summary,
            )
        )

    return {
        "summary": {
            "total_clips_processed": total_clips,
            "total_fused_events": sum(bucket_counts_overall.values()),
            "unique_videos": len(bucket_counts_by_video),
            "unique_actions": len(actions),
            "unique_actions_with_outputs": len(bucket_counts_by_action),
            "overall_bucket_counts": dict(bucket_counts_overall),
        },
        "per_action": [asdict(row) for row in rows],
        "per_video": [
            {"video_id": video_id, **buckets}
            for video_id, buckets in sorted(bucket_counts_by_video.items())
        ],
    }


def build_status_summary(*, final: int, semantic: int, pending: int, rejected: int, gt: int) -> str:
    parts = [f"F={final}", f"S={semantic}", f"P={pending}", f"R={rejected}"]
    if gt > 0:
        parts.append(f"supported_events={final + semantic}, gt_clips={gt}")
    return " ".join(parts)


def read_charades_gt_actions(comparison_path: Path) -> list[str]:
    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    actions = []
    for row in payload.get("comparisons", []):
        if row.get("charades_present") and row.get("action"):
            actions.append(row["action"])
    return actions


def iter_fused_events(payload: dict[str, object]):
    for bucket in BUCKET_KEYS:
        events = payload.get(bucket, [])
        if isinstance(events, list):
            yield from events


def read_clips_manifest(path: Path | None) -> dict[str, str]:
    """Map clip_id -> canonical action from the slice output manifest."""
    import csv

    if path is None or not path.exists():
        return {}
    from scripts.compare_charades_stage2_vlm import CHARADES_TO_CANONICAL

    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            clip_id = row.get("clip_id", "")
            action_id = row.get("action_id", "")
            canonical = CHARADES_TO_CANONICAL.get(action_id, action_id)
            if clip_id and canonical:
                mapping[clip_id] = canonical
    return mapping


def clips_manifest_clip_count(mapping: dict[str, str], action: str) -> int:
    return sum(1 for value in mapping.values() if value == action)


def render_markdown(payload: dict[str, object]) -> str:
    lines: list[str] = []
    summary = payload["summary"]
    lines.append(f"# Charades {summary['total_clips_processed']} Clip Fusion Aggregation Report")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append(f"- **Total clips processed**: {summary['total_clips_processed']}")
    lines.append(f"- **Total fused events**: {summary['total_fused_events']}")
    lines.append(f"- **Unique videos**: {summary['unique_videos']}")
    lines.append(f"- **Unique canonical actions**: {summary['unique_actions']}")
    lines.append(f"- **Unique canonical actions with outputs**: {summary['unique_actions_with_outputs']}")
    lines.append("")
    overall = summary["overall_bucket_counts"]
    lines.append("## Bucket totals")
    lines.append("")
    for bucket, header in zip(BUCKET_KEYS, BUCKET_HEADERS):
        lines.append(f"- **{header}**: {overall.get(bucket, 0)}")
    lines.append("")

    lines.append("## Per canonical action")
    lines.append("")
    header = (
        "| Action | Selected clips | Charades GT clips | "
        + " | ".join(BUCKET_HEADERS)
        + " | Stage2 hits | VLM present | Status |"
    )
    sep = "|" + "---|" * 10
    lines.append(header)
    lines.append(sep)
    for row in payload["per_action"]:
        lines.append(
            "| {action} | {selected} | {gt} | {final} | {semantic} | {pending} | {rejected} | {stage2} | {vlm} | {status} |".format(
                action=row["canonical_action"],
                selected=row["clips_total"],
                gt=row["charades_groundtruth"],
                final=row["final"],
                semantic=row["semantic"],
                pending=row["pending"],
                rejected=row["rejected"],
                stage2=row["stage2_hits"],
                vlm=row["vlm_present"],
                status=row["status_summary"],
            )
        )
    lines.append("")

    lines.append("## Per video")
    lines.append("")
    lines.append("| Video | " + " | ".join(BUCKET_HEADERS) + " |")
    lines.append("|" + "---|" * 5)
    for row in payload["per_video"]:
        lines.append(
            "| {video} | {final} | {semantic} | {pending} | {rejected} |".format(
                video=row["video_id"],
                final=row["final_events"],
                semantic=row["semantic_candidates"],
                pending=row["pending_events"],
                rejected=row["rejected_events"],
            )
        )

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
