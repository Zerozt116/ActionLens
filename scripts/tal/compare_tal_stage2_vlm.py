"""Compare ActionFormer TAL predictions with Stage2/VLM fused Charades results."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.compare_charades_stage2_vlm import (  # noqa: E402
    CANONICAL_ACTION_NAMES,
    CHARADES_TO_CANONICAL,
)


EXTRA_CHARADES_TO_CANONICAL = {
    "c017": "putting_phone",
    "c018": "taking_phone",
    "c108": "pouring_drink_container",
    "c129": "taking_medicine",
}

EXTRA_CANONICAL_ACTION_NAMES = {
    "putting_phone": "放下手机/相机",
    "taking_phone": "拿起手机/相机",
    "pouring_drink_container": "倒水/倒入杯瓶",
    "taking_medicine": "吃药",
}

ACTION_ID_TO_CANONICAL = {**CHARADES_TO_CANONICAL, **EXTRA_CHARADES_TO_CANONICAL}
ACTION_NAMES = {**CANONICAL_ACTION_NAMES, **EXTRA_CANONICAL_ACTION_NAMES}

SUPPORTED_FUSED_STATUSES = {
    "confirmed_event",
    "semantic_candidate",
    "needs_temporal_review",
    "vlm_candidate_event",
    "vlm_recovered_label",
}


@dataclass(frozen=True)
class ActionComparison:
    clip_id: str
    video_id: str
    action: str
    action_name: str
    charades_present: bool
    stage2_present: bool
    vlm_present: bool
    fused_status: str
    fused_supported: bool
    tal_present: bool
    tal_best_score: float
    tal_best_iou: float
    tal_best_segment: list[float]
    tal_prediction_count: int
    final_events: int
    semantic_candidates: int
    pending_events: int
    rejected_events: int
    outcome: str


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare TAL top-k predictions against Stage2/VLM Charades batch outputs."
    )
    parser.add_argument(
        "--tal-topk",
        type=Path,
        default=Path("outputs/actionformer_epoch034_val_topk.json"),
        help="TAL top-k JSON produced by convert_actionformer_predictions.py.",
    )
    parser.add_argument(
        "--batch-root",
        type=Path,
        default=Path("outputs/charades_clip_batch_50"),
        help="Charades batch root containing per-clip comparison.json and fused_events.json.",
    )
    parser.add_argument(
        "--clips-manifest",
        type=Path,
        default=Path("outputs/charades_clips_50.csv"),
        help="Charades clip manifest CSV.",
    )
    parser.add_argument("--output-json", type=Path, required=True, help="Output JSON report path.")
    parser.add_argument("--output-md", type=Path, required=True, help="Output Markdown report path.")
    parser.add_argument(
        "--scope-note",
        default="current TAL eval covers the ActionFormer val split only, not all 50 batch clips.",
        help="Scope note to include in the JSON and Markdown report.",
    )
    args = parser.parse_args()

    tal_by_clip = load_tal_topk(args.tal_topk)
    clip_manifest = load_clip_manifest(args.clips_manifest)
    rows = compare(tal_by_clip, args.batch_root, clip_manifest)
    summary = build_summary(rows, tal_by_clip, args.batch_root)
    summary["scope_note"] = args.scope_note
    payload = {
        "summary": summary,
        "per_action": build_per_action(rows),
        "rows": [asdict(row) for row in rows],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


def load_tal_topk(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {clip["clip_id"]: clip for clip in payload.get("clips", [])}


def load_clip_manifest(path: Path) -> dict[str, dict[str, str]]:
    clips: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            clips[row["clip_id"]] = row
    return clips


def compare(
    tal_by_clip: dict[str, dict[str, Any]],
    batch_root: Path,
    clip_manifest: dict[str, dict[str, str]],
) -> list[ActionComparison]:
    rows: list[ActionComparison] = []
    for clip_id, tal_payload in sorted(tal_by_clip.items()):
        clip_dir = batch_root / clip_id
        comparison_path = clip_dir / "comparison.json"
        fused_path = clip_dir / "fused_events.json"
        if not comparison_path.exists():
            continue
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        fused = json.loads(fused_path.read_text(encoding="utf-8")) if fused_path.exists() else {}
        tal_actions = group_tal_predictions(tal_payload.get("predictions", []))
        fused_counts = count_fused_events(fused)
        action_rows = {row["action"]: row for row in comparison.get("comparisons", [])}

        manifest_action = canonical_from_action_id(clip_manifest.get(clip_id, {}).get("action_id", ""))
        actions = set(action_rows) | set(tal_actions) | set(fused_counts)
        if manifest_action:
            actions.add(manifest_action)

        for action in sorted(actions):
            source = action_rows.get(action, {})
            tal = tal_actions.get(action, [])
            best_tal = tal[0] if tal else {}
            buckets = fused_counts.get(action, {})
            stage2_present = bool(source.get("stage2_present", False))
            vlm_present = bool(source.get("vlm_present", False))
            fused_status = str(source.get("fused_status", "unknown"))
            fused_supported = fused_status in SUPPORTED_FUSED_STATUSES or any(
                buckets.get(key, 0) > 0
                for key in ("final_events", "semantic_candidates", "pending_events")
            )
            charades_present = bool(source.get("charades_present", False)) or action == manifest_action
            rows.append(
                ActionComparison(
                    clip_id=clip_id,
                    video_id=tal_payload.get("video_id", clip_id.split("_")[0]),
                    action=action,
                    action_name=ACTION_NAMES.get(action, action),
                    charades_present=charades_present,
                    stage2_present=stage2_present,
                    vlm_present=vlm_present,
                    fused_status=fused_status,
                    fused_supported=fused_supported,
                    tal_present=bool(tal),
                    tal_best_score=round(float(best_tal.get("score", 0.0)), 8),
                    tal_best_iou=round(float(best_tal.get("best_gt_iou_same_label", 0.0)), 6),
                    tal_best_segment=[
                        round(float(best_tal["t_start"]), 4),
                        round(float(best_tal["t_end"]), 4),
                    ]
                    if best_tal
                    else [],
                    tal_prediction_count=len(tal),
                    final_events=int(buckets.get("final_events", 0)),
                    semantic_candidates=int(buckets.get("semantic_candidates", 0)),
                    pending_events=int(buckets.get("pending_events", 0)),
                    rejected_events=int(buckets.get("rejected_events", 0)),
                    outcome=classify_outcome(
                        charades_present=charades_present,
                        tal_present=bool(tal),
                        stage2_present=stage2_present,
                        vlm_present=vlm_present,
                        fused_supported=fused_supported,
                    ),
                )
            )
    return rows


def canonical_from_action_id(action_id: str) -> str:
    return ACTION_ID_TO_CANONICAL.get(action_id, action_id)


def group_tal_predictions(predictions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        action = canonical_from_action_id(str(prediction.get("charades_action_id", "")))
        if not action:
            continue
        grouped[action].append(prediction)
    for items in grouped.values():
        items.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return grouped


def count_fused_events(payload: dict[str, Any]) -> dict[str, Counter[str]]:
    buckets = {
        "final_events": "final_events",
        "semantic_candidates": "semantic_candidates",
        "pending_events": "pending_events",
        "rejected_events": "rejected_events",
    }
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for bucket_key, bucket_name in buckets.items():
        for event in payload.get(bucket_key, []):
            action = event.get("action")
            if action:
                counts[str(action)][bucket_name] += 1
    return counts


def classify_outcome(
    *,
    charades_present: bool,
    tal_present: bool,
    stage2_present: bool,
    vlm_present: bool,
    fused_supported: bool,
) -> str:
    baseline_present = stage2_present or vlm_present or fused_supported
    if charades_present and tal_present and baseline_present:
        return "gt_hit_by_both"
    if charades_present and tal_present and not baseline_present:
        return "gt_hit_by_tal_only"
    if charades_present and not tal_present and baseline_present:
        return "gt_hit_by_stage2_vlm_only"
    if charades_present:
        return "gt_missed_by_both"
    if tal_present and baseline_present:
        return "non_gt_predicted_by_both"
    if tal_present:
        return "tal_only_non_gt"
    if baseline_present:
        return "stage2_vlm_only_non_gt"
    return "not_present"


def build_summary(
    rows: list[ActionComparison],
    tal_by_clip: dict[str, dict[str, Any]],
    batch_root: Path,
) -> dict[str, Any]:
    evaluated_clips = {row.clip_id for row in rows}
    gt_rows = [row for row in rows if row.charades_present]
    counter = Counter(row.outcome for row in rows)
    gt_total = len(gt_rows)
    tal_gt_hits = sum(1 for row in gt_rows if row.tal_present)
    stage2_vlm_gt_hits = sum(
        1 for row in gt_rows if row.stage2_present or row.vlm_present or row.fused_supported
    )
    both_gt_hits = sum(
        1
        for row in gt_rows
        if row.tal_present and (row.stage2_present or row.vlm_present or row.fused_supported)
    )
    return {
        "tal_clip_count": len(tal_by_clip),
        "evaluated_clip_count": len(evaluated_clips),
        "batch_root": str(batch_root),
        "gt_action_instances": gt_total,
        "tal_topk_gt_hits": tal_gt_hits,
        "stage2_vlm_gt_hits": stage2_vlm_gt_hits,
        "both_gt_hits": both_gt_hits,
        "tal_only_gt_hits": counter["gt_hit_by_tal_only"],
        "stage2_vlm_only_gt_hits": counter["gt_hit_by_stage2_vlm_only"],
        "missed_by_both": counter["gt_missed_by_both"],
        "tal_gt_recall": round(tal_gt_hits / gt_total, 4) if gt_total else 0.0,
        "stage2_vlm_gt_recall": round(stage2_vlm_gt_hits / gt_total, 4) if gt_total else 0.0,
        "outcome_counts": dict(counter),
    }


def build_per_action(rows: list[ActionComparison]) -> list[dict[str, Any]]:
    grouped: dict[str, list[ActionComparison]] = defaultdict(list)
    for row in rows:
        grouped[row.action].append(row)
    report = []
    for action, items in sorted(grouped.items()):
        gt_items = [item for item in items if item.charades_present]
        report.append(
            {
                "action": action,
                "action_name": ACTION_NAMES.get(action, action),
                "rows": len(items),
                "gt_instances": len(gt_items),
                "tal_hits": sum(1 for item in gt_items if item.tal_present),
                "stage2_vlm_hits": sum(
                    1
                    for item in gt_items
                    if item.stage2_present or item.vlm_present or item.fused_supported
                ),
                "tal_predictions": sum(1 for item in items if item.tal_present),
                "final_events": sum(item.final_events for item in items),
                "semantic_candidates": sum(item.semantic_candidates for item in items),
                "pending_events": sum(item.pending_events for item in items),
                "rejected_events": sum(item.rejected_events for item in items),
            }
        )
    return report


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    rows = [ActionComparison(**row) for row in payload["rows"]]
    lines: list[str] = []
    lines.append("# TAL / Stage2 / VLM A/B Report")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- TAL clips: {summary['tal_clip_count']}")
    lines.append(f"- Evaluated overlap clips: {summary['evaluated_clip_count']}")
    lines.append(f"- GT action instances in overlap clips: {summary['gt_action_instances']}")
    lines.append(f"- Note: {summary['scope_note']}")
    lines.append("")
    lines.append("## GT Recall Proxy")
    lines.append("")
    lines.append("| System | GT hits | Recall proxy |")
    lines.append("|---|---:|---:|")
    lines.append(
        f"| TAL top-k | {summary['tal_topk_gt_hits']} | {summary['tal_gt_recall']:.2%} |"
    )
    lines.append(
        f"| Stage2/VLM/Fusion | {summary['stage2_vlm_gt_hits']} | {summary['stage2_vlm_gt_recall']:.2%} |"
    )
    lines.append(f"| Both | {summary['both_gt_hits']} | - |")
    lines.append("")
    lines.append("## Outcome Counts")
    lines.append("")
    lines.append("| Outcome | Count |")
    lines.append("|---|---:|")
    for outcome, count in sorted(summary["outcome_counts"].items()):
        lines.append(f"| {outcome} | {count} |")
    lines.append("")
    lines.append("## Per Action")
    lines.append("")
    lines.append("| Action | GT | TAL hits | Stage2/VLM hits | TAL preds | Final | Pending | Rejected |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for item in payload["per_action"]:
        if item["gt_instances"] == 0 and item["tal_predictions"] == 0 and item["final_events"] == 0:
            continue
        lines.append(
            "| {action} | {gt_instances} | {tal_hits} | {stage2_vlm_hits} | {tal_predictions} | "
            "{final_events} | {pending_events} | {rejected_events} |".format(**item)
        )
    lines.append("")
    lines.append("## Difference Examples")
    lines.append("")
    append_examples(lines, rows, "gt_hit_by_tal_only", "TAL hits GT, Stage2/VLM misses")
    append_examples(lines, rows, "gt_hit_by_stage2_vlm_only", "Stage2/VLM hits GT, TAL misses")
    append_examples(lines, rows, "gt_hit_by_both", "Both hit GT")
    return "\n".join(lines) + "\n"


def append_examples(
    lines: list[str],
    rows: list[ActionComparison],
    outcome: str,
    title: str,
    limit: int = 8,
) -> None:
    examples = [row for row in rows if row.outcome == outcome]
    lines.append(f"### {title}")
    lines.append("")
    if not examples:
        lines.append("- None")
        lines.append("")
        return
    lines.append("| Clip | Action | TAL score | TAL IoU | Fused status | Final/Pending/Rejected |")
    lines.append("|---|---|---:|---:|---|---|")
    for row in examples[:limit]:
        lines.append(
            f"| `{row.clip_id}` | {row.action} | {row.tal_best_score:.4f} | "
            f"{row.tal_best_iou:.3f} | {row.fused_status} | "
            f"{row.final_events}/{row.pending_events}/{row.rejected_events} |"
        )
    lines.append("")


if __name__ == "__main__":
    main()
