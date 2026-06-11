"""Aggregate TAL+VLM fusion gain over the Stage2/VLM baseline."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BASELINE_HIT_OUTCOMES = {
    "gt_hit_by_both",
    "gt_hit_by_stage2_vlm_only",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate TAL+VLM confirmed proposals against the Stage2/VLM A/B baseline."
    )
    parser.add_argument(
        "--ab-report",
        type=Path,
        default=Path("outputs/tal_stage2_vlm_ab_epoch034.json"),
        help="A/B report JSON from compare_tal_stage2_vlm.py.",
    )
    parser.add_argument(
        "--tal-vlm-fusion",
        type=Path,
        default=Path("outputs/tal_vlm_fusion_epoch034.json"),
        help="TAL+VLM fusion JSON from review_tal_proposals_with_vlm.py.",
    )
    parser.add_argument("--output-json", type=Path, required=True, help="Aggregate JSON output.")
    parser.add_argument("--output-md", type=Path, required=True, help="Aggregate Markdown output.")
    args = parser.parse_args()

    ab_report = json.loads(args.ab_report.read_text(encoding="utf-8"))
    fusion = json.loads(args.tal_vlm_fusion.read_text(encoding="utf-8"))
    payload = aggregate(ab_report, fusion)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


def aggregate(ab_report: dict[str, Any], fusion: dict[str, Any]) -> dict[str, Any]:
    rows = ab_report.get("rows", [])
    row_by_key = {(row["clip_id"], row["action"]): row for row in rows}
    gt_keys = {
        (row["clip_id"], row["action"])
        for row in rows
        if row.get("charades_present")
    }
    baseline_hit_keys = {
        (row["clip_id"], row["action"])
        for row in rows
        if row.get("charades_present") and row.get("outcome") in BASELINE_HIT_OUTCOMES
    }
    confirmed_events = fusion.get("final_events", [])
    rejected_events = fusion.get("rejected_events", [])
    confirmed_keys = {(event["clip_id"], event["action"]) for event in confirmed_events}
    rejected_keys = {(event["clip_id"], event["action"]) for event in rejected_events}
    tal_vlm_gt_hit_keys = confirmed_keys & gt_keys
    added_gt_hit_keys = tal_vlm_gt_hit_keys - baseline_hit_keys
    combined_hit_keys = baseline_hit_keys | tal_vlm_gt_hit_keys
    gt_total = len(gt_keys)

    return {
        "summary": {
            "evaluated_clip_count": ab_report.get("summary", {}).get("evaluated_clip_count", 0),
            "gt_action_instances": gt_total,
            "baseline_stage2_vlm_gt_hits": len(baseline_hit_keys),
            "tal_vlm_confirmed_events": len(confirmed_events),
            "tal_vlm_confirmed_gt_hits": len(tal_vlm_gt_hit_keys),
            "tal_vlm_added_gt_hits": len(added_gt_hit_keys),
            "tal_vlm_rejected_events": len(rejected_events),
            "combined_gt_hits": len(combined_hit_keys),
            "baseline_recall_proxy": round(len(baseline_hit_keys) / gt_total, 4) if gt_total else 0.0,
            "combined_recall_proxy": round(len(combined_hit_keys) / gt_total, 4) if gt_total else 0.0,
            "absolute_recall_gain": round((len(combined_hit_keys) - len(baseline_hit_keys)) / gt_total, 4) if gt_total else 0.0,
            "relative_hit_gain": round((len(combined_hit_keys) - len(baseline_hit_keys)) / len(baseline_hit_keys), 4)
            if baseline_hit_keys
            else 0.0,
        },
        "per_action": build_per_action(
            gt_keys=gt_keys,
            baseline_hit_keys=baseline_hit_keys,
            tal_vlm_gt_hit_keys=tal_vlm_gt_hit_keys,
            added_gt_hit_keys=added_gt_hit_keys,
            confirmed_events=confirmed_events,
            rejected_events=rejected_events,
            row_by_key=row_by_key,
        ),
        "added_gt_hits": [
            describe_key(key, row_by_key, confirmed_events)
            for key in sorted(added_gt_hit_keys)
        ],
        "confirmed_events": confirmed_events,
        "rejected_events": rejected_events,
        "rejected_non_gt": [
            describe_rejected(event, row_by_key)
            for event in rejected_events
            if (event["clip_id"], event["action"]) not in gt_keys
        ],
        "rejected_gt": [
            describe_rejected(event, row_by_key)
            for event in rejected_events
            if (event["clip_id"], event["action"]) in gt_keys
        ],
    }


def build_per_action(
    *,
    gt_keys: set[tuple[str, str]],
    baseline_hit_keys: set[tuple[str, str]],
    tal_vlm_gt_hit_keys: set[tuple[str, str]],
    added_gt_hit_keys: set[tuple[str, str]],
    confirmed_events: list[dict[str, Any]],
    rejected_events: list[dict[str, Any]],
    row_by_key: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    actions = sorted({action for _, action in gt_keys} | {event["action"] for event in confirmed_events} | {event["action"] for event in rejected_events})
    confirmed_counter = Counter(event["action"] for event in confirmed_events)
    rejected_counter = Counter(event["action"] for event in rejected_events)
    rows = []
    for action in actions:
        sample_row = next((row for key, row in row_by_key.items() if key[1] == action), {})
        rows.append(
            {
                "action": action,
                "action_name": sample_row.get("action_name", action),
                "gt_instances": count_action(gt_keys, action),
                "baseline_hits": count_action(baseline_hit_keys, action),
                "tal_vlm_gt_hits": count_action(tal_vlm_gt_hit_keys, action),
                "added_gt_hits": count_action(added_gt_hit_keys, action),
                "confirmed_events": confirmed_counter[action],
                "rejected_events": rejected_counter[action],
            }
        )
    return rows


def count_action(keys: set[tuple[str, str]], action: str) -> int:
    return sum(1 for _, key_action in keys if key_action == action)


def describe_key(
    key: tuple[str, str],
    row_by_key: dict[tuple[str, str], dict[str, Any]],
    confirmed_events: list[dict[str, Any]],
) -> dict[str, Any]:
    row = row_by_key.get(key, {})
    event = next((item for item in confirmed_events if (item["clip_id"], item["action"]) == key), {})
    return {
        "clip_id": key[0],
        "action": key[1],
        "action_name": row.get("action_name", key[1]),
        "baseline_outcome": row.get("outcome", ""),
        "tal_score": event.get("tal", {}).get("score"),
        "vlm_confidence": event.get("event_vlm", {}).get("confidence"),
        "start_seconds": event.get("start_seconds"),
        "end_seconds": event.get("end_seconds"),
        "evidence": event.get("event_vlm", {}).get("evidence", ""),
    }


def describe_rejected(event: dict[str, Any], row_by_key: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    row = row_by_key.get((event["clip_id"], event["action"]), {})
    return {
        "clip_id": event["clip_id"],
        "action": event["action"],
        "action_name": row.get("action_name", event["action"]),
        "charades_present": bool(row.get("charades_present", False)),
        "baseline_outcome": row.get("outcome", ""),
        "tal_score": event.get("tal", {}).get("score"),
        "evidence": event.get("event_vlm", {}).get("evidence", ""),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines: list[str] = []
    lines.append("# TAL+VLM Fusion Aggregate")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for key, value in summary.items():
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.4f} |")
        else:
            lines.append(f"| {key} | {value} |")
    lines.append("")
    lines.append("## Per Action")
    lines.append("")
    lines.append("| Action | GT | Baseline hits | TAL+VLM GT hits | Added GT hits | Confirmed | Rejected |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in payload["per_action"]:
        lines.append(
            "| {action} | {gt_instances} | {baseline_hits} | {tal_vlm_gt_hits} | "
            "{added_gt_hits} | {confirmed_events} | {rejected_events} |".format(**row)
        )
    lines.append("")
    lines.append("## Added GT Hits")
    lines.append("")
    if payload["added_gt_hits"]:
        lines.append("| Clip | Action | Time | TAL score | VLM confidence | Evidence |")
        lines.append("|---|---|---|---:|---:|---|")
        for item in payload["added_gt_hits"]:
            lines.append(
                f"| `{item['clip_id']}` | {item['action']} | "
                f"{item['start_seconds']:.2f}-{item['end_seconds']:.2f}s | "
                f"{item['tal_score']:.4f} | {item['vlm_confidence']:.2f} | {item['evidence']} |"
            )
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Rejected TAL Proposals")
    lines.append("")
    rejected = payload["rejected_gt"] + payload["rejected_non_gt"]
    if rejected:
        lines.append("| Clip | Action | GT? | TAL score | Evidence |")
        lines.append("|---|---|---|---:|---|")
        for item in rejected:
            gt = "yes" if item["charades_present"] else "no"
            lines.append(
                f"| `{item['clip_id']}` | {item['action']} | {gt} | "
                f"{item['tal_score']:.4f} | {item['evidence']} |"
            )
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
