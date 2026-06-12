"""Analyze TAL proposal filter rules against reviewed VLM outcomes."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_TOP_K = [3, 5, 10]
DEFAULT_SCORE_THRESHOLDS = [0.0, 0.03, 0.035, 0.04, 0.045, 0.05]
DEFAULT_DURATION_RANGES = [
    (0.0, 999.0),
    (0.5, 30.0),
    (1.0, 20.0),
    (2.0, 15.0),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate score/top-k/duration filters using already reviewed TAL proposals."
    )
    parser.add_argument(
        "--review-analysis",
        type=Path,
        default=Path("outputs/tal_vlm_review_epoch034_combined_analysis.json"),
        help="Combined VLM review analysis JSON.",
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=Path("outputs/actionformer_epoch034_val_predictions.csv"),
        help="Full TAL prediction CSV used to compute per-clip/action ranks.",
    )
    parser.add_argument("--output-json", type=Path, required=True, help="Output JSON report.")
    parser.add_argument("--output-md", type=Path, required=True, help="Output Markdown report.")
    args = parser.parse_args()

    reviewed = load_reviewed(args.review_analysis)
    ranks = load_ranks(args.predictions_csv)
    for record in reviewed:
        record["rank_in_clip"] = ranks.get(
            (record["clip_id"], record["action"], round(float(record["tal_score"]), 8)),
            None,
        )
        record["duration"] = round(float(record["end_seconds"]) - float(record["start_seconds"]), 4)

    strategies = evaluate_strategies(reviewed)
    payload = {
        "summary": build_summary(reviewed, strategies),
        "best_strategies": select_best_strategies(strategies),
        "strategies": strategies,
        "reviewed_records": reviewed,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


def load_reviewed(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [dict(record) for record in payload.get("records", [])]


def load_ranks(path: Path) -> dict[tuple[str, str, float], int]:
    by_clip: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            by_clip[row["clip_id"]].append(row)

    ranks: dict[tuple[str, str, float], int] = {}
    for clip_id, rows in by_clip.items():
        rows.sort(key=lambda row: float(row["score"]), reverse=True)
        for index, row in enumerate(rows, start=1):
            action = canonical_from_charades_action_id(row["charades_action_id"])
            ranks[(clip_id, action, round(float(row["score"]), 8))] = index
    return ranks


def canonical_from_charades_action_id(action_id: str) -> str:
    mapping = {
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
    return mapping.get(action_id, action_id)


def evaluate_strategies(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    strategies = []
    total_reviewed = len(records)
    total_confirmed = sum(is_confirmed(record) for record in records)
    total_rejected = sum(is_rejected(record) for record in records)
    for top_k in DEFAULT_TOP_K:
        for score_threshold in DEFAULT_SCORE_THRESHOLDS:
            for min_duration, max_duration in DEFAULT_DURATION_RANGES:
                kept = [
                    record
                    for record in records
                    if keep_record(record, top_k, score_threshold, min_duration, max_duration)
                ]
                confirmed = sum(is_confirmed(record) for record in kept)
                rejected = sum(is_rejected(record) for record in kept)
                kept_count = len(kept)
                filtered_count = total_reviewed - kept_count
                strategies.append(
                    {
                        "top_k_per_clip": top_k,
                        "score_threshold": score_threshold,
                        "min_duration": min_duration,
                        "max_duration": max_duration,
                        "kept_reviewed_proposals": kept_count,
                        "filtered_reviewed_proposals": filtered_count,
                        "kept_confirmed": confirmed,
                        "kept_rejected": rejected,
                        "filtered_confirmed": total_confirmed - confirmed,
                        "filtered_rejected": total_rejected - rejected,
                        "precision_on_reviewed": round(confirmed / kept_count, 4) if kept_count else 0.0,
                        "confirmed_retention": round(confirmed / total_confirmed, 4) if total_confirmed else 0.0,
                        "rejected_filter_rate": round((total_rejected - rejected) / total_rejected, 4) if total_rejected else 0.0,
                        "vlm_call_reduction": round(filtered_count / total_reviewed, 4) if total_reviewed else 0.0,
                        "kept_records": [
                            {"clip_id": r["clip_id"], "action": r["action"], "decision": r["decision"], "score": r["tal_score"]}
                            for r in kept
                        ],
                    }
                )
    return strategies


def keep_record(
    record: dict[str, Any],
    top_k: int,
    score_threshold: float,
    min_duration: float,
    max_duration: float,
) -> bool:
    rank = record.get("rank_in_clip")
    if rank is None or int(rank) > top_k:
        return False
    score = float(record["tal_score"])
    duration = float(record["duration"])
    return score >= score_threshold and min_duration <= duration <= max_duration


def is_confirmed(record: dict[str, Any]) -> bool:
    return record.get("decision") in {"vlm_confirmed", "vlm_weak_confirmed"}


def is_rejected(record: dict[str, Any]) -> bool:
    return record.get("decision") == "vlm_rejected"


def build_summary(records: list[dict[str, Any]], strategies: list[dict[str, Any]]) -> dict[str, Any]:
    confirmed = sum(is_confirmed(record) for record in records)
    rejected = sum(is_rejected(record) for record in records)
    no_loss = [strategy for strategy in strategies if strategy["confirmed_retention"] == 1.0]
    best_no_loss = max(
        no_loss,
        key=lambda item: (item["vlm_call_reduction"], item["precision_on_reviewed"], item["rejected_filter_rate"]),
        default=None,
    )
    balanced = max(
        strategies,
        key=lambda item: (
            item["confirmed_retention"] >= 0.67,
            item["vlm_call_reduction"],
            item["precision_on_reviewed"],
        ),
        default=None,
    )
    return {
        "reviewed_total": len(records),
        "confirmed_total": confirmed,
        "rejected_total": rejected,
        "strategies_evaluated": len(strategies),
        "best_no_confirmed_loss": summarize_strategy(best_no_loss),
        "balanced_recommendation": summarize_strategy(balanced),
    }


def summarize_strategy(strategy: dict[str, Any] | None) -> dict[str, Any] | None:
    if strategy is None:
        return None
    keys = [
        "top_k_per_clip",
        "score_threshold",
        "min_duration",
        "max_duration",
        "kept_reviewed_proposals",
        "kept_confirmed",
        "kept_rejected",
        "precision_on_reviewed",
        "confirmed_retention",
        "rejected_filter_rate",
        "vlm_call_reduction",
    ]
    return {key: strategy[key] for key in keys}


def select_best_strategies(strategies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = sorted(
        strategies,
        key=lambda item: (
            -item["confirmed_retention"],
            -item["vlm_call_reduction"],
            -item["precision_on_reviewed"],
            item["kept_reviewed_proposals"],
        ),
    )
    return [summarize_strategy(strategy) for strategy in candidates[:12]]


def render_markdown(payload: dict[str, Any]) -> str:
    lines = ["# TAL Proposal Filter Analysis", ""]
    lines.extend(render_summary(payload["summary"]))
    lines.extend(render_strategy_table("Best Strategies", payload["best_strategies"]))
    lines.extend(render_reviewed_records(payload["reviewed_records"]))
    return "\n".join(lines) + "\n"


def render_summary(summary: dict[str, Any]) -> list[str]:
    lines = ["## Summary", "", "| Metric | Value |", "|---|---:|"]
    for key in ["reviewed_total", "confirmed_total", "rejected_total", "strategies_evaluated"]:
        lines.append(f"| {key} | {summary[key]} |")
    lines.append("")
    for label, key in [
        ("Best No Confirmed Loss", "best_no_confirmed_loss"),
        ("Balanced Recommendation", "balanced_recommendation"),
    ]:
        strategy = summary.get(key)
        lines.append(f"## {label}")
        lines.append("")
        if not strategy:
            lines.append("- None")
            lines.append("")
            continue
        lines.append("| Setting | Value |")
        lines.append("|---|---:|")
        for item_key, value in strategy.items():
            lines.append(f"| {item_key} | {value} |")
        lines.append("")
    return lines


def render_strategy_table(title: str, strategies: list[dict[str, Any]]) -> list[str]:
    lines = [f"## {title}", ""]
    lines.append("| top-k | score | duration | kept | confirmed | rejected | precision | retention | reject filter | call reduction |")
    lines.append("|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for item in strategies:
        lines.append(
            f"| {item['top_k_per_clip']} | {item['score_threshold']:.3f} | "
            f"{item['min_duration']:.1f}-{item['max_duration']:.1f}s | "
            f"{item['kept_reviewed_proposals']} | {item['kept_confirmed']} | {item['kept_rejected']} | "
            f"{item['precision_on_reviewed']:.2%} | {item['confirmed_retention']:.2%} | "
            f"{item['rejected_filter_rate']:.2%} | {item['vlm_call_reduction']:.2%} |"
        )
    lines.append("")
    return lines


def render_reviewed_records(records: list[dict[str, Any]]) -> list[str]:
    lines = ["## Reviewed Records", ""]
    lines.append("| Clip | Action | Score | Duration | Rank | Decision | Group |")
    lines.append("|---|---|---:|---:|---:|---|---|")
    for record in sorted(records, key=lambda item: -float(item["tal_score"])):
        rank = record.get("rank_in_clip")
        rank_text = str(rank) if rank is not None else "-"
        lines.append(
            f"| `{record['clip_id']}` | {record['action']} | {record['tal_score']:.4f} | "
            f"{record['duration']:.2f} | {rank_text} | {record['decision']} | {record.get('review_group', '')} |"
        )
    lines.append("")
    return lines


if __name__ == "__main__":
    main()
