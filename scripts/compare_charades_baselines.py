from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


BUCKET_KEYS = ("final_events", "semantic_candidates", "pending_events", "rejected_events")


@dataclass(frozen=True)
class BaselineInput:
    label: str
    batch_root: Path


@dataclass(frozen=True)
class MetricRow:
    metric: str
    left: int | float
    right: int | float
    delta: int | float


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two Charades baseline batch outputs.")
    parser.add_argument("--left-label", default="batch20", help="Display label for the left baseline.")
    parser.add_argument("--left-root", type=Path, required=True, help="Left batch root with batch_summary.json and aggregation_report.json.")
    parser.add_argument("--right-label", default="batch50", help="Display label for the right baseline.")
    parser.add_argument("--right-root", type=Path, required=True, help="Right batch root with batch_summary.json and aggregation_report.json.")
    parser.add_argument("--report-json", type=Path, default=Path("outputs/charades_baseline_comparison.json"))
    parser.add_argument("--report-md", type=Path, default=Path("outputs/charades_baseline_comparison.md"))
    args = parser.parse_args()

    payload = compare_baselines(
        BaselineInput(args.left_label, args.left_root),
        BaselineInput(args.right_label, args.right_root),
    )
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {args.report_json}")
    print(f"wrote {args.report_md}")


def compare_baselines(left: BaselineInput, right: BaselineInput) -> dict[str, Any]:
    left_payload = load_baseline(left)
    right_payload = load_baseline(right)
    metric_rows = build_metric_rows(left_payload, right_payload)
    action_rows = build_action_rows(left_payload, right_payload)
    bottlenecks = build_bottlenecks(right_payload)
    return {
        "summary": {
            "left_label": left.label,
            "right_label": right.label,
            "left_root": str(left.batch_root),
            "right_root": str(right.batch_root),
            "left_clips": left_payload["metrics"]["total_clips"],
            "right_clips": right_payload["metrics"]["total_clips"],
            "final_events_delta": right_payload["metrics"]["final_events"] - left_payload["metrics"]["final_events"],
            "semantic_candidates_delta": right_payload["metrics"]["semantic_candidates"] - left_payload["metrics"]["semantic_candidates"],
            "rejected_events_delta": right_payload["metrics"]["rejected_events"] - left_payload["metrics"]["rejected_events"],
            "unique_actions_with_outputs_delta": right_payload["metrics"]["unique_actions_with_outputs"] - left_payload["metrics"]["unique_actions_with_outputs"],
        },
        "metrics": [asdict(row) for row in metric_rows],
        "per_action": action_rows,
        "bottlenecks": bottlenecks,
    }


def load_baseline(baseline: BaselineInput) -> dict[str, Any]:
    batch_summary_path = baseline.batch_root / "batch_summary.json"
    aggregation_path = baseline.batch_root / "aggregation_report.json"
    if not batch_summary_path.exists():
        raise FileNotFoundError(f"Missing batch summary: {batch_summary_path}")
    if not aggregation_path.exists():
        raise FileNotFoundError(f"Missing aggregation report: {aggregation_path}")

    batch = json.loads(batch_summary_path.read_text(encoding="utf-8"))["summary"]
    aggregation = json.loads(aggregation_path.read_text(encoding="utf-8"))
    agg_summary = aggregation["summary"]
    buckets = agg_summary.get("overall_bucket_counts", {})

    metrics = {
        "total_clips": batch.get("total_clips", agg_summary.get("total_clips_processed", 0)),
        "stage2_event_hits": batch.get("stage2_event_hits", 0),
        "stage2_event_rate": batch.get("stage2_event_rate", 0.0),
        "vlm_present_hits": batch.get("vlm_present_hits", 0),
        "vlm_present_rate": batch.get("vlm_present_rate", 0.0),
        "fused_ok": batch.get("fused_ok", 0),
        "total_fused_events": agg_summary.get("total_fused_events", 0),
        "unique_actions": agg_summary.get("unique_actions", 0),
        "unique_actions_with_outputs": agg_summary.get("unique_actions_with_outputs", 0),
        "final_events": buckets.get("final_events", 0),
        "semantic_candidates": buckets.get("semantic_candidates", 0),
        "pending_events": buckets.get("pending_events", 0),
        "rejected_events": buckets.get("rejected_events", 0),
    }
    return {
        "label": baseline.label,
        "root": str(baseline.batch_root),
        "metrics": metrics,
        "per_action": {row["canonical_action"]: row for row in aggregation.get("per_action", [])},
    }


def build_metric_rows(left_payload: dict[str, Any], right_payload: dict[str, Any]) -> list[MetricRow]:
    order = [
        "total_clips",
        "stage2_event_hits",
        "stage2_event_rate",
        "vlm_present_hits",
        "vlm_present_rate",
        "fused_ok",
        "total_fused_events",
        "unique_actions",
        "unique_actions_with_outputs",
        "final_events",
        "semantic_candidates",
        "pending_events",
        "rejected_events",
    ]
    rows = []
    for metric in order:
        left = left_payload["metrics"].get(metric, 0)
        right = right_payload["metrics"].get(metric, 0)
        rows.append(MetricRow(metric=metric, left=left, right=right, delta=round_delta(right - left)))
    return rows


def build_action_rows(left_payload: dict[str, Any], right_payload: dict[str, Any]) -> list[dict[str, Any]]:
    actions = sorted(set(left_payload["per_action"]) | set(right_payload["per_action"]))
    rows = []
    for action in actions:
        left = left_payload["per_action"].get(action, empty_action_row(action))
        right = right_payload["per_action"].get(action, empty_action_row(action))
        left_supported = left.get("final", 0) + left.get("semantic", 0)
        right_supported = right.get("final", 0) + right.get("semantic", 0)
        rows.append(
            {
                "action": action,
                "left_selected_clips": left.get("clips_total", 0),
                "right_selected_clips": right.get("clips_total", 0),
                "left_charades_gt_clips": left.get("charades_groundtruth", 0),
                "right_charades_gt_clips": right.get("charades_groundtruth", 0),
                "left_supported_events": left_supported,
                "right_supported_events": right_supported,
                "supported_delta": right_supported - left_supported,
                "left_final": left.get("final", 0),
                "right_final": right.get("final", 0),
                "left_semantic": left.get("semantic", 0),
                "right_semantic": right.get("semantic", 0),
                "left_rejected": left.get("rejected", 0),
                "right_rejected": right.get("rejected", 0),
                "left_stage2_hits": left.get("stage2_hits", 0),
                "right_stage2_hits": right.get("stage2_hits", 0),
            }
        )
    return rows


def build_bottlenecks(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    rows = list(payload["per_action"].values())
    semantic_only = [
        summarize_action(row)
        for row in rows
        if row.get("semantic", 0) > 0 and row.get("final", 0) == 0 and row.get("stage2_hits", 0) == 0
    ]
    no_output = [
        summarize_action(row)
        for row in rows
        if row.get("charades_groundtruth", 0) > 0 and supported_events(row) == 0 and row.get("rejected", 0) == 0
    ]
    high_rejection = [
        summarize_action(row)
        for row in sorted(rows, key=lambda item: item.get("rejected", 0), reverse=True)
        if row.get("rejected", 0) > 0
    ]
    return {
        "semantic_only_actions": semantic_only,
        "no_output_actions": no_output,
        "high_rejection_actions": high_rejection,
    }


def summarize_action(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "action": row.get("canonical_action"),
        "charades_gt_clips": row.get("charades_groundtruth", 0),
        "final": row.get("final", 0),
        "semantic": row.get("semantic", 0),
        "rejected": row.get("rejected", 0),
        "stage2_hits": row.get("stage2_hits", 0),
        "supported_events": supported_events(row),
    }


def supported_events(row: dict[str, Any]) -> int:
    return row.get("final", 0) + row.get("semantic", 0)


def empty_action_row(action: str) -> dict[str, Any]:
    return {
        "canonical_action": action,
        "clips_total": 0,
        "charades_groundtruth": 0,
        "final": 0,
        "semantic": 0,
        "pending": 0,
        "rejected": 0,
        "stage2_hits": 0,
    }


def round_delta(value: int | float) -> int | float:
    if isinstance(value, float):
        return round(value, 4)
    return value


def render_markdown(payload: dict[str, Any]) -> str:
    left = payload["summary"]["left_label"]
    right = payload["summary"]["right_label"]
    lines = [
        f"# Charades Baseline Comparison: {left} vs {right}",
        "",
        "## Overall Metrics",
        "",
        f"| Metric | {left} | {right} | Delta |",
        "|---|---:|---:|---:|",
    ]
    for row in payload["metrics"]:
        lines.append(f"| {row['metric']} | {row['left']} | {row['right']} | {row['delta']} |")

    lines.extend(
        [
            "",
            "## Per Action",
            "",
            (
                f"| Action | {left} supported | {right} supported | Delta | "
                f"{right} GT clips | {right} Stage2 hits | {right} rejected |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["per_action"]:
        lines.append(
            "| {action} | {left_supported} | {right_supported} | {delta} | {gt} | {stage2} | {rejected} |".format(
                action=row["action"],
                left_supported=row["left_supported_events"],
                right_supported=row["right_supported_events"],
                delta=row["supported_delta"],
                gt=row["right_charades_gt_clips"],
                stage2=row["right_stage2_hits"],
                rejected=row["right_rejected"],
            )
        )

    lines.extend(["", "## Bottlenecks", ""])
    append_action_list(lines, "Semantic-only actions", payload["bottlenecks"]["semantic_only_actions"])
    append_action_list(lines, "No-output actions", payload["bottlenecks"]["no_output_actions"])
    append_action_list(lines, "High-rejection actions", payload["bottlenecks"]["high_rejection_actions"])
    return "\n".join(lines) + "\n"


def append_action_list(lines: list[str], title: str, rows: list[dict[str, Any]]) -> None:
    lines.extend([f"### {title}", ""])
    if not rows:
        lines.extend(["None.", ""])
        return
    lines.extend(
        [
            "| Action | GT clips | Supported | Final | Semantic | Stage2 hits | Rejected |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {action} | {gt} | {supported} | {final} | {semantic} | {stage2} | {rejected} |".format(
                action=row["action"],
                gt=row["charades_gt_clips"],
                supported=row["supported_events"],
                final=row["final"],
                semantic=row["semantic"],
                stage2=row["stage2_hits"],
                rejected=row["rejected"],
            )
        )
    lines.append("")


if __name__ == "__main__":
    main()
