"""Summarize TAL top-k recall against clip-level GT."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize top-k TAL recall over a split.")
    parser.add_argument("--topk", type=Path, required=True, help="Top-k JSON from convert_actionformer_predictions.py.")
    parser.add_argument("--groundtruth", type=Path, required=True, help="Clip-level GT JSON.")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    topk = json.loads(args.topk.read_text(encoding="utf-8"))
    gt = json.loads(args.groundtruth.read_text(encoding="utf-8"))
    payload = summarize(topk, gt)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


def summarize(topk: dict[str, Any], gt: dict[str, Any]) -> dict[str, Any]:
    topk_by_clip = {clip["clip_id"]: clip.get("predictions", []) for clip in topk.get("clips", [])}
    rows = []
    per_action: dict[str, Counter[str]] = defaultdict(Counter)
    for clip_id, item in sorted(gt.get("database", {}).items()):
        predictions = topk_by_clip.get(clip_id, [])
        for annotation in item.get("annotations", []):
            label_id = int(annotation["label_id"])
            label = annotation.get("label", str(label_id))
            segment = annotation.get("segment", [])
            best_iou = best_prediction_iou(label_id, segment, predictions)
            action_hit = any(int(prediction.get("label_id", -1)) == label_id for prediction in predictions)
            row = {
                "clip_id": clip_id,
                "label_id": label_id,
                "label": label,
                "segment": segment,
                "topk_action_hit": action_hit,
                "topk_best_iou": round(best_iou, 6),
                "topk_iou_positive_hit": best_iou > 0,
                "topk_iou_at_0_5_hit": best_iou >= 0.5,
            }
            rows.append(row)
            bucket = per_action[label]
            bucket["gt"] += 1
            bucket["action_hit"] += int(row["topk_action_hit"])
            bucket["iou_positive_hit"] += int(row["topk_iou_positive_hit"])
            bucket["iou_at_0_5_hit"] += int(row["topk_iou_at_0_5_hit"])

    total = len(rows)
    action_hits = sum(row["topk_action_hit"] for row in rows)
    iou_positive_hits = sum(row["topk_iou_positive_hit"] for row in rows)
    iou_05_hits = sum(row["topk_iou_at_0_5_hit"] for row in rows)
    return {
        "summary": {
            "clip_count": len(gt.get("database", {})),
            "topk_clip_count": len(topk_by_clip),
            "gt_instances": total,
            "topk_action_hits": action_hits,
            "topk_iou_positive_hits": iou_positive_hits,
            "topk_iou_at_0_5_hits": iou_05_hits,
            "topk_action_recall": round(action_hits / total, 4) if total else 0.0,
            "topk_iou_positive_recall": round(iou_positive_hits / total, 4) if total else 0.0,
            "topk_iou_at_0_5_recall": round(iou_05_hits / total, 4) if total else 0.0,
        },
        "per_action": build_per_action(per_action),
        "rows": rows,
    }


def best_prediction_iou(label_id: int, segment: list[float], predictions: list[dict[str, Any]]) -> float:
    if len(segment) != 2:
        return 0.0
    best = 0.0
    for prediction in predictions:
        if int(prediction.get("label_id", -1)) != label_id:
            continue
        best = max(best, temporal_iou(float(segment[0]), float(segment[1]), float(prediction["t_start"]), float(prediction["t_end"])))
    return best


def temporal_iou(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    intersection = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    union = max(end_a, end_b) - min(start_a, start_b)
    if union <= 0:
        return 0.0
    return intersection / union


def build_per_action(per_action: dict[str, Counter[str]]) -> list[dict[str, Any]]:
    rows = []
    for action, counts in sorted(per_action.items()):
        gt_count = counts["gt"]
        rows.append(
            {
                "action": action,
                "gt": gt_count,
                "action_hits": counts["action_hit"],
                "iou_positive_hits": counts["iou_positive_hit"],
                "iou_at_0_5_hits": counts["iou_at_0_5_hit"],
                "action_recall": round(counts["action_hit"] / gt_count, 4) if gt_count else 0.0,
                "iou_positive_recall": round(counts["iou_positive_hit"] / gt_count, 4) if gt_count else 0.0,
                "iou_at_0_5_recall": round(counts["iou_at_0_5_hit"] / gt_count, 4) if gt_count else 0.0,
            }
        )
    return rows


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# TAL Top-k Recall Summary",
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
            "## Per Action",
            "",
            "| Action | GT | Action hits | IoU>0 hits | IoU>=0.5 hits | Action recall | IoU>0 recall | IoU>=0.5 recall |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["per_action"]:
        lines.append(
            "| {action} | {gt} | {action_hits} | {iou_positive_hits} | {iou_at_0_5_hits} | "
            "{action_recall:.2%} | {iou_positive_recall:.2%} | {iou_at_0_5_recall:.2%} |".format(**row)
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
