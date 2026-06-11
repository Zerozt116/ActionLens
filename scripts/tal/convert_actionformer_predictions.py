"""Convert ActionFormer eval_results.pkl into project-friendly reports."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ActionFormer eval_results.pkl to CSV/JSON plus top-k per clip."
    )
    parser.add_argument("predictions", type=Path, help="ActionFormer eval_results.pkl path.")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/charades/tal_annotations.json"),
        help="ActionFormer-format annotation JSON with label_dict.",
    )
    parser.add_argument(
        "--clips-manifest",
        type=Path,
        default=Path("outputs/charades_clips_50.csv"),
        help="Charades clip manifest CSV.",
    )
    parser.add_argument(
        "--eval-groundtruth",
        type=Path,
        default=Path("outputs/charades_clip_eval_val.json"),
        help="Optional clip-level eval GT JSON for best-IoU checks.",
    )
    parser.add_argument("--csv-output", type=Path, required=True, help="Output prediction CSV.")
    parser.add_argument("--json-output", type=Path, required=True, help="Output prediction JSON.")
    parser.add_argument("--topk-output", type=Path, required=True, help="Output per-clip top-k JSON.")
    parser.add_argument("--top-k", type=int, default=10, help="Predictions to keep per clip in top-k output.")
    args = parser.parse_args()

    predictions = load_predictions(args.predictions)
    label_id_to_name, label_id_to_action_id = load_label_maps(args.annotations)
    clip_info = load_clip_manifest(args.clips_manifest)
    gt = load_eval_groundtruth(args.eval_groundtruth)

    rows = normalize_predictions(
        predictions=predictions,
        label_id_to_name=label_id_to_name,
        label_id_to_action_id=label_id_to_action_id,
        clip_info=clip_info,
        groundtruth=gt,
    )
    summary = build_summary(rows, args.predictions)
    topk = build_topk(rows, args.top_k)

    write_csv(args.csv_output, rows)
    write_json(args.json_output, {"summary": summary, "predictions": rows})
    write_json(args.topk_output, {"summary": summary, "top_k": args.top_k, "clips": topk})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {args.csv_output}")
    print(f"wrote {args.json_output}")
    print(f"wrote {args.topk_output}")


def load_predictions(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        payload = pickle.load(file)
    required = {"video-id", "t-start", "t-end", "label", "score"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"{path} is missing required keys: {sorted(missing)}")
    lengths = {key: len(payload[key]) for key in required}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"prediction arrays have mismatched lengths: {lengths}")
    return payload


def load_label_maps(path: Path) -> tuple[dict[int, str], dict[int, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    label_id_to_name = {int(label_id): label for label, label_id in payload["label_dict"].items()}
    label_id_to_action_id: dict[int, str] = {}
    for video in payload.get("database", {}).values():
        for annotation in video.get("annotations", []):
            label_id = int(annotation["label_id"])
            action_id = annotation.get("charades_action_id")
            if action_id:
                label_id_to_action_id.setdefault(label_id, action_id)
    return label_id_to_name, label_id_to_action_id


def load_clip_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    clips: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            clip_id = row.get("clip_id", "")
            if not clip_id:
                continue
            clips[clip_id] = row
    return clips


def load_eval_groundtruth(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        clip_id: item.get("annotations", [])
        for clip_id, item in payload.get("database", {}).items()
    }


def normalize_predictions(
    *,
    predictions: dict[str, Any],
    label_id_to_name: dict[int, str],
    label_id_to_action_id: dict[int, str],
    clip_info: dict[str, dict[str, Any]],
    groundtruth: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(predictions["video-id"])
    for index in range(total):
        clip_id = str(predictions["video-id"][index])
        label_id = int(predictions["label"][index])
        score = float(predictions["score"][index])
        clip = clip_info.get(clip_id, {})
        clip_start = parse_float(clip.get("clip_start_seconds"), 0.0)
        clip_duration = parse_float(clip.get("clip_duration_seconds"), 0.0)
        t_start = max(0.0, float(predictions["t-start"][index]))
        t_end = max(t_start, float(predictions["t-end"][index]))
        if clip_duration > 0:
            t_start = min(t_start, clip_duration)
            t_end = min(t_end, clip_duration)
        gt_iou, gt_match = best_groundtruth_iou(
            t_start=t_start,
            t_end=t_end,
            label_id=label_id,
            annotations=groundtruth.get(clip_id, []),
        )
        rows.append(
            {
                "rank_global": index + 1,
                "clip_id": clip_id,
                "video_id": clip.get("video_id") or clip_id.split("_")[0],
                "label_id": label_id,
                "charades_action_id": label_id_to_action_id.get(label_id, ""),
                "action_name": label_id_to_name.get(label_id, f"label_{label_id}"),
                "t_start": round(t_start, 4),
                "t_end": round(t_end, 4),
                "duration": round(max(0.0, t_end - t_start), 4),
                "global_start": round(clip_start + t_start, 4),
                "global_end": round(clip_start + t_end, 4),
                "score": round(score, 8),
                "best_gt_iou_same_label": round(gt_iou, 6),
                "matched_gt_label": gt_match.get("label", ""),
                "matched_gt_segment": gt_match.get("segment", []),
            }
        )
    rows.sort(key=lambda item: item["score"], reverse=True)
    for index, row in enumerate(rows, start=1):
        row["rank_global"] = index
    return rows


def parse_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def best_groundtruth_iou(
    *,
    t_start: float,
    t_end: float,
    label_id: int,
    annotations: list[dict[str, Any]],
) -> tuple[float, dict[str, Any]]:
    best_iou = 0.0
    best_match: dict[str, Any] = {}
    for annotation in annotations:
        if int(annotation.get("label_id", -1)) != label_id:
            continue
        segment = annotation.get("segment", [])
        if len(segment) != 2:
            continue
        current = temporal_iou(t_start, t_end, float(segment[0]), float(segment[1]))
        if current > best_iou:
            best_iou = current
            best_match = annotation
    return best_iou, best_match


def temporal_iou(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    intersection = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    union = max(end_a, end_b) - min(start_a, start_b)
    if union <= 0:
        return 0.0
    return intersection / union


def build_summary(rows: list[dict[str, Any]], source: Path) -> dict[str, Any]:
    clips = {row["clip_id"] for row in rows}
    labels = Counter(row["action_name"] for row in rows)
    positive_iou = sum(1 for row in rows if row["best_gt_iou_same_label"] > 0)
    high_iou = sum(1 for row in rows if row["best_gt_iou_same_label"] >= 0.5)
    scores = [row["score"] for row in rows]
    return {
        "source": str(source),
        "prediction_count": len(rows),
        "clip_count": len(clips),
        "label_count": len(labels),
        "score_min": min(scores) if scores else 0.0,
        "score_max": max(scores) if scores else 0.0,
        "score_mean": round(sum(scores) / len(scores), 8) if scores else 0.0,
        "same_label_iou_positive_count": positive_iou,
        "same_label_iou_at_0_5_count": high_iou,
        "top_labels_by_prediction_count": labels.most_common(10),
    }


def build_topk(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["clip_id"]].append(row)
    clips: list[dict[str, Any]] = []
    for clip_id, items in sorted(grouped.items()):
        top_items = sorted(items, key=lambda item: item["score"], reverse=True)[:top_k]
        clips.append(
            {
                "clip_id": clip_id,
                "video_id": top_items[0]["video_id"] if top_items else clip_id.split("_")[0],
                "predictions": top_items,
            }
        )
    return clips


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank_global",
        "clip_id",
        "video_id",
        "label_id",
        "charades_action_id",
        "action_name",
        "t_start",
        "t_end",
        "duration",
        "global_start",
        "global_end",
        "score",
        "best_gt_iou_same_label",
        "matched_gt_label",
        "matched_gt_segment",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
