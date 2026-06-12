"""Build clip-level Charades GT JSON for ActionFormer prediction conversion."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build clip-level GT from full-video Charades TAL annotations."
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/charades/tal_annotations.json"),
        help="Full-video TAL annotations JSON.",
    )
    parser.add_argument(
        "--clips-manifest",
        type=Path,
        default=Path("outputs/charades_clips_50.csv"),
        help="Clip manifest CSV.",
    )
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        default=Path("data/charades/features_manifest.json"),
        help="Feature manifest used by the ActionFormer Charades dataset.",
    )
    parser.add_argument(
        "--split-folder",
        type=Path,
        default=Path("data/charades/splits"),
        help="Folder containing train/val/test split text files.",
    )
    parser.add_argument("--split", default="test", help="Split name, e.g. train, val, or test.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to outputs/charades_clip_eval_{split}.json.",
    )
    args = parser.parse_args()

    annotations_payload = json.loads(args.annotations.read_text(encoding="utf-8"))
    feature_payload = json.loads(args.feature_manifest.read_text(encoding="utf-8"))
    allowed_video_ids = read_split_video_ids(args.split_folder / f"{args.split}.txt")
    feature_db = feature_payload.get("database", {})
    annotation_db = annotations_payload.get("database", {})
    label_dict = annotations_payload.get("label_dict", {})
    output = args.output or Path(f"outputs/charades_clip_eval_{args.split}.json")

    database: dict[str, dict[str, Any]] = {}
    for row in read_clip_rows(args.clips_manifest):
        video_id = row["video_id"]
        clip_id = row["clip_id"]
        if allowed_video_ids and video_id not in allowed_video_ids:
            continue
        if clip_id not in feature_db:
            continue

        clip_start = parse_float(row.get("clip_start_seconds"), 0.0)
        clip_end = parse_float(row.get("clip_end_seconds"), clip_start)
        duration = parse_float(
            feature_db[clip_id].get("duration_seconds"),
            max(clip_end - clip_start, 0.0),
        )
        fps = parse_float(
            feature_db[clip_id].get("source_fps"),
            parse_float(annotation_db.get(video_id, {}).get("fps"), 30.0),
        )

        database[clip_id] = {
            "subset": args.split,
            "duration": round(float(duration), 4),
            "fps": round(float(fps), 4),
            "annotations": intersect_clip_annotations(
                annotation_db.get(video_id, {}).get("annotations", []),
                clip_start=clip_start,
                clip_end=clip_end,
            ),
        }

    payload = {
        "version": "1.0",
        "source_json_file": str(args.annotations),
        "source_clips_manifest": str(args.clips_manifest),
        "label_dict": label_dict,
        "database": database,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "split": args.split,
                "clip_count": len(database),
                "annotation_count": sum(len(item["annotations"]) for item in database.values()),
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def read_split_video_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def read_clip_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("status") in {"sliced", "exists"}:
                rows.append(row)
    return rows


def parse_float(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def intersect_clip_annotations(
    annotations: list[dict[str, Any]],
    *,
    clip_start: float,
    clip_end: float,
) -> list[dict[str, Any]]:
    output = []
    for annotation in annotations:
        start, end = annotation.get("segment", [0.0, 0.0])
        left = max(float(start), float(clip_start))
        right = min(float(end), float(clip_end))
        if right <= left:
            continue
        output.append(
            {
                "segment": [round(left - clip_start, 4), round(right - clip_start, 4)],
                "label_id": int(annotation["label_id"]),
                "label": annotation.get("label", str(annotation["label_id"])),
                "charades_action_id": annotation.get("charades_action_id", ""),
            }
        )
    return output


if __name__ == "__main__":
    main()
