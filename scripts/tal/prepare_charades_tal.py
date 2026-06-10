"""Convert Charades action manifest into ActionFormer JSON annotation format.

Output schema (matches external/actionformer_release/libs/datasets/data_utils.py
and the THUMOS14 / EpicKitchens loaders):

    {
      "version": "1.0",
      "database": {
        "<video_id>": {
          "duration": <float, seconds>,
          "fps": <float>,
          "annotations": [
            {"segment": [start_s, end_s], "label_id": <int>, "label": "<str>"},
            ...
          ]
        },
        ...
      },
      "label_dict": {
        "<action_label>": <label_id>,
        ...
      }
    }

The label_id is the index in label_dict. ActionFormer uses these as classification
targets.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Charades videos: 24 fps is the common convention.  Real fps is read from the
# CSV's `fps` column if present, else 30 is a safe default.
DEFAULT_FPS = 30.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Charades action manifest to ActionFormer JSON format."
    )
    parser.add_argument(
        "manifest",
        type=Path,
        help="Charades manifest CSV with columns: video_id, action_id, action_name, start_seconds, end_seconds, video_length, ...",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output ActionFormer-format JSON path.",
    )
    parser.add_argument(
        "--action-allowlist",
        type=str,
        default=None,
        help="Optional comma-separated Charades action IDs to keep (e.g. c015,c019,c106,c107). Default: keep all.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.0,
        help="Drop annotations shorter than this many seconds (default 0 = keep all).",
    )
    args = parser.parse_args()

    allowlist = None
    if args.action_allowlist:
        allowlist = {a.strip() for a in args.action_allowlist.split(",") if a.strip()}

    payload = build_payload(args.manifest, allowlist=allowlist, min_duration=args.min_duration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(payload['database'])} videos, {sum(len(v['annotations']) for v in payload['database'].values())} annotations to {args.output}")
    print(f"label_dict: {len(payload['label_dict'])} classes")


def build_payload(
    manifest_path: Path,
    allowlist: set[str] | None,
    min_duration: float,
    fps: float = DEFAULT_FPS,
) -> dict:
    # First pass: collect all unique action labels so we can build label_dict.
    label_to_id: dict[str, int] = {}
    rows: list[dict[str, str]] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            action_id = row.get("action_id", "").strip()
            action_name = row.get("action_name", action_id).strip()
            if not action_id or not action_name:
                continue
            if allowlist and action_id not in allowlist:
                continue
            if action_name not in label_to_id:
                label_to_id[action_name] = len(label_to_id)
            rows.append(row)

    # Second pass: build the database.
    db: dict[str, dict] = {}
    for row in rows:
        video_id = row["video_id"].strip()
        start = float(row["start_seconds"])
        end = float(row["end_seconds"])
        if end - start < min_duration:
            continue
        action_id = row["action_id"].strip()
        action_name = row["action_name"].strip()
        # duration falls back to the row's video_length column if present
        duration = float(row.get("video_length") or (end + 0.0))
        if video_id not in db:
            db[video_id] = {
                "duration": round(duration, 4),
                "fps": fps,
                "annotations": [],
            }
        db[video_id]["annotations"].append(
            {
                "segment": [round(start, 4), round(end, 4)],
                "label_id": label_to_id[action_name],
                "label": action_name,
                "charades_action_id": action_id,
            }
        )

    return {
        "version": "1.0",
        "source_manifest": str(manifest_path),
        "label_dict": {label: idx for idx, label in enumerate(sorted(label_to_id, key=lambda k: label_to_id[k]))},
        "database": db,
    }


if __name__ == "__main__":
    main()
