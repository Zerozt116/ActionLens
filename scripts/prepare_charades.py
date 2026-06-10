from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class CharadesAction:
    video_id: str
    subject: str
    scene: str
    quality: str
    relevance: str
    verified: str
    video_length: float
    action_id: str
    action_index: int
    action: str
    action_name: str
    object_id: str
    object_name: str
    verb_id: str
    verb_name: str
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    objects: str
    descriptions: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a temporal-action manifest from Charades annotations.")
    parser.add_argument("csv_path", type=Path, help="Charades_v1_train.csv or Charades_v1_test.csv path.")
    parser.add_argument("--classes", type=Path, default=None, help="Optional Charades_v1_classes.txt path.")
    parser.add_argument("--mapping", type=Path, default=None, help="Optional Charades_v1_mapping.txt path.")
    parser.add_argument("--objects", type=Path, default=None, help="Optional Charades_v1_objectclasses.txt path.")
    parser.add_argument("--verbs", type=Path, default=None, help="Optional Charades_v1_verbclasses.txt path.")
    parser.add_argument("--actions", nargs="+", default=None, help="Optional action IDs to keep, e.g. c015 15 c106.")
    parser.add_argument("--verified-only", action="store_true", help="Keep only rows whose video annotation is verified.")
    parser.add_argument("--min-quality", type=int, default=None, help="Keep only videos with quality at least this value.")
    parser.add_argument("--output", type=Path, default=Path("data/charades/charades_manifest.csv"), help="CSV manifest output path.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional JSON manifest output path.")
    parser.add_argument("--summary-output", type=Path, default=None, help="Optional summary JSON output path.")
    args = parser.parse_args()

    base = args.csv_path.parent
    class_labels = load_classes(args.classes or base / "Charades_v1_classes.txt")
    object_labels = load_id_labels(args.objects or base / "Charades_v1_objectclasses.txt")
    verb_labels = load_id_labels(args.verbs or base / "Charades_v1_verbclasses.txt")
    mapping = load_mapping(args.mapping or base / "Charades_v1_mapping.txt")
    action_ids = {normalize_action_id(action) for action in args.actions} if args.actions else None

    actions = build_manifest(
        csv_path=args.csv_path,
        class_labels=class_labels,
        mapping=mapping,
        object_labels=object_labels,
        verb_labels=verb_labels,
        action_ids=action_ids,
        verified_only=args.verified_only,
        min_quality=args.min_quality,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, actions)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps([asdict(item) for item in actions], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(summarize(actions), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"wrote {len(actions)} actions to {args.output}")
    if args.json_output is not None:
        print(f"wrote {len(actions)} actions to {args.json_output}")
    if args.summary_output is not None:
        print(f"wrote summary to {args.summary_output}")


def build_manifest(
    csv_path: Path,
    class_labels: dict[str, str],
    mapping: dict[str, tuple[str, str]],
    object_labels: dict[str, str],
    verb_labels: dict[str, str],
    action_ids: set[str] | None = None,
    verified_only: bool = False,
    min_quality: int | None = None,
) -> list[CharadesAction]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Charades CSV does not exist: {csv_path}")

    actions: list[CharadesAction] = []
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if verified_only and row.get("verified") != "Yes":
                continue
            if min_quality is not None and parse_optional_int(row.get("quality")) < min_quality:
                continue

            video_length = parse_float(row.get("length"), 0.0)
            for action_id, start_seconds, end_seconds in parse_actions(row.get("actions", "")):
                if action_ids is not None and action_id not in action_ids:
                    continue

                object_id, verb_id = mapping.get(action_id, ("", ""))
                actions.append(
                    CharadesAction(
                        video_id=row.get("id", ""),
                        subject=row.get("subject", ""),
                        scene=row.get("scene", ""),
                        quality=row.get("quality", ""),
                        relevance=row.get("relevance", ""),
                        verified=row.get("verified", ""),
                        video_length=round(video_length, 4),
                        action_id=action_id,
                        action_index=int(action_id[1:]),
                        action=slugify(class_labels.get(action_id, action_id)),
                        action_name=class_labels.get(action_id, action_id),
                        object_id=object_id,
                        object_name=object_labels.get(object_id, ""),
                        verb_id=verb_id,
                        verb_name=verb_labels.get(verb_id, ""),
                        start_seconds=round(start_seconds, 4),
                        end_seconds=round(end_seconds, 4),
                        duration_seconds=round(max(0.0, end_seconds - start_seconds), 4),
                        objects=row.get("objects", ""),
                        descriptions=row.get("descriptions", ""),
                    )
                )
    return actions


def parse_actions(value: str) -> list[tuple[str, float, float]]:
    actions: list[tuple[str, float, float]] = []
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.split()
        if len(parts) != 3:
            continue
        actions.append((normalize_action_id(parts[0]), float(parts[1]), float(parts[2])))
    return actions


def load_classes(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    labels: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            action_id, label = line.split(maxsplit=1)
            labels[normalize_action_id(action_id)] = label
    return labels


def load_id_labels(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    labels: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            item_id, label = line.split(maxsplit=1)
            labels[item_id] = label
    return labels


def load_mapping(path: Path) -> dict[str, tuple[str, str]]:
    if not path.exists():
        return {}
    mapping: dict[str, tuple[str, str]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3:
                mapping[normalize_action_id(parts[0])] = (parts[1], parts[2])
    return mapping


def normalize_action_id(value: str) -> str:
    value = value.strip()
    if value.startswith("c"):
        return f"c{int(value[1:]):03d}"
    return f"c{int(value):03d}"


def slugify(label: str) -> str:
    value = label.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown_action"


def parse_float(value: str | None, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def parse_optional_int(value: str | None) -> int:
    if value is None or value == "":
        return 0
    return int(float(value))


def summarize(actions: list[CharadesAction]) -> dict[str, object]:
    videos = {item.video_id for item in actions}
    action_counts = Counter(item.action_id for item in actions)
    scene_counts = Counter(item.scene for item in actions)
    verb_counts = Counter(item.verb_name for item in actions)
    object_counts = Counter(item.object_name for item in actions)
    return {
        "actions": len(actions),
        "videos": len(videos),
        "classes": len(action_counts),
        "scenes": len(scene_counts),
        "average_action_duration_seconds": round(sum(item.duration_seconds for item in actions) / len(actions), 4) if actions else 0.0,
        "top_actions": top_counts(action_counts, actions, "action_id", "action_name"),
        "top_scenes": scene_counts.most_common(20),
        "top_verbs": verb_counts.most_common(20),
        "top_objects": object_counts.most_common(20),
    }


def top_counts(counter: Counter[str], actions: list[CharadesAction], key_attr: str, label_attr: str) -> list[dict[str, object]]:
    labels = {getattr(item, key_attr): getattr(item, label_attr) for item in actions}
    return [{"id": key, "name": labels.get(key, key), "count": count} for key, count in counter.most_common(20)]


def write_csv(path: Path, actions: list[CharadesAction]) -> None:
    fieldnames = [
        "video_id",
        "subject",
        "scene",
        "quality",
        "relevance",
        "verified",
        "video_length",
        "action_id",
        "action_index",
        "action",
        "action_name",
        "object_id",
        "object_name",
        "verb_id",
        "verb_name",
        "start_seconds",
        "end_seconds",
        "duration_seconds",
        "objects",
        "descriptions",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for action in actions:
            writer.writerow(asdict(action))


if __name__ == "__main__":
    main()
