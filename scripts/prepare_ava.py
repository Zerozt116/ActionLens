from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


ACTION_LABELS = {
    15: ("answer_phone", "answer phone", "OBJECT_MANIPULATION"),
    27: ("drink", "drink", "OBJECT_MANIPULATION"),
    57: ("text_on_look_at_cellphone", "text on/look at a cellphone", "OBJECT_MANIPULATION"),
}


@dataclass(frozen=True)
class AvaAnnotation:
    video_id: str
    timestamp: int
    x1: float
    y1: float
    x2: float
    y2: float
    action_id: int
    action: str
    action_name: str
    action_type: str
    person_id: int


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a manifest for selected AVA action annotations.")
    parser.add_argument("csv_path", type=Path, help="AVA train/val CSV path.")
    parser.add_argument("--actions", type=int, nargs="+", default=None, help="Optional AVA action IDs to include. Omit to include all actions.")
    parser.add_argument("--action-list", type=Path, default=None, help="Optional AVA action list pbtxt path.")
    parser.add_argument("--output", type=Path, default=Path("data/ava_targets.csv"), help="CSV manifest output path.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional JSON manifest output path.")
    args = parser.parse_args()

    action_labels = load_action_labels(args.action_list or default_action_list_path(args.csv_path))
    annotations = build_manifest(args.csv_path, set(args.actions) if args.actions else None, action_labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, annotations)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps([asdict(item) for item in annotations], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"wrote {len(annotations)} annotations to {args.output}")
    if args.json_output is not None:
        print(f"wrote {len(annotations)} annotations to {args.json_output}")


def build_manifest(csv_path: Path, action_ids: set[int] | None, action_labels: dict[int, tuple[str, str, str]] | None = None) -> list[AvaAnnotation]:
    if not csv_path.exists():
        raise FileNotFoundError(f"AVA CSV does not exist: {csv_path}")

    labels = action_labels or ACTION_LABELS
    annotations: list[AvaAnnotation] = []
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            if not row or len(row) < 8:
                continue
            action_id = int(row[6])
            if action_ids is not None and action_id not in action_ids:
                continue
            action, action_name, action_type = labels.get(action_id, (f"ava_action_{action_id}", f"AVA action {action_id}", "UNKNOWN"))
            annotations.append(
                AvaAnnotation(
                    video_id=row[0],
                    timestamp=int(float(row[1])),
                    x1=float(row[2]),
                    y1=float(row[3]),
                    x2=float(row[4]),
                    y2=float(row[5]),
                    action_id=action_id,
                    action=action,
                    action_name=action_name,
                    action_type=action_type,
                    person_id=int(row[7]),
                )
            )
    return annotations


def default_action_list_path(csv_path: Path) -> Path | None:
    candidate = csv_path.parent / "ava_action_list_v2.2.pbtxt"
    return candidate if candidate.exists() else None


def load_action_labels(action_list_path: Path | None) -> dict[int, tuple[str, str, str]]:
    if action_list_path is None or not action_list_path.exists():
        return ACTION_LABELS

    text = action_list_path.read_text(encoding="utf-8")
    labels: dict[int, tuple[str, str, str]] = {}
    pattern = re.compile(
        r'label\s*\{\s*name:\s*"(?P<name>[^"]+)"\s*label_id:\s*(?P<label_id>\d+)\s*label_type:\s*(?P<label_type>[A-Z_]+)\s*\}',
        re.MULTILINE,
    )
    for match in pattern.finditer(text):
        label_id = int(match.group("label_id"))
        name = match.group("name")
        label_type = match.group("label_type")
        labels[label_id] = (slugify(name), name, label_type)
    return labels or ACTION_LABELS


def slugify(label: str) -> str:
    value = label.lower()
    value = re.sub(r"\([^)]*\)", "", value)
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown_action"


def write_csv(path: Path, annotations: list[AvaAnnotation]) -> None:
    fieldnames = [
        "video_id",
        "timestamp",
        "x1",
        "y1",
        "x2",
        "y2",
        "action_id",
        "action",
        "action_name",
        "action_type",
        "person_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for annotation in annotations:
            writer.writerow(asdict(annotation))


if __name__ == "__main__":
    main()
