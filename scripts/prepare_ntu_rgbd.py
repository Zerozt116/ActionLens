from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


ACTION_LABELS = {
    1: ("drink_water", "喝水"),
    28: ("phone_call", "打电话"),
    29: ("play_with_phone_tablet", "玩手机/平板"),
}

FILENAME_RE = re.compile(
    r"S(?P<setup>\d{3})C(?P<camera>\d{3})P(?P<person>\d{3})R(?P<replication>\d{3})A(?P<action>\d{3})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class NtuSample:
    path: str
    file_name: str
    setup_id: int
    camera_id: int
    person_id: int
    replication_id: int
    action_id: int
    action: str
    action_name: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a manifest for selected NTU RGB+D action samples.")
    parser.add_argument("dataset_root", type=Path, help="Root directory containing NTU RGB+D RGB videos or skeleton files.")
    parser.add_argument("--actions", type=int, nargs="+", default=sorted(ACTION_LABELS), help="Action IDs to include.")
    parser.add_argument("--extensions", nargs="+", default=[".avi", ".mp4", ".skeleton"], help="File extensions to scan.")
    parser.add_argument("--output", type=Path, default=Path("data/ntu_rgbd_120_targets.csv"), help="CSV manifest output path.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional JSON manifest output path.")
    args = parser.parse_args()

    samples = build_manifest(args.dataset_root, set(args.actions), set(args.extensions))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, samples)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps([asdict(sample) for sample in samples], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"wrote {len(samples)} samples to {args.output}")
    if args.json_output is not None:
        print(f"wrote {len(samples)} samples to {args.json_output}")


def build_manifest(dataset_root: Path, action_ids: set[int], extensions: set[str]) -> list[NtuSample]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    normalized_extensions = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    samples: list[NtuSample] = []
    for path in sorted(dataset_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in normalized_extensions:
            continue

        parsed = parse_ntu_filename(path.name)
        if parsed is None or parsed["action_id"] not in action_ids:
            continue

        action, action_name = ACTION_LABELS.get(parsed["action_id"], (f"A{parsed['action_id']:03d}", "未知动作"))
        samples.append(
            NtuSample(
                path=str(path),
                file_name=path.name,
                setup_id=parsed["setup_id"],
                camera_id=parsed["camera_id"],
                person_id=parsed["person_id"],
                replication_id=parsed["replication_id"],
                action_id=parsed["action_id"],
                action=action,
                action_name=action_name,
            )
        )
    return samples


def parse_ntu_filename(file_name: str) -> dict[str, int] | None:
    match = FILENAME_RE.search(file_name)
    if match is None:
        return None
    return {
        "setup_id": int(match.group("setup")),
        "camera_id": int(match.group("camera")),
        "person_id": int(match.group("person")),
        "replication_id": int(match.group("replication")),
        "action_id": int(match.group("action")),
    }


def write_csv(path: Path, samples: list[NtuSample]) -> None:
    fieldnames = [
        "path",
        "file_name",
        "setup_id",
        "camera_id",
        "person_id",
        "replication_id",
        "action_id",
        "action",
        "action_name",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow(asdict(sample))


if __name__ == "__main__":
    main()
