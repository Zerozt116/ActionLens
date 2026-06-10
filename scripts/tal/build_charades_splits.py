"""Build deterministic train/val/test splits for the Charades TAL pilot set."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Charades TAL split text files.")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/charades/tal_annotations.json"),
        help="ActionFormer-format Charades annotation JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/charades/splits"),
        help="Directory for train.txt/val.txt/test.txt.",
    )
    parser.add_argument(
        "--clips-manifest",
        type=Path,
        default=None,
        help="Optional clips CSV. When provided, split only videos present in this manifest.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--must-test",
        nargs="*",
        default=["8BG1T", "OINMN", "024PD"],
        help="Video IDs that must be assigned to test.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/charades/splits/summary.json"),
    )
    args = parser.parse_args()

    payload = json.loads(args.annotations.read_text(encoding="utf-8"))
    if args.clips_manifest is not None:
        video_ids = read_video_ids_from_clips(args.clips_manifest)
    else:
        video_ids = sorted(payload.get("database", {}).keys())
    splits = build_splits(
        video_ids,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        must_test=set(args.must_test),
    )
    write_splits(splits, args.output_dir)
    summary = {name: len(video_ids) for name, video_ids in splits.items()}
    summary["seed"] = args.seed
    summary["must_test"] = sorted(set(args.must_test))
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def read_video_ids_from_clips(path: Path) -> list[str]:
    video_ids = set()
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("status") not in {"sliced", "exists"}:
                continue
            video_id = row.get("video_id", "").strip()
            if video_id:
                video_ids.add(video_id)
    return sorted(video_ids)


def build_splits(
    video_ids: list[str],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    must_test: set[str],
) -> dict[str, list[str]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    existing_must_test = sorted(must_test.intersection(video_ids))
    remaining = [video_id for video_id in video_ids if video_id not in must_test]
    rng = random.Random(seed)
    rng.shuffle(remaining)

    train_count = int(len(remaining) * train_ratio)
    val_count = int(len(remaining) * val_ratio)
    train = sorted(remaining[:train_count])
    val = sorted(remaining[train_count: train_count + val_count])
    test = sorted(remaining[train_count + val_count:] + existing_must_test)
    return {"train": train, "val": val, "test": test}


def write_splits(splits: dict[str, list[str]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, video_ids in splits.items():
        (output_dir / f"{name}.txt").write_text("\n".join(video_ids) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
