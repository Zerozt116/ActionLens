from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


REQUIRED_COLUMNS = (
    "video_id",
    "action_id",
    "action_name",
    "start_seconds",
    "end_seconds",
    "clip_start_seconds",
    "clip_end_seconds",
)


@dataclass(frozen=True)
class SliceRecord:
    clip_id: str
    video_id: str
    action_id: str
    action_name: str
    start_seconds: float
    end_seconds: float
    clip_start_seconds: float
    clip_end_seconds: float
    clip_duration_seconds: float
    source_video: str
    clip_path: str
    status: str
    error: str = ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Slice Charades pilot subset into per-action mp4 clips using ffmpeg."
    )
    parser.add_argument("manifest", type=Path, help="Charades pilot subset CSV (must have clip_start/clip_end columns).")
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path("data/charades/videos_480p"),
        help="Directory with downloaded <video_id>.mp4 files.",
    )
    parser.add_argument(
        "--clips-dir",
        type=Path,
        default=Path("data/charades/clips_480p"),
        help="Output directory for sliced clips.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("data/charades/charades_clips.csv"),
        help="Output clips manifest CSV.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/charades/charades_clip_slice_summary.json"),
        help="Output slice summary JSON.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum clips to slice.")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N rows of the manifest.")
    parser.add_argument("--overwrite", action="store_true", help="Re-slice even if the output clip already exists.")
    args = parser.parse_args()

    records = slice_clips(
        manifest_path=args.manifest,
        videos_dir=args.videos_dir,
        clips_dir=args.clips_dir,
        limit=args.limit,
        offset=args.offset,
        overwrite=args.overwrite,
    )

    args.clips_dir.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_clips_csv(args.output_manifest, records)
    args.summary_output.write_text(
        json.dumps(build_summary(records), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    ok_count = sum(record.status in {"sliced", "exists"} for record in records)
    print(f"processed {len(records)} rows, sliced/present: {ok_count}")
    print(f"wrote clips manifest to {args.output_manifest}")
    print(f"wrote summary to {args.summary_output}")


def slice_clips(
    manifest_path: Path,
    videos_dir: Path,
    clips_dir: Path,
    limit: int | None = None,
    offset: int = 0,
    overwrite: bool = False,
) -> list[SliceRecord]:
    rows = read_manifest_rows(manifest_path, offset=offset, limit=limit)
    clips_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = require_tool("ffmpeg")

    records: list[SliceRecord] = []
    for row in rows:
        record = slice_one(row=row, videos_dir=videos_dir, clips_dir=clips_dir, overwrite=overwrite, ffmpeg=ffmpeg)
        records.append(record)
    return records


def slice_one(
    row: dict[str, str],
    videos_dir: Path,
    clips_dir: Path,
    overwrite: bool,
    ffmpeg: Path,
) -> SliceRecord:
    video_id = row["video_id"]
    action_id = row["action_id"]
    start_seconds = float(row["clip_start_seconds"])
    end_seconds = float(row["clip_end_seconds"])
    duration = max(0.0, end_seconds - start_seconds)

    clip_id = f"{video_id}_{action_id}_t{start_seconds:.2f}_{end_seconds:.2f}"
    source_path = videos_dir / f"{video_id}.mp4"
    output_path = clips_dir / f"{clip_id}.mp4"

    if output_path.exists() and not overwrite:
        return SliceRecord(
            clip_id=clip_id,
            video_id=video_id,
            action_id=action_id,
            action_name=row.get("action_name", ""),
            start_seconds=float(row["start_seconds"]),
            end_seconds=float(row["end_seconds"]),
            clip_start_seconds=round(start_seconds, 4),
            clip_end_seconds=round(end_seconds, 4),
            clip_duration_seconds=round(duration, 4),
            source_video=str(source_path),
            clip_path=str(output_path),
            status="exists",
        )

    if not source_path.exists():
        return SliceRecord(
            clip_id=clip_id,
            video_id=video_id,
            action_id=action_id,
            action_name=row.get("action_name", ""),
            start_seconds=float(row["start_seconds"]),
            end_seconds=float(row["end_seconds"]),
            clip_start_seconds=round(start_seconds, 4),
            clip_end_seconds=round(end_seconds, 4),
            clip_duration_seconds=round(duration, 4),
            source_video=str(source_path),
            clip_path=str(output_path),
            status="missing_source",
            error=f"Source video not found: {source_path}",
        )

    command = [
        str(ffmpeg),
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{start_seconds:.3f}",
        "-to",
        f"{end_seconds:.3f}",
        "-i",
        str(source_path),
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        str(output_path),
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except Exception as exc:  # pragma: no cover - defensive runtime path.
        return SliceRecord(
            clip_id=clip_id,
            video_id=video_id,
            action_id=action_id,
            action_name=row.get("action_name", ""),
            start_seconds=float(row["start_seconds"]),
            end_seconds=float(row["end_seconds"]),
            clip_start_seconds=round(start_seconds, 4),
            clip_end_seconds=round(end_seconds, 4),
            clip_duration_seconds=round(duration, 4),
            source_video=str(source_path),
            clip_path=str(output_path),
            status="failed",
            error=str(exc),
        )

    if completed.returncode != 0 or not output_path.exists():
        return SliceRecord(
            clip_id=clip_id,
            video_id=video_id,
            action_id=action_id,
            action_name=row.get("action_name", ""),
            start_seconds=float(row["start_seconds"]),
            end_seconds=float(row["end_seconds"]),
            clip_start_seconds=round(start_seconds, 4),
            clip_end_seconds=round(end_seconds, 4),
            clip_duration_seconds=round(duration, 4),
            source_video=str(source_path),
            clip_path=str(output_path),
            status="failed",
            error=completed.stderr.strip() or completed.stdout.strip() or "ffmpeg returned non-zero exit",
        )

    return SliceRecord(
        clip_id=clip_id,
        video_id=video_id,
        action_id=action_id,
        action_name=row.get("action_name", ""),
        start_seconds=float(row["start_seconds"]),
        end_seconds=float(row["end_seconds"]),
        clip_start_seconds=round(start_seconds, 4),
        clip_end_seconds=round(end_seconds, 4),
        clip_duration_seconds=round(duration, 4),
        source_video=str(source_path),
        clip_path=str(output_path),
        status="sliced",
    )


def read_manifest_rows(manifest_path: Path, offset: int = 0, limit: int | None = None) -> list[dict[str, str]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    rows: list[dict[str, str]] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        missing = [col for col in REQUIRED_COLUMNS if col not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Manifest missing required columns: {missing}")
        for index, row in enumerate(reader):
            if index < offset:
                continue
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_clips_csv(path: Path, records: list[SliceRecord]) -> None:
    fieldnames = list(SliceRecord.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def build_summary(records: list[SliceRecord]) -> dict[str, object]:
    status_counts: dict[str, int] = {}
    for record in records:
        status_counts[record.status] = status_counts.get(record.status, 0) + 1
    sliced = [record for record in records if record.status in {"sliced", "exists"}]
    avg_duration = sum(record.clip_duration_seconds for record in sliced) / len(sliced) if sliced else 0.0
    return {
        "total": len(records),
        "status_counts": status_counts,
        "unique_videos": len({record.video_id for record in sliced}),
        "unique_actions": len({record.action_id for record in sliced}),
        "average_clip_duration_seconds": round(avg_duration, 4),
        "clip_ids": [record.clip_id for record in records],
    }


def require_tool(name: str) -> Path:
    resolved = shutil.which(name)
    if resolved is not None:
        return Path(resolved)

    candidate = Path(sys.executable).parent / name
    if candidate.exists():
        return candidate

    raise RuntimeError(f"Required command not found: {name}")


if __name__ == "__main__":
    main()
