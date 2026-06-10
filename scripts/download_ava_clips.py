from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


YOUTUBE_URL = "https://www.youtube.com/watch?v={video_id}"


@dataclass(frozen=True)
class ClipRequest:
    video_id: str
    timestamp: int
    action_id: int
    action: str
    person_id: int

    @property
    def file_stem(self) -> str:
        return f"{self.video_id}_t{self.timestamp:04d}_p{self.person_id}_a{self.action_id}_{self.action}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download short YouTube clips for AVA manifest rows.")
    parser.add_argument("manifest", type=Path, help="AVA subset manifest CSV.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/ava/clips"))
    parser.add_argument("--seconds-before", type=float, default=2.0)
    parser.add_argument("--seconds-after", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=5, help="Maximum unique rows to download.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without downloading.")
    args = parser.parse_args()

    requests = read_clip_requests(args.manifest, args.limit)
    commands = [
        build_download_command(
            request=request,
            output_dir=args.output_dir,
            seconds_before=args.seconds_before,
            seconds_after=args.seconds_after,
        )
        for request in requests
    ]

    if args.dry_run:
        for command in commands:
            print(" ".join(command))
        return

    yt_dlp = require_tool("yt-dlp")
    ffmpeg = require_tool("ffmpeg")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    tool_dir = str(yt_dlp.parent)
    env["PATH"] = f"{tool_dir}{os.pathsep}{ffmpeg.parent}{os.pathsep}{env.get('PATH', '')}"

    for command in commands:
        command[0] = str(yt_dlp)
        print(" ".join(command))
        subprocess.run(command, check=False, env=env)


def read_clip_requests(manifest: Path, limit: int) -> list[ClipRequest]:
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest}")

    requests: list[ClipRequest] = []
    seen: set[tuple[str, int, int, int]] = set()
    with manifest.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for record in reader:
            key = (
                record["video_id"],
                int(float(record["timestamp"])),
                int(record["person_id"]),
                int(record["action_id"]),
            )
            if key in seen:
                continue
            seen.add(key)
            requests.append(
                ClipRequest(
                    video_id=record["video_id"],
                    timestamp=int(float(record["timestamp"])),
                    action_id=int(record["action_id"]),
                    action=record["action"],
                    person_id=int(record["person_id"]),
                )
            )
            if len(requests) >= limit:
                break
    return requests


def build_download_command(
    request: ClipRequest,
    output_dir: Path,
    seconds_before: float,
    seconds_after: float,
) -> list[str]:
    start = max(0.0, request.timestamp - seconds_before)
    end = request.timestamp + seconds_after
    output_template = str(output_dir / f"{request.file_stem}.%(ext)s")
    return [
        "yt-dlp",
        "--no-playlist",
        "--download-sections",
        f"*{start:.3f}-{end:.3f}",
        "--force-keyframes-at-cuts",
        "-f",
        "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best",
        "-o",
        output_template,
        YOUTUBE_URL.format(video_id=request.video_id),
    ]


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
