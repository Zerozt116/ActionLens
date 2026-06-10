from __future__ import annotations

import argparse
import csv
import io
import json
import struct
import urllib.request
import zipfile
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_ZIP_URL = "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip"


@dataclass(frozen=True)
class DownloadRecord:
    video_id: str
    member_name: str
    output_path: str
    bytes_written: int
    status: str
    error: str = ""


class HTTPRangeReader(io.RawIOBase):
    def __init__(self, url: str, size: int) -> None:
        self.url = url
        self.size = size
        self.position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.position = offset
        elif whence == io.SEEK_CUR:
            self.position += offset
        elif whence == io.SEEK_END:
            self.position = self.size + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        return self.position

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.size - self.position
        if size == 0 or self.position >= self.size:
            return b""
        data = fetch_range(self.url, self.position, size)
        self.position += len(data)
        return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Download selected Charades videos from the remote 480p zip without fetching the whole archive.")
    parser.add_argument("subset", type=Path, help="Charades subset CSV with a video_id column.")
    parser.add_argument("--zip-url", default=DEFAULT_ZIP_URL, help="Remote Charades zip URL.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/charades/videos_480p"), help="Directory for extracted mp4 files.")
    parser.add_argument("--summary-output", type=Path, default=Path("data/charades/charades_download_summary.json"))
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of unique videos to download.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    args = parser.parse_args()

    video_ids = read_video_ids(args.subset)
    if args.limit is not None:
        video_ids = video_ids[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    zip_size = fetch_content_length(args.zip_url)
    records = download_videos(args.zip_url, zip_size, video_ids, args.output_dir, args.overwrite)

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps([asdict(record) for record in records], ensure_ascii=False, indent=2), encoding="utf-8")

    ok_count = sum(record.status in {"downloaded", "exists"} for record in records)
    print(f"processed {len(records)} videos, available locally: {ok_count}")
    print(f"wrote summary to {args.summary_output}")


def read_video_ids(path: Path) -> list[str]:
    seen: set[str] = set()
    video_ids: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for record in reader:
            video_id = record["video_id"]
            if video_id not in seen:
                seen.add(video_id)
                video_ids.append(video_id)
    return video_ids


def download_videos(url: str, zip_size: int, video_ids: list[str], output_dir: Path, overwrite: bool) -> list[DownloadRecord]:
    with zipfile.ZipFile(HTTPRangeReader(url, zip_size)) as archive:
        members = find_members_for_video_ids(archive.namelist(), video_ids)
        records: list[DownloadRecord] = []
        for video_id in video_ids:
            member_name = members.get(video_id, "")
            output_path = output_dir / f"{video_id}.mp4"
            if not member_name:
                records.append(DownloadRecord(video_id, "", str(output_path), 0, "missing", "video id not found in remote zip"))
                continue
            if output_path.exists() and not overwrite:
                records.append(DownloadRecord(video_id, member_name, str(output_path), output_path.stat().st_size, "exists"))
                continue

            try:
                info = archive.getinfo(member_name)
                data = fetch_zip_member(url, info)
                output_path.write_bytes(data)
                records.append(DownloadRecord(video_id, member_name, str(output_path), len(data), "downloaded"))
            except Exception as exc:  # pragma: no cover - defensive runtime path.
                records.append(DownloadRecord(video_id, member_name, str(output_path), 0, "failed", str(exc)))
        return records


def find_members_for_video_ids(member_names: list[str], video_ids: list[str]) -> dict[str, str]:
    requested = set(video_ids)
    members: dict[str, str] = {}
    for member_name in member_names:
        path = Path(member_name)
        if path.suffix.lower() != ".mp4":
            continue
        video_id = path.stem
        if video_id in requested and video_id not in members:
            members[video_id] = member_name
    return members


def fetch_zip_member(url: str, info: zipfile.ZipInfo) -> bytes:
    local_header = fetch_range(url, info.header_offset, 30)
    if len(local_header) != 30:
        raise ValueError(f"Could not read local header for {info.filename}")
    signature, *_rest = struct.unpack("<IHHHHHIIIHH", local_header)
    if signature != 0x04034B50:
        raise ValueError(f"Invalid local header signature for {info.filename}")
    file_name_length, extra_length = struct.unpack("<HH", local_header[26:30])
    data_offset = info.header_offset + 30 + file_name_length + extra_length
    compressed = fetch_range(url, data_offset, info.compress_size)

    if info.compress_type == zipfile.ZIP_STORED:
        data = compressed
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(compressed, -15)
    else:
        raise ValueError(f"Unsupported compression type {info.compress_type} for {info.filename}")

    crc = zlib.crc32(data) & 0xFFFFFFFF
    if crc != info.CRC:
        raise ValueError(f"CRC mismatch for {info.filename}")
    return data


def fetch_content_length(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=30) as response:
        return int(response.headers["Content-Length"])


def fetch_range(url: str, start: int, size: int) -> bytes:
    if size <= 0:
        return b""
    end = start + size - 1
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


if __name__ == "__main__":
    main()
