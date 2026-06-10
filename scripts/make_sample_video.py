from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny sample video from an image.")
    parser.add_argument("-o", "--output", type=Path, default=Path("data/sample_bus.mp4"))
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--image", type=Path, default=None)
    args = parser.parse_args()

    image_path = args.image or _default_ultralytics_asset()
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read source image: {image_path}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    height, width = image.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, args.fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer: {args.output}")

    try:
        frame_total = max(1, int(round(args.seconds * args.fps)))
        for _ in range(frame_total):
            writer.write(image)
    finally:
        writer.release()

    print(args.output)


def _default_ultralytics_asset() -> Path:
    try:
        from ultralytics.utils import ASSETS
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is required for the default sample image. "
            "Install dependencies or pass --image yourself."
        ) from exc
    return Path(ASSETS) / "bus.jpg"


if __name__ == "__main__":
    main()
