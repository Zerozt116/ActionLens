"""Check the SlowFast R50 feature backend before running Charades extraction.

This script is meant for the Linux GPU machine. It validates the four things
that usually fail first: torch/CUDA visibility, pytorchvideo import, SlowFast
model load, and the projection-hook feature shape.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tal.extract_slowfast_features import SlowFastR50Extractor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Check SlowFast R50 backend on GPU/CPU.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    parser.add_argument("--clip-frames", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/slowfast_backend_check_summary.json"),
    )
    args = parser.parse_args()

    summary = build_base_summary(args.device, args.clip_frames, args.alpha, args.crop_size)
    try:
        t0 = time.time()
        extractor = SlowFastR50Extractor(
            device=args.device,
            clip_frames=args.clip_frames,
            alpha=args.alpha,
            crop_size=args.crop_size,
        )
        window = build_synthetic_window(args.clip_frames)
        feature = extractor.extract_window_feature(window)
        elapsed_ms = (time.time() - t0) * 1000.0
        summary.update(
            {
                "status": "ok",
                "resolved_device": str(extractor.device),
                "feature_shape": list(feature.shape),
                "feature_dim_ok": bool(feature.shape == (2304,)),
                "elapsed_ms": round(elapsed_ms, 2),
            }
        )
    except Exception as exc:
        summary.update({"status": "failed", "error": str(exc)})
        write_summary(args.summary_output, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        raise SystemExit(1) from exc

    write_summary(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[PASS] wrote {args.summary_output}")


def build_base_summary(device: str, clip_frames: int, alpha: int, crop_size: int) -> dict[str, object]:
    summary: dict[str, object] = {
        "status": "pending",
        "requested_device": device,
        "clip_frames": clip_frames,
        "alpha": alpha,
        "crop_size": crop_size,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        summary.update(
            {
                "torch": torch.__version__,
                "torch_cuda": getattr(torch.version, "cuda", None),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()),
                "cuda_device_name": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() and torch.cuda.device_count() > 0
                    else ""
                ),
            }
        )
    except Exception as exc:
        summary.update({"torch_error": str(exc)})
    try:
        import pytorchvideo

        summary["pytorchvideo"] = getattr(pytorchvideo, "__version__", "installed")
    except Exception as exc:
        summary["pytorchvideo_error"] = str(exc)
    return summary


def build_synthetic_window(clip_frames: int) -> np.ndarray:
    width, height = 320, 240
    frames = []
    for index in range(clip_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[..., 0] = (index * 7) % 255
        frame[..., 1] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
        frame[..., 2] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
        frames.append(frame)
    return np.stack(frames, axis=0)


def write_summary(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
