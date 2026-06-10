"""Phase 1 smoke test: verify ActionFormer model loads and forward-passes on M5 (MPS/CPU).

What this tests:
  1. actionformer_release codebase imports cleanly
  2. Our pure-Python nms_1d_cpu fallback satisfies libs.utils.nms
  3. LocPointTransformer model builds from a config
  4. Forward pass on a single video with random features produces output of the
     expected shape (segments, scores, labels)

This is intentionally a tiny sanity check — no real features, no real weights.
For real inference on Charades, see scripts/tal/prepare_charades_tal.py and
scripts/tal/infer_real.py (to be added in Phase 2 once features exist).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import torch
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = PROJECT_ROOT / "external" / "actionformer_release"
TAL_DIR = PROJECT_ROOT / "scripts" / "tal"


def setup_paths() -> None:
    """Make actionformer importable and inject pure-Python nms_1d_cpu fallback."""
    if not EXTERNAL_ROOT.exists():
        raise FileNotFoundError(
            f"actionformer_release not found at {EXTERNAL_ROOT}. "
            "Run: git clone --depth=1 https://github.com/happyharrycn/actionformer_release external/actionformer_release"
        )
    if str(EXTERNAL_ROOT) not in sys.path:
        sys.path.insert(0, str(EXTERNAL_ROOT))
    if str(TAL_DIR) not in sys.path:
        sys.path.insert(0, str(TAL_DIR))
    # Pre-inject the fallback module so actionformer's nms.py finds it
    # before it tries to import the absent C++ extension.
    import importlib
    if "nms_1d_cpu" not in sys.modules:
        fallback = importlib.import_module("nms_1d_cpu")
        sys.modules["nms_1d_cpu"] = fallback
        print(f"[setup] injected pure-Python nms_1d_cpu from {fallback.__file__}")


def load_config(config_path: Path, device: str):
    from libs.core import load_config
    cfg = load_config(str(config_path))
    # Override the default cuda device to match the host.
    cfg["devices"] = [device]
    return cfg


def build_model(cfg, device: str):
    from libs.modeling import make_meta_arch
    model = make_meta_arch(cfg["model_name"], **cfg["model"])
    model = model.to(device)
    model.eval()
    return model


def run_forward(model, cfg, device: str, num_frames: int, num_classes: int):
    input_dim = cfg["dataset"]["input_dim"]
    feat_stride = cfg["dataset"]["feat_stride"]
    num_feat_frames = cfg["dataset"]["num_frames"]
    fps = 30.0
    duration = num_frames / fps
    # Random features shaped (C, T) — what the model expects per video.
    feats = torch.randn(input_dim, num_frames, dtype=torch.float32, device=device)
    video_list = [
        {
            "feats": feats,
            "segments": None,
            "labels": None,
            "video_id": "smoke_test",
            "fps": fps,
            "duration": duration,
            "feat_stride": feat_stride,
            "feat_num_frames": num_feat_frames,
        }
    ]
    t0 = time.time()
    with torch.no_grad():
        results = model(video_list)
    elapsed = time.time() - t0
    return results, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="ActionFormer smoke test on local M5.")
    parser.add_argument(
        "--config",
        type=Path,
        default=EXTERNAL_ROOT / "configs" / "thumos_i3d.yaml",
        help="ActionFormer config file (default: THUMOS14 I3D).",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=400,
        help="Synthetic feature sequence length (default 400 ~= 13s at 30fps / feat_stride 4).",
    )
    parser.add_argument(
        "--device",
        choices=["mps", "cpu", "auto"],
        default="auto",
    )
    args = parser.parse_args()

    setup_paths()

    # Pick device.
    if args.device == "auto":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"[setup] device = {device}")

    cfg = load_config(args.config, device)
    num_classes = cfg["dataset"]["num_classes"]
    print(f"[setup] config = {args.config.name}, num_classes = {num_classes}, "
          f"input_dim = {cfg['dataset']['input_dim']}, feat_stride = {cfg['dataset']['feat_stride']}")

    model = build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] built {cfg['model_name']} with {n_params:,} parameters")

    results_list, elapsed = run_forward(model, cfg, device, args.num_frames, num_classes)
    print(f"[forward] elapsed = {elapsed * 1000:.1f} ms on {args.num_frames} frames")

    # inference() returns a list of per-video dicts (batch_size=1 here).
    assert isinstance(results_list, list) and len(results_list) == 1
    results = results_list[0]
    print(f"[output] segments.shape = {tuple(results['segments'].shape)}")
    print(f"[output] scores.shape   = {tuple(results['scores'].shape)}")
    print(f"[output] labels.shape   = {tuple(results['labels'].shape)}")

    # Sample top predictions.
    if results["scores"].numel() > 0:
        top_scores, top_idxs = torch.topk(results["scores"], k=min(5, results["scores"].numel()))
        print(f"[output] top-5 confidence = {[f'{s:.3f}' for s in top_scores.tolist()]}")
        print(f"[output] top-5 class_ids = {results['labels'][top_idxs].tolist()}")
        print(f"[output] top-5 segments  = {results['segments'][top_idxs].tolist()}")
    else:
        print("[output] no segments above pre-NMS threshold")

    print("\n[PASS] smoke test succeeded — model loads and produces output on this device.")


if __name__ == "__main__":
    main()
