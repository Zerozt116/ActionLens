"""Smoke-test the official ActionFormer Charades dataset adapter.

This uses the project-local Charades clip manifest plus smoke feature cache to
exercise the official ActionFormer dataset registry and model forward path.
It is a contract check, not a meaningful trained model evaluation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = PROJECT_ROOT / "external" / "actionformer_release"
TAL_DIR = PROJECT_ROOT / "scripts" / "tal"


def setup_paths() -> None:
    if not EXTERNAL_ROOT.exists():
        raise FileNotFoundError(
            f"actionformer_release not found at {EXTERNAL_ROOT}. "
            "Run: git clone --depth=1 https://github.com/happyharrycn/actionformer_release external/actionformer_release"
        )
    if str(EXTERNAL_ROOT) not in sys.path:
        sys.path.insert(0, str(EXTERNAL_ROOT))
    if str(TAL_DIR) not in sys.path:
        sys.path.insert(0, str(TAL_DIR))

    import importlib

    if "nms_1d_cpu" not in sys.modules:
        fallback = importlib.import_module("nms_1d_cpu")
        sys.modules["nms_1d_cpu"] = fallback
        print(f"[setup] injected pure-Python nms_1d_cpu from {fallback.__file__}")


def choose_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() is false. "
            f"Installed torch CUDA runtime: {getattr(torch.version, 'cuda', None)}"
        )
    return requested


def load_config(config_path: Path, device: str):
    from libs.core import load_config

    cfg = load_config(str(config_path))
    cfg["devices"] = [device]
    return cfg


def make_dataset_from_config(cfg, is_training: bool):
    from libs.datasets import make_dataset

    split = cfg["train_split"] if is_training else cfg["val_split"]
    return make_dataset(cfg["dataset_name"], is_training, split, **cfg["dataset"])


def build_model(cfg, device: str):
    from libs.modeling import make_meta_arch

    model = make_meta_arch(cfg["model_name"], **cfg["model"])
    return model.to(device)


def run_smoke(cfg, mode: str, index: int, device: str) -> dict[str, object]:
    is_training = mode == "train-loss"
    dataset = make_dataset_from_config(cfg, is_training=is_training)
    if len(dataset) == 0:
        raise RuntimeError("Charades dataset adapter produced zero items")
    if index < 0 or index >= len(dataset):
        raise IndexError(f"index {index} out of range for dataset length {len(dataset)}")

    data_item = dataset[index]
    model = build_model(cfg, device)
    model.train(is_training)

    t0 = time.time()
    with torch.no_grad():
        output = model([data_item])
    elapsed_ms = (time.time() - t0) * 1000.0

    summary: dict[str, object] = {
        "mode": mode,
        "device": device,
        "dataset_length": len(dataset),
        "index": index,
        "video_id": data_item["video_id"],
        "feats_shape_cxt": list(data_item["feats"].shape),
        "segments_shape": list(data_item["segments"].shape),
        "labels_shape": list(data_item["labels"].shape),
        "labels": data_item["labels"].tolist(),
        "fps": data_item["fps"],
        "duration": data_item["duration"],
        "feat_stride": data_item["feat_stride"],
        "feat_num_frames": data_item["feat_num_frames"],
        "elapsed_ms": round(elapsed_ms, 2),
        "model_params": sum(param.numel() for param in model.parameters()),
    }

    if is_training:
        summary["losses"] = {
            key: round(float(value.detach().cpu().item()), 6)
            for key, value in output.items()
        }
    else:
        result = output[0]
        summary["prediction_shapes"] = {
            "segments": list(result["segments"].shape),
            "scores": list(result["scores"].shape),
            "labels": list(result["labels"].shape),
        }
        if result["scores"].numel() > 0:
            top_scores, top_idxs = torch.topk(
                result["scores"],
                k=min(5, result["scores"].numel()),
            )
            summary["top_scores"] = [round(float(value), 6) for value in top_scores]
            summary["top_labels"] = result["labels"][top_idxs].tolist()
            summary["top_segments_seconds"] = [
                [round(float(bound), 4) for bound in segment]
                for segment in result["segments"][top_idxs].tolist()
            ]
        else:
            summary["top_scores"] = []
            summary["top_labels"] = []
            summary["top_segments_seconds"] = []

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="ActionFormer Charades adapter smoke test.")
    parser.add_argument(
        "--config",
        type=Path,
        default=EXTERNAL_ROOT / "configs" / "charades_smoke.yaml",
        help="ActionFormer Charades smoke config.",
    )
    parser.add_argument("--index", type=int, default=0, help="Dataset item index to forward-pass.")
    parser.add_argument(
        "--mode",
        choices=["train-loss", "eval"],
        default="train-loss",
        help="Forward mode. train-loss checks GT labels/segments; eval checks postprocess.",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/actionformer_charades_smoke_summary.json"),
        help="Where to write the smoke summary JSON.",
    )
    args = parser.parse_args()

    setup_paths()
    device = choose_device(args.device)
    print(f"[setup] device = {device}")
    cfg = load_config(args.config, device)
    print(
        f"[setup] config = {args.config}, "
        f"classes = {cfg['dataset']['num_classes']}, input_dim = {cfg['dataset']['input_dim']}"
    )

    summary = run_smoke(cfg, mode=args.mode, index=args.index, device=device)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[PASS] wrote {args.summary_output}")


if __name__ == "__main__":
    main()
