"""ActionLens fallback shim for ActionFormer's optional nms_1d_cpu extension.

The upstream training entrypoint imports ``nms_1d_cpu`` as a top-level module.
When the C++/CUDA extension is not built, route that import to the project-local
pure Python fallback used by our TAL smoke tests.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FALLBACK_PATH = PROJECT_ROOT / "scripts" / "tal" / "nms_1d_cpu.py"

spec = importlib.util.spec_from_file_location("_actionlens_nms_1d_cpu", FALLBACK_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load ActionLens nms_1d_cpu fallback from {FALLBACK_PATH}")

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

nms = module.nms
softnms = module.softnms

__all__ = ["nms", "softnms"]
