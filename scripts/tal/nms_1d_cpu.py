"""Pure Python fallback for actionformer's C++ nms_1d_cpu extension.

Mirrors the API of the C++ extension at
external/actionformer_release/libs/utils/csrc/nms_cpu.cpp so the
existing nms.py can import this module instead when the C++ build is
unavailable (e.g. on macOS without libomp).

Functions:
    nms(segs, scores, iou_threshold) -> LongTensor of kept indices
    softnms(segs, scores, dets, iou_threshold, sigma, min_score, method) -> LongTensor

This is a 1D NMS. Input/output conventions match the C++ version:
    segs:    (N, 2) float32, second column is x2
    scores:  (N,)   float32
    dets:    (N, 3) float32, [x1, x2, score]
"""
from __future__ import annotations

import torch


def _to_cpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(device="cpu", dtype=torch.float32).contiguous()


def nms(segs: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if segs.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    segs = _to_cpu(segs)
    scores = _to_cpu(scores)
    x1 = segs[:, 0]
    x2 = segs[:, 1]
    areas = (x2 - x1).clamp(min=1e-6)
    order = torch.argsort(scores, descending=True)
    keep: list[int] = []
    suppressed = torch.zeros(scores.shape[0], dtype=torch.bool)
    for idx in order.tolist():
        if suppressed[idx]:
            continue
        keep.append(idx)
        if len(keep) >= 10000:
            break
        ix1 = float(x1[idx])
        ix2 = float(x2[idx])
        iarea = float(areas[idx])
        for j in order.tolist():
            if j == idx or suppressed[j]:
                continue
            xx1 = max(ix1, float(x1[j]))
            xx2 = min(ix2, float(x2[j]))
            inter = max(0.0, xx2 - xx1)
            union = iarea + float(areas[j]) - inter
            if union <= 0:
                continue
            iou = inter / union
            if iou >= iou_threshold:
                suppressed[j] = True
    return torch.tensor(keep, dtype=torch.long)


def softnms(
    segs: torch.Tensor,
    scores: torch.Tensor,
    dets: torch.Tensor,
    iou_threshold: float,
    sigma: float,
    min_score: float,
    method: int,
) -> torch.Tensor:
    """Soft-NMS. Method 0=hard, 1=linear, 2=gaussian (matches C++)."""
    if segs.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    segs = _to_cpu(segs)
    scores = _to_cpu(scores).clone()
    dets = dets.detach().to(device="cpu", dtype=torch.float32).contiguous()
    x1 = segs[:, 0].clone()
    x2 = segs[:, 1].clone()
    sc = scores.clone()
    areas = (x2 - x1).clamp(min=1e-6)
    n = scores.shape[0]
    inds = torch.arange(n, dtype=torch.long)

    # Pre-allocate the dets buffer the C++ version uses
    if dets.shape != (n, 3):
        dets = torch.zeros((n, 3), dtype=torch.float32)

    i = 0
    while i < n:
        # find max score in [i, n)
        sub = sc[i:n]
        rel = int(torch.argmax(sub).item())
        max_pos = i + rel
        # write the best to dets[i]
        dets[i, 0] = x1[max_pos]
        dets[i, 1] = x2[max_pos]
        dets[i, 2] = sc[max_pos]
        ix1, ix2, iscore, iarea, iind = (
            float(x1[max_pos]),
            float(x2[max_pos]),
            float(sc[max_pos]),
            float(areas[max_pos]),
            int(inds[max_pos]),
        )
        # swap best to position i
        x1[max_pos] = x1[i]
        x2[max_pos] = x2[i]
        sc[max_pos] = sc[i]
        areas[max_pos] = areas[i]
        inds[max_pos] = inds[i]
        x1[i] = ix1
        x2[i] = ix2
        sc[i] = iscore
        areas[i] = iarea
        inds[i] = iind

        # decay scores of remaining segments
        j = i + 1
        while j < n:
            xx1 = max(ix1, float(x1[j]))
            xx2 = min(ix2, float(x2[j]))
            inter = max(0.0, xx2 - xx1)
            union = iarea + float(areas[j]) - inter
            ovr = inter / union if union > 0 else 0.0
            weight = 1.0
            if method == 0:
                if ovr >= iou_threshold:
                    weight = 0.0
            elif method == 1:
                if ovr >= iou_threshold:
                    weight = 1.0 - ovr
            elif method == 2:
                weight = float(torch.exp(torch.tensor(-(ovr * ovr) / max(sigma, 1e-6))).item())
            sc[j] = sc[j] * weight
            if sc[j] < min_score:
                # swap with last
                last = n - 1
                x1[j] = x1[last]
                x2[j] = x2[last]
                sc[j] = sc[last]
                areas[j] = areas[last]
                inds[j] = inds[last]
                n -= 1
                j -= 1
            j += 1
        i += 1
    return inds[:n].clone()


if __name__ == "__main__":
    # quick self-test
    segs = torch.tensor([[0.0, 10.0], [5.0, 15.0], [20.0, 30.0]])
    scores = torch.tensor([0.9, 0.8, 0.7])
    kept = nms(segs, scores, 0.5)
    print("hard NMS kept:", kept.tolist())

    dets = torch.zeros((3, 3), dtype=torch.float32)
    kept2 = softnms(segs, scores, dets, 0.5, sigma=0.5, min_score=0.0, method=1)
    print("soft NMS (linear) kept:", kept2.tolist(), "dets:", dets.tolist())
