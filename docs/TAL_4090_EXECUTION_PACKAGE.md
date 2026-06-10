# TAL / ActionFormer 4090 执行包

更新时间：2026-06-09

这份执行包用于 Linux + RTX 4090。目标是先生成真实 SlowFast 特征，再进入 ActionFormer 训练 smoke。当前还不是完整训练阶段。

## 0. 前提

在 4090 上进入项目目录：

```bash
cd ~/projects/video-analyst
conda activate actionformer
```

安装依赖：

```bash
pip install -r requirements.txt
pip install h5py tensorboard pandas joblib einops pytorchvideo
```

验证 CUDA：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0))"
```

期望输出包含：

```text
True
NVIDIA GeForce RTX 4090
```

## 1. SlowFast 后端自检

先不要处理真实视频，跑合成 32 帧检查 SlowFast R50：

```bash
python scripts/tal/check_slowfast_backend.py \
  --device cuda \
  --summary-output outputs/slowfast_backend_check_summary.json
```

成功标准：

```json
{
  "status": "ok",
  "resolved_device": "cuda",
  "feature_shape": [2304],
  "feature_dim_ok": true
}
```

如果这里失败，先不要继续。常见原因：

- `pytorchvideo` 没安装。
- PyTorch CUDA 不可用。
- 首次下载 SlowFast 预训练权重时网络失败。

## 2. limit1 真实视频特征

只提 1 个 Charades clip：

```bash
python scripts/tal/extract_slowfast_features.py \
  outputs/charades_clips_50.csv \
  --backend slowfast-r50 \
  --output-dir data/charades/features \
  --summary-output outputs/tal_features_slowfast_limit1_summary.json \
  --feature-manifest-output data/charades/features_manifest_limit1.json \
  --device cuda \
  --limit 1 \
  --overwrite
```

检查：

```bash
python - <<'PY'
import json, numpy as np
s = json.load(open("outputs/tal_features_slowfast_limit1_summary.json"))
print(s["summary"])
record = s["records"][0]
with np.load(record["feature_path"]) as data:
    print(data["feats"].shape, data["feats"].dtype)
PY
```

成功标准：

```text
feature_dims: [2304]
status_counts: {'extracted': 1}
(T, 2304) float32
```

## 3. 提取 50 个真实特征

limit1 通过后再跑全量：

```bash
python scripts/tal/extract_slowfast_features.py \
  outputs/charades_clips_50.csv \
  --backend slowfast-r50 \
  --output-dir data/charades/features \
  --summary-output outputs/tal_features_slowfast_50_summary.json \
  --feature-manifest-output data/charades/features_manifest.json \
  --device cuda \
  --overwrite
```

成功标准：

```text
total = 50
feature_dims = [2304]
status_counts.extracted + status_counts.exists = 50
```

## 4. 真实特征 dataloader smoke

```bash
python scripts/tal/dataloader_smoke.py \
  outputs/charades_clips_50.csv \
  --annotations data/charades/tal_annotations.json \
  --feature-manifest data/charades/features_manifest.json \
  --summary-output outputs/tal_dataloader_slowfast_50_summary.json
```

成功标准：

```text
status_counts.ok = 50
feature_dims_c = [2304]
```

## 5. ActionFormer 训练前 smoke

先跑官方模型 forward，不训练：

```bash
python scripts/tal/actionformer_charades_smoke.py \
  --config external/actionformer_release/configs/charades_slowfast.yaml \
  --device cuda \
  --mode train-loss \
  --index 0 \
  --summary-output outputs/actionformer_charades_slowfast_train_smoke_summary.json
```

成功标准：

```text
mode = train-loss
losses.final_loss 有数值
feats_shape_cxt = [2304, T]
```

## 6. 何时开始训练

只有以上 5 步都通过后，才跑 1 epoch 训练 smoke：

```bash
cd external/actionformer_release
python ./train.py ./configs/charades_slowfast.yaml --output charades_50v_smoke -p 1 -c 1
```

1 epoch smoke 通过后，再决定是否跑完整 30 epoch。

## 7. 回传文件

完成真实特征和训练 smoke 后，把这些文件传回 Mac：

```text
data/charades/features_manifest.json
outputs/tal_features_slowfast_50_summary.json
outputs/tal_dataloader_slowfast_50_summary.json
outputs/actionformer_charades_slowfast_train_smoke_summary.json
external/actionformer_release/ckpt/charades_slowfast_charades_50v_smoke/
```
