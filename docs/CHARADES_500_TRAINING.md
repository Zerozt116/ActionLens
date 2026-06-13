# Charades 500-Clip Training Plan

目的：在 200-clip 已证明 TAL proposal 有效后，扩到 500 clips，验证 ActionFormer 的泛化是否继续提升。

## 本轮新增资产

```text
data/charades/charades_pilot_subset_500.csv
data/charades/charades_pilot_subset_500.json
data/charades/charades_pilot_subset_500_summary.json
data/charades/splits_500/train.txt
data/charades/splits_500/val.txt
data/charades/splits_500/test.txt
data/charades/splits_500/summary.json
external/actionformer_release/configs/charades_slowfast_500.yaml
external/actionformer_release/configs/charades_slowfast_500_train_export.yaml
external/actionformer_release/configs/charades_slowfast_500_test_export.yaml
```

500-clip split：

```text
train = 347
val = 74
test = 79
```

动作覆盖：

```text
20 actions x 25 clips/action = 500 clips
```

## 1. 拉取最新代码

在 4090 机器执行：

```bash
cd ~/ActionLens
git pull --rebase origin main
```

如果 pull 被未跟踪文件挡住，先把对应文件移动到 `/tmp/actionlens_backup/`，再重新 pull。

## 2. 下载缺失视频

500 清单复用已有 200 个视频，并新增 300 个 Charades 视频。执行：

```bash
python scripts/download_charades_remote_zip.py \
  data/charades/charades_pilot_subset_500.csv \
  --output-dir data/charades/videos_480p \
  --summary-output data/charades/charades_download_500_summary.json
```

预期：

```text
processed 500 videos, available locally: 500
```

如果网络中断，可以重复执行；已存在视频会跳过。

## 3. 切 500 个 clip

```bash
python scripts/slice_charades_clips.py \
  data/charades/charades_pilot_subset_500.csv \
  --videos-dir data/charades/videos_480p \
  --clips-dir data/charades/clips_480p_500 \
  --output-manifest outputs/charades_clips_500.csv \
  --summary-output outputs/charades_clips_500_summary.json
```

预期：

```text
sliced/present: 500
```

## 4. 提取 SlowFast 特征

建议先跑 limit1：

```bash
python scripts/tal/extract_slowfast_features.py \
  outputs/charades_clips_500.csv \
  --backend slowfast-r50 \
  --output-dir data/charades/features_500 \
  --summary-output outputs/tal_features_slowfast_500_limit1_summary.json \
  --feature-manifest-output data/charades/features_manifest_500_limit1.json \
  --device cuda \
  --limit 1 \
  --overwrite
```

limit1 成功后跑全量：

```bash
python scripts/tal/extract_slowfast_features.py \
  outputs/charades_clips_500.csv \
  --backend slowfast-r50 \
  --output-dir data/charades/features_500 \
  --summary-output outputs/tal_features_slowfast_500_summary.json \
  --feature-manifest-output data/charades/features_manifest_500.json \
  --device cuda \
  --overwrite
```

预期：

```text
ok = 500
feature_dim = 2304
```

## 5. Dataloader smoke

```bash
python scripts/tal/dataloader_smoke.py \
  outputs/charades_clips_500.csv \
  --feature-manifest data/charades/features_manifest_500.json \
  --feat-num-frames 32 \
  --downsample-rate 1 \
  --summary-output outputs/tal_dataloader_slowfast_500_summary.json
```

如果 summary 中 `ok` 接近 500，且没有大量 `missing_feature_entry`、`feature_load_failed` 或 `segments_outside_feature_grid`，即可训练。

## 6. 训练 ActionFormer 500-clip checkpoint

```bash
cd ~/ActionLens

python external/actionformer_release/train.py \
  external/actionformer_release/configs/charades_slowfast_500.yaml \
  --output charades_500v \
  -p 1 \
  -c 1
```

checkpoint 目录预期：

```text
external/actionformer_release/ckpt/charades_slowfast_500_charades_500v/
```

不要上传 `.pth.tar` 权重到 Git。权重只保留在 4090 本机。

## 7. 导出 val/test predictions

val：

```bash
cd ~/ActionLens

python external/actionformer_release/eval.py \
  external/actionformer_release/configs/charades_slowfast_500.yaml \
  external/actionformer_release/ckpt/charades_slowfast_500_charades_500v \
  -epoch 60 \
  --saveonly \
  -p 1

cp external/actionformer_release/ckpt/charades_slowfast_500_charades_500v/eval_results.pkl \
  outputs/actionformer_500_epoch060_val_predictions.pkl
```

test：

```bash
python external/actionformer_release/eval.py \
  external/actionformer_release/configs/charades_slowfast_500_test_export.yaml \
  external/actionformer_release/ckpt/charades_slowfast_500_charades_500v \
  -epoch 60 \
  --saveonly \
  -p 1

cp external/actionformer_release/ckpt/charades_slowfast_500_charades_500v/eval_results.pkl \
  outputs/actionformer_500_epoch060_test_predictions.pkl
```

如果训练只跑到较早 epoch，把命令里的 `-epoch 60` 改成实际 epoch。

## 8. 需要回传的文件

基础数据处理结果：

```text
data/charades/charades_download_500_summary.json
outputs/charades_clips_500.csv
outputs/charades_clips_500_summary.json
outputs/tal_features_slowfast_500_summary.json
data/charades/features_manifest_500.json
outputs/tal_dataloader_slowfast_500_summary.json
```

训练与推理结果：

```text
outputs/actionformer_500_epoch060_val_predictions.pkl
outputs/actionformer_500_epoch060_test_predictions.pkl
```

可选日志：

```text
external/actionformer_release/ckpt/charades_slowfast_500_charades_500v/config.txt
external/actionformer_release/ckpt/charades_slowfast_500_charades_500v/log.txt
```

不要回传：

```text
*.pth.tar
external/actionformer_release/ckpt/charades_slowfast_500_charades_500v/logs/
data/charades/videos_480p/*.mp4
data/charades/clips_480p_500/
data/charades/features_500/
```

## 9. 回传后本机分析流程

回传 prediction pkl 后，本机执行：

```bash
python scripts/tal/build_charades_clip_eval_gt.py \
  --split val \
  --clips-manifest outputs/charades_clips_500.csv \
  --feature-manifest data/charades/features_manifest_500.json \
  --split-folder data/charades/splits_500 \
  --output outputs/charades_clip_eval_500_val.json

python scripts/tal/build_charades_clip_eval_gt.py \
  --split test \
  --clips-manifest outputs/charades_clips_500.csv \
  --feature-manifest data/charades/features_manifest_500.json \
  --split-folder data/charades/splits_500 \
  --output outputs/charades_clip_eval_500_test.json
```

然后分别跑：

```text
convert predictions -> top-k recall -> A/B -> filtered VLM review -> aggregate
```

重点看 test split：

```text
TAL top-k action recall 是否 >= 60%
TAL top-k IoU>=0.5 recall 是否 >= 45%
TAL+VLM overlap recall 是否稳定高于 Stage2/VLM baseline
```
