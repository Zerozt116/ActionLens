# Charades 200-Clip Training Plan

目的：把当前 50-clip pilot 扩到 200 clips，重新训练 ActionFormer/TAL proposal，验证 detector 泛化是否改善。

## 本轮新增资产

```text
data/charades/charades_pilot_subset_200.csv
data/charades/charades_pilot_subset_200.json
data/charades/charades_pilot_subset_200_summary.json
data/charades/splits_200/train.txt
data/charades/splits_200/val.txt
data/charades/splits_200/test.txt
data/charades/splits_200/summary.json
external/actionformer_release/configs/charades_slowfast_200.yaml
external/actionformer_release/configs/charades_slowfast_200_train_export.yaml
external/actionformer_release/configs/charades_slowfast_200_test_export.yaml
```

200-clip split：

```text
train = 137
val = 29
test = 34
```

动作覆盖：

```text
20 actions x 10 clips/action = 200 clips
```

## 1. 拉取最新代码

在 4090 机器执行：

```bash
cd ~/ActionLens
git pull --rebase origin main
```

如果 pull 被未跟踪文件挡住，先把对应文件移动到 `/tmp/actionlens_backup/`，再重新 pull。

## 2. 下载缺失视频

当前 200 清单复用原 50 个视频，并新增 150 个 Charades 视频。执行：

```bash
python scripts/download_charades_remote_zip.py \
  data/charades/charades_pilot_subset_200.csv \
  --output-dir data/charades/videos_480p \
  --summary-output data/charades/charades_download_200_summary.json
```

预期：

```text
processed 200 videos, available locally: 200
```

如果网络中断，可以重复执行；已存在视频会跳过。

## 3. 切 200 个 clip

```bash
python scripts/slice_charades_clips.py \
  data/charades/charades_pilot_subset_200.csv \
  --videos-dir data/charades/videos_480p \
  --clips-dir data/charades/clips_480p_200 \
  --output-manifest outputs/charades_clips_200.csv \
  --summary-output outputs/charades_clips_200_summary.json
```

预期：

```text
sliced/present: 200
```

## 4. 提取 SlowFast 特征

```bash
python scripts/tal/extract_slowfast_features.py \
  outputs/charades_clips_200.csv \
  --backend slowfast-r50 \
  --output-dir data/charades/features_200 \
  --summary-output outputs/tal_features_slowfast_200_summary.json \
  --feature-manifest-output data/charades/features_manifest_200.json \
  --device cuda \
  --overwrite
```

预期：

```text
ok = 200
feature_dim = 2304
```

## 5. Dataloader smoke

```bash
python scripts/tal/dataloader_smoke.py \
  external/actionformer_release/configs/charades_slowfast_200.yaml \
  --mode train \
  --index 0 \
  --summary-output outputs/tal_dataloader_slowfast_200_train_summary.json

python scripts/tal/dataloader_smoke.py \
  external/actionformer_release/configs/charades_slowfast_200.yaml \
  --mode eval \
  --index 0 \
  --summary-output outputs/tal_dataloader_slowfast_200_val_summary.json
```

两条 smoke 都通过后再训练。

## 6. 训练 ActionFormer 200-clip checkpoint

```bash
cd ~/ActionLens/external/actionformer_release

python ./train.py ./configs/charades_slowfast_200.yaml \
  --output charades_200v \
  -p 1 \
  -c 1
```

checkpoint 目录预期：

```text
external/actionformer_release/ckpt/charades_slowfast_200_charades_200v/
```

不要上传 `.pth.tar` 权重到 Git。权重只保留在 4090 本机。

## 7. 导出 val/test predictions

val：

```bash
cd ~/ActionLens

python external/actionformer_release/eval.py \
  external/actionformer_release/configs/charades_slowfast_200.yaml \
  external/actionformer_release/ckpt/charades_slowfast_200_charades_200v \
  -epoch 50 \
  --saveonly \
  -p 1

cp external/actionformer_release/ckpt/charades_slowfast_200_charades_200v/eval_results.pkl \
  outputs/actionformer_200_epoch050_val_predictions.pkl
```

test：

```bash
python external/actionformer_release/eval.py \
  external/actionformer_release/configs/charades_slowfast_200_test_export.yaml \
  external/actionformer_release/ckpt/charades_slowfast_200_charades_200v \
  -epoch 50 \
  --saveonly \
  -p 1

cp external/actionformer_release/ckpt/charades_slowfast_200_charades_200v/eval_results.pkl \
  outputs/actionformer_200_epoch050_test_predictions.pkl
```

如果训练只跑到某个较早 epoch，把命令里的 `-epoch 50` 改成实际 epoch。

## 8. 需要回传的文件

基础数据处理结果：

```text
data/charades/charades_download_200_summary.json
outputs/charades_clips_200.csv
outputs/charades_clips_200_summary.json
outputs/tal_features_slowfast_200_summary.json
data/charades/features_manifest_200.json
outputs/tal_dataloader_slowfast_200_train_summary.json
outputs/tal_dataloader_slowfast_200_val_summary.json
```

训练与推理结果：

```text
outputs/actionformer_200_epoch050_val_predictions.pkl
outputs/actionformer_200_epoch050_test_predictions.pkl
```

可选日志：

```text
external/actionformer_release/ckpt/charades_slowfast_200_charades_200v/config.txt
external/actionformer_release/ckpt/charades_slowfast_200_charades_200v/log.txt
```

不要回传：

```text
*.pth.tar
external/actionformer_release/ckpt/charades_slowfast_200_charades_200v/logs/
```

## 9. 回传后本机分析流程

回传 prediction pkl 后，本机将执行：

```bash
python scripts/tal/build_charades_clip_eval_gt.py \
  --split val \
  --clips-manifest outputs/charades_clips_200.csv \
  --feature-manifest data/charades/features_manifest_200.json \
  --split-folder data/charades/splits_200 \
  --output outputs/charades_clip_eval_200_val.json

python scripts/tal/build_charades_clip_eval_gt.py \
  --split test \
  --clips-manifest outputs/charades_clips_200.csv \
  --feature-manifest data/charades/features_manifest_200.json \
  --split-folder data/charades/splits_200 \
  --output outputs/charades_clip_eval_200_test.json
```

然后分别跑：

```text
convert predictions -> A/B -> filtered VLM review -> aggregate
```

最终重点看 test split：

```text
TAL top-k recall 是否高于 17.86%
TAL+VLM recall 是否稳定高于 Stage2/VLM baseline
VLM confirmed 是否不再主要依赖低阈值
```
