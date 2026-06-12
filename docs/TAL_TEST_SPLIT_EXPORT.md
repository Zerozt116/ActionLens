# TAL Test Split Prediction Export

目的：使用现有 `epoch034` ActionFormer checkpoint，不重新训练，只导出 Charades 50-clip pilot 中 test split 的预测结果，用于未见样本泛化验证。

## 在 4090 机器执行

```bash
cd ~/ActionLens
git pull --rebase origin main

python external/actionformer_release/eval.py \
  external/actionformer_release/configs/charades_slowfast_test_export.yaml \
  ckpt/charades_slowfast_charades_50v_smoke \
  -epoch 34 \
  --saveonly \
  -p 1

cp ckpt/charades_slowfast_charades_50v_smoke/eval_results.pkl \
  outputs/actionformer_epoch034_test_predictions.pkl
```

## 回传文件

```text
outputs/actionformer_epoch034_test_predictions.pkl
```

## 本机回传后处理

```bash
python scripts/tal/build_charades_clip_eval_gt.py \
  --split test \
  --output outputs/charades_clip_eval_test.json

python scripts/tal/convert_actionformer_predictions.py \
  outputs/actionformer_epoch034_test_predictions.pkl \
  --eval-groundtruth outputs/charades_clip_eval_test.json \
  --csv-output outputs/actionformer_epoch034_test_predictions.csv \
  --json-output outputs/actionformer_epoch034_test_predictions.json \
  --topk-output outputs/actionformer_epoch034_test_topk.json \
  --top-k 10

python scripts/tal/compare_tal_stage2_vlm.py \
  --tal-topk outputs/actionformer_epoch034_test_topk.json \
  --batch-root outputs/charades_clip_batch_50 \
  --clips-manifest outputs/charades_clips_50.csv \
  --output-json outputs/tal_stage2_vlm_ab_epoch034_test.json \
  --output-md outputs/tal_stage2_vlm_ab_epoch034_test.md \
  --scope-note "current TAL export covers the ActionFormer test split (11 clips); this split was not used for training."
```

如果 test A/B 中存在足够的 TAL-only 候选，再继续执行 filtered VLM review：

```bash
python scripts/tal/review_tal_proposals_with_vlm.py \
  --ab-report outputs/tal_stage2_vlm_ab_epoch034_test.json \
  --predictions-csv outputs/actionformer_epoch034_test_predictions.csv \
  --outcomes gt_hit_by_tal_only tal_only_non_gt \
  --limit 999 \
  --overwrite \
  --output-root outputs/tal_vlm_review_epoch034_test_filtered \
  --summary-output outputs/tal_vlm_review_epoch034_test_filtered_summary.json \
  --report-md outputs/tal_vlm_review_epoch034_test_filtered_summary.md \
  --fusion-output outputs/tal_vlm_fusion_epoch034_test_filtered.json \
  --fusion-report-md outputs/tal_vlm_fusion_epoch034_test_filtered.md

python scripts/tal/aggregate_tal_vlm_fusion.py \
  --ab-report outputs/tal_stage2_vlm_ab_epoch034_test.json \
  --tal-vlm-fusion outputs/tal_vlm_fusion_epoch034_test_filtered.json \
  --output-json outputs/tal_vlm_fusion_aggregate_epoch034_test_filtered.json \
  --output-md outputs/tal_vlm_fusion_aggregate_epoch034_test_filtered.md
```

## 预期耗时

test split 约 11 个 clip。参考 val split 7 个 clip 约 17 秒，预计 test split 导出约 20-40 秒，取决于 GPU 与 IO 状态。

## 注意

- 该命令不会训练，只做推理和保存预测。
- `eval.py --saveonly` 固定写入 `ckpt/.../eval_results.pkl`，所以执行后必须立刻复制到 `outputs/actionformer_epoch034_test_predictions.pkl`。
- 不要把 checkpoint 权重提交到 Git；只需要回传 pkl。
