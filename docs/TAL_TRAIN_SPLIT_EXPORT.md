# TAL Train Split Prediction Export

目的：使用现有 `epoch034` ActionFormer checkpoint，不重新训练，只导出 Charades 50-clip pilot 中 train split 的预测结果，扩大 TAL / Stage2 / VLM overlap 分析样本。

## 在 4090 机器执行

```bash
cd ~/ActionLens
git pull --rebase origin main

python external/actionformer_release/eval.py \
  external/actionformer_release/configs/charades_slowfast_train_export.yaml \
  ckpt/charades_slowfast_charades_50v_smoke \
  -epoch 34 \
  --saveonly \
  -p 1

cp ckpt/charades_slowfast_charades_50v_smoke/eval_results.pkl \
  outputs/actionformer_epoch034_train_predictions.pkl
```

## 回传文件

```text
outputs/actionformer_epoch034_train_predictions.pkl
```

## 预期耗时

train split 约 32 个 clip。参考 val split 7 个 clip 约 17 秒，预计 train split 导出约 1-2 分钟，取决于 GPU 与 IO 状态。

## 注意

- 该命令不会训练，只做推理和保存预测。
- `eval.py --saveonly` 固定写入 `ckpt/.../eval_results.pkl`，所以执行后必须立刻复制到 `outputs/actionformer_epoch034_train_predictions.pkl`。
- 不要把 checkpoint 权重提交到 Git；只需要回传 pkl。
