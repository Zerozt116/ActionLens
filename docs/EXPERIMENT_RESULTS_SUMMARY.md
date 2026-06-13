# ActionLens 当前实验结果汇总

更新时间：2026-06-13

## 1. 当前课题路线

ActionLens 当前目标是做一个尽量泛化的视频行为分析系统。系统不只针对“喝水”，而是希望对日常动作进行开放式检测、定位和解释。

目前采用两阶段方案：

1. **TAL proposal**
   使用 SlowFast-R50 提取视频特征，再用 ActionFormer 输出候选动作类别、时间段和置信度。

2. **VLM verifier**
   对 TAL 候选片段抽帧，用多模态大模型复核动作是否真实发生，并生成证据文本。

阶段性判断：早期效果差的主要瓶颈不是 VLM，而是 TAL detector 的训练数据太少。把 Charades 训练集从 50 clips 扩到 200 clips 后，TAL proposal 的召回明显提升。

## 2. 数据与配置进展

| 阶段 | 数据规模 | 类别覆盖 | 训练状态 | 说明 |
| --- | ---: | ---: | --- | --- |
| Charades 50 | 50 clips | 20 actions | 已训练 | 第一版 smoke / pilot |
| Charades 200 | 200 clips | 20 actions | 已训练到 epoch 50 | 当前主要有效结果 |
| Charades 500 | 500 clips | 20 actions | 已准备，未训练 | 下一轮扩展 |

Charades 200 数据链路：

```text
download_200: downloaded = 150, exists = 50
clips_200: sliced = 200
features_200: extracted = 200
dataloader_smoke_200: ok = 200
feature_dim = 2304
total_feature_frames = 12569
dataloader_total_segments = 611
```

Charades 200 split：

```text
train = 137
val = 29
test = 34
```

Charades 500 已准备：

```text
selected_videos = 500
actions = 20
samples_per_action = 25
scenes = 16
average_clip_duration_seconds = 24.8136

train = 347
val = 74
test = 79
```

## 3. TAL 检测结果

统一口径使用 `scripts/tal/summarize_tal_topk_recall.py` 统计 top-k recall proxy。

| 实验 | Split | Clips | GT instances | Action recall | IoU>0 recall | IoU>=0.5 recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 50-clip epoch034 | test | 11 | 30 | 16.67% | 16.67% | 10.00% |
| 200-clip epoch050 | val | 29 | 93 | 63.44% | 60.22% | 36.56% |
| 200-clip epoch050 | test | 34 | 96 | 54.17% | 54.17% | 39.58% |

关键结论：

- 50 到 200 后，test action recall 从 `16.67%` 提升到 `54.17%`。
- test IoU>=0.5 recall 从 `10.00%` 提升到 `39.58%`。
- 这说明扩数据训练 ActionFormer 对泛化有显著帮助，且提升不只是类别命中，时间定位也变好了。

## 4. 200-Clip Test 分动作表现

表现较好的动作：

| Action | GT | Action recall | IoU>=0.5 recall |
| --- | ---: | ---: | ---: |
| Sitting in a chair | 6 | 100.00% | 100.00% |
| Drinking from a cup/glass/bottle | 6 | 83.33% | 66.67% |
| Holding a phone/camera | 8 | 75.00% | 75.00% |
| Holding a cup/glass/bottle of something | 5 | 80.00% | 60.00% |
| Talking on a phone/camera | 3 | 66.67% | 66.67% |

薄弱动作：

| Action | GT | Action recall | IoU>=0.5 recall |
| --- | ---: | ---: | ---: |
| Sitting on the floor | 3 | 0.00% | 0.00% |
| Taking a cup/glass/bottle from somewhere | 4 | 0.00% | 0.00% |
| Playing with a phone/camera | 6 | 16.67% | 0.00% |
| Someone is eating something | 11 | 54.55% | 18.18% |
| Someone is going from standing to sitting | 7 | 57.14% | 14.29% |

解释：

- “坐在椅子”“喝水”“拿手机/拿杯瓶”这类视觉模式比较明确，200 版已经能较好召回。
- “拿起/放下杯瓶”“玩手机”“从站到坐/站起”更依赖细粒度时序，仍需要更多样本或更强时序建模。
- “吃东西”类别内部差异较大，语义命中尚可，但时间边界不稳定。

## 5. TAL 与 Stage2/VLM A/B

目前 A/B 只覆盖已有 `outputs/charades_clip_batch_50` 的 Stage2/VLM 输出，因此是 overlap 子集，不是完整 200 test。

200 epoch050 test overlap：

```text
evaluated_clip_count = 12
gt_action_instances = 33
TAL top-k hits = 18 / 33 = 54.55%
Stage2/VLM hits = 9 / 33 = 27.27%
tal_only_gt_hits = 12
stage2_vlm_only_gt_hits = 3
missed_by_both = 12
```

结论：

- 在相同 overlap 样本上，TAL top-k recall 明显高于原 Stage2/VLM baseline。
- Stage2/VLM 仍能补到 3 个 TAL 没命中的 GT，因此后续不是完全替换，而是融合。

## 6. TAL+VLM 复核与融合

对 200 epoch050 test overlap 中的 TAL-only 候选做 VLM 复核。

VLM review：

```text
candidates_before_filter = 85
candidates_after_filter = 16
completed = 16
failed = 0
vlm_confirmed = 4
vlm_rejected = 12
```

融合效果：

```text
gt_action_instances = 33
baseline_stage2_vlm_gt_hits = 9
tal_vlm_confirmed_events = 4
tal_vlm_confirmed_gt_hits = 3
tal_vlm_added_gt_hits = 3
tal_vlm_rejected_events = 12
combined_gt_hits = 12
baseline_recall_proxy = 27.27%
combined_recall_proxy = 36.36%
absolute_recall_gain = 9.09 percentage points
relative_hit_gain = 33.33%
```

新增 GT hit：

| Clip | Action | 说明 |
| --- | --- | --- |
| `5B8M5_c156_t0.00_30.71` | sitting_on_chair | TAL 定位，VLM 确认坐在椅子上 |
| `8BG1T_c106_t5.60_45.46` | sitting_down | TAL 定位，VLM 确认从站到坐 |
| `K8AUX_c110_t0.00_29.54` | holding_drink_container | TAL 定位，VLM 确认手持杯子 |

额外发现：

- `406LH_c097_t1.00_21.00 / holding_drink_container` 被 VLM 确认，但 Charades GT 中没有该标签，属于潜在标注缺失或额外动作发现。
- VLM 拒绝 12 / 16 个候选，说明它对压制 TAL-only 误报有价值。

## 7. 当前阶段结论

目前可以形成三点明确结论：

1. **扩数据是有效方向**
   50 到 200 后，TAL test action recall 从 `16.67%` 到 `54.17%`，IoU>=0.5 recall 从 `10.00%` 到 `39.58%`。

2. **TAL+VLM 比单纯 VLM baseline 更有潜力**
   在 test overlap 上，Stage2/VLM baseline recall proxy 为 `27.27%`，加入 TAL+VLM 复核后到 `36.36%`。

3. **系统已具备可解释输出雏形**
   TAL 给出时间段和类别，VLM 给出确认/拒绝与自然语言证据，适合作为最终系统展示形式。

## 8. 当前不足

- Charades 200 仍偏小，完整泛化结论需要 500 或 1000 clips 验证。
- 当前只覆盖 20 个核心动作，还不是 Charades 全 157 类。
- overlap A/B 的样本只有 12 个 test clips，结论方向明确，但统计量仍小。
- VLM review 依赖抽帧，短时动作、遮挡动作、动作边界附近容易误判。
- 当前指标是 top-k recall proxy，不等同于标准 TAL mAP；标准 mAP 可作为后续补充。

## 9. 下一步实验计划

优先级最高的是跑 Charades 500：

1. 在 4090 上按 `docs/CHARADES_500_TRAINING.md` 下载、切片、提特征。
2. 先确认 `download_500`、`clips_500`、`features_500 limit1` 正常。
3. 全量特征与 dataloader smoke 通过后，训练 `charades_slowfast_500.yaml`。
4. 回传 `actionformer_500_epoch060_val_predictions.pkl` 和 `actionformer_500_epoch060_test_predictions.pkl`。
5. 本机继续做 500 的 top-k recall、Stage2/VLM overlap、TAL+VLM fusion aggregate。

500 阶段目标：

```text
test top-k action recall >= 60%
test top-k IoU>=0.5 recall >= 45%
TAL+VLM overlap recall proxy 稳定高于 Stage2/VLM baseline 10pp 以上
```

如果 500 继续提升，后续再考虑：

- 扩到 1000 clips。
- 从 20 个动作扩到更多 Charades 动作。
- 加入标准 mAP 评估和更完整的误差分析。
- 优化 VLM 抽帧策略，尤其是短动作与边界动作。
