# ActionLens 阶段报告：Charades 50 到 200 的 TAL/VLM 验证

更新时间：2026-06-13

## 结论

当前路线是有效的：用 Charades 扩展训练 ActionFormer/TAL proposal，再用事件级 VLM 做复核，可以显著提高开放行为检测的候选召回，并能过滤一部分 TAL 误报。

最关键的变化是 50-clip 到 200-clip：

| 实验 | Test clips | GT instances | TAL top-k action recall | TAL top-k IoU>=0.5 recall |
| --- | ---: | ---: | ---: | ---: |
| 50-clip epoch034 | 11 | 30 | 16.67% | 10.00% |
| 200-clip epoch050 | 34 | 96 | 54.17% | 39.58% |

这说明早期瓶颈不是 VLM 本身，而是 TAL detector 的训练规模太小。扩到 200 clips 后，proposal 的语义命中和时间定位都明显改善。

## 当前方案

项目目前采用两层结构：

1. **TAL proposal 层**
   ActionFormer 使用 SlowFast-R50 特征，输出动作类别、时间片段和置信度。

2. **VLM verifier 层**
   对 TAL 候选进行事件级抽帧复核，判断动作是否真的出现，并给出证据文本。

这条路线适合“尽量泛化”的目标：TAL 负责把长视频切成可能发生行为的时间段，VLM 负责语义确认，最终输出可解释事件。

## 200-clip 数据链路

200-clip 实验覆盖 20 个核心动作，每类 10 个 clip。

```text
download_200: downloaded = 150, exists = 50
clips_200: sliced = 200
features_200: extracted = 200
dataloader_smoke_200: ok = 200
feature_dim = 2304
total_feature_frames = 12569
dataloader_total_segments = 611
```

split：

```text
train = 137
val = 29
test = 34
```

## 200-clip 检测结果

val：

```text
gt_instances = 93
topk_action_hits = 59 / 93 = 63.44%
topk_iou_positive_hits = 56 / 93 = 60.22%
topk_iou_at_0_5_hits = 34 / 93 = 36.56%
```

test：

```text
gt_instances = 96
topk_action_hits = 52 / 96 = 54.17%
topk_iou_positive_hits = 52 / 96 = 54.17%
topk_iou_at_0_5_hits = 38 / 96 = 39.58%
```

表现较好的动作包括：

| Action | Test GT | Action recall | IoU>=0.5 recall |
| --- | ---: | ---: | ---: |
| Sitting in a chair | 6 | 100.00% | 100.00% |
| Drinking from a cup/glass/bottle | 6 | 83.33% | 66.67% |
| Holding a phone/camera | 8 | 75.00% | 75.00% |
| Holding a cup/glass/bottle | 5 | 80.00% | 60.00% |

薄弱动作包括：

| Action | Test GT | Action recall | IoU>=0.5 recall |
| --- | ---: | ---: | ---: |
| Sitting on the floor | 3 | 0.00% | 0.00% |
| Taking a cup/glass/bottle from somewhere | 4 | 0.00% | 0.00% |
| Playing with a phone/camera | 6 | 16.67% | 0.00% |
| Someone is eating something | 11 | 54.55% | 18.18% |

这说明下一轮扩数据要重点观察细粒度容器动作、手机动作和姿态转换动作。

## TAL / Stage2 / VLM A/B

当前 overlap A/B 只覆盖已有 `outputs/charades_clip_batch_50` 的 Stage2/VLM 输出，不是完整 200 clips。

在 test overlap 上：

```text
evaluated_clip_count = 12
gt_action_instances = 33
TAL top-k hits = 18 / 33 = 54.55%
Stage2/VLM hits = 9 / 33 = 27.27%
tal_only_gt_hits = 12
stage2_vlm_only_gt_hits = 3
missed_by_both = 12
```

TAL 在相同样本上已经明显强于原 Stage2/VLM baseline。

## TAL+VLM 融合效果

对 test overlap 中的 TAL-only 候选做 VLM 复核：

```text
candidates_before_filter = 85
candidates_after_filter = 16
completed = 16
failed = 0
vlm_confirmed = 4
vlm_rejected = 12
```

融合后：

```text
baseline_stage2_vlm_gt_hits = 9
tal_vlm_added_gt_hits = 3
combined_gt_hits = 12
baseline_recall_proxy = 27.27%
combined_recall_proxy = 36.36%
absolute_recall_gain = 9.09 percentage points
relative_hit_gain = 33.33%
```

新增 GT hit：

```text
5B8M5_c156_t0.00_30.71 / sitting_on_chair
8BG1T_c106_t5.60_45.46 / sitting_down
K8AUX_c110_t0.00_29.54 / holding_drink_container
```

VLM verifier 的意义也很清楚：16 个候选中拒绝 12 个，能压住 TAL-only 误报。

## 当前风险

- 200 clips 仍然偏小，容易受 split 和动作分布影响。
- overlap A/B 只覆盖 12 个 test clips，不能代表完整 200 test。
- 当前类别仍是 20 个核心动作，并非完整 Charades 157 类。
- VLM 复核依赖抽帧，短动作或遮挡动作可能被漏判。

## 下一步

建议走 500-clip 扩展：

1. 生成并冻结 500-clip 子集、split、训练配置。
2. 在 4090 上下载、切片、提取 SlowFast 特征。
3. 训练 `charades_slowfast_500.yaml` 到 60 epoch。
4. 回传 val/test pkl 后，按同一套脚本做 top-k recall、A/B、VLM review、fusion aggregate。
5. 若 500 test 继续提升，再考虑扩到 1000 clips 或逐步增加动作类别。

阶段目标：

```text
500-clip test top-k action recall >= 60%
500-clip test top-k IoU>=0.5 recall >= 45%
TAL+VLM overlap recall proxy 稳定高于 Stage2/VLM baseline 10pp 以上
```
