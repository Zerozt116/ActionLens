# Charades 测试结果报告

更新时间：2026-06-10

## 1. 测试目的

本报告总结当前 ActionLens 方案在 Charades 数据上的一次阶段性测试结果。

当前被测试的是：

```text
Charades clip
-> Stage2 规则检测
-> VLM 全视频/事件复核
-> 结果融合
```

其中 TAL / ActionFormer 目前只完成了数据适配与 smoke test，还没有进入真实 SlowFast 特征训练，因此本报告的主要结果来自 **规则 + VLM 融合方案**。

## 2. 数据与设置

测试数据：

- 数据集：Charades
- 测试规模：50 个 clip
- 视频数：50 个 unique videos
- 动作覆盖：19 个 canonical actions
- 输入清单：`outputs/charades_clips_50.csv`
- 批量输出目录：`outputs/charades_clip_batch_50/`

主要输出文件：

- `outputs/charades_clip_batch_50/batch_summary.json`
- `outputs/charades_clip_batch_50/aggregation_report.json`
- `outputs/charades_clip_batch_50/aggregation_report.md`
- `outputs/charades_clip_batch_50/pending_event_review_summary.json`

## 3. 总体结果

### 3.1 批处理状态

| 指标 | 数值 |
| --- | ---: |
| total_clips | 50 |
| stage2_ok | 50 |
| vlm_ok | 50 |
| fused_ok | 50 |
| stage2_event_hits | 19 |
| stage2_event_rate | 0.38 |
| vlm_present_hits | 26 |
| vlm_present_rate | 0.52 |

结论：

- 50 个 clip 全部完成 Stage2、VLM 和融合流程。
- Stage2 规则在 50 个 clip 中命中 19 个，命中率为 38%。
- VLM 在 50 个 clip 中确认存在目标行为 26 个，确认率为 52%。

### 3.2 融合结果

| 指标 | 数值 |
| --- | ---: |
| total_fused_events | 115 |
| unique_videos | 50 |
| unique_actions | 19 |
| unique_actions_with_outputs | 16 |
| final_events | 20 |
| semantic_candidates | 24 |
| pending_events | 0 |
| rejected_events | 71 |

解释：

- `final_events`：规则、Charades 标注、VLM 证据较一致的最终事件。
- `semantic_candidates`：规则没有稳定检出，但 Charades/VLM 提供语义证据的候选事件。
- `rejected_events`：规则疑似误报，经 VLM 或融合逻辑否定。
- `pending_events`：需要进一步复核但尚未处理的事件；本轮已全部清零。

## 4. 动作级别结果

### 4.1 支持较好的动作

| Action | Charades GT clips | Final | Semantic | Stage2 hits | VLM present | Rejected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| drinking_water | 11 | 15 | 1 | 59 | 8 | 44 |
| talking_on_phone | 3 | 5 | 2 | 32 | 0 | 27 |
| eating | 13 | 0 | 4 | 0 | 0 | 0 |
| holding_phone | 11 | 0 | 3 | 0 | 0 | 0 |
| looking_at_phone | 7 | 0 | 2 | 0 | 0 | 0 |
| sitting_on_chair | 10 | 0 | 2 | 0 | 0 | 0 |

观察：

- 当前规则最能产生时间段事件的是 `drinking_water` 和 `talking_on_phone`。
- `eating`、`holding_phone`、`looking_at_phone` 等动作主要依赖 Charades 标注和 VLM 语义补充，还没有形成稳定的规则事件。
- `drinking_water` 和 `talking_on_phone` 虽然 final 数量较高，但 rejected 也高，说明规则召回较强但误报明显。

### 4.2 暂无输出的动作

| Action | Charades GT clips | Final | Semantic | Stage2 hits | Rejected |
| --- | ---: | ---: | ---: | ---: | ---: |
| running | 4 | 0 | 0 | 0 | 0 |
| sitting_on_floor | 7 | 0 | 0 | 0 | 0 |
| standing_up | 14 | 0 | 0 | 0 | 0 |

这些动作对当前规则体系不友好：

- `running` 更依赖运动轨迹和速度特征。
- `standing_up` / `sitting_on_floor` 更依赖姿态时序变化。
- 这类动作更适合交给 TAL / ActionFormer 或更强的姿态时序模型处理。

## 5. 与 batch20 的对比

| 指标 | batch20 | batch50 | 变化 |
| --- | ---: | ---: | ---: |
| total_clips | 20 | 50 | +30 |
| stage2_event_hits | 9 | 19 | +10 |
| stage2_event_rate | 0.45 | 0.38 | -0.07 |
| vlm_present_hits | 5 | 26 | +21 |
| vlm_present_rate | 0.25 | 0.52 | +0.27 |
| total_fused_events | 31 | 115 | +84 |
| unique_actions_with_outputs | 4 | 16 | +12 |
| final_events | 6 | 20 | +14 |
| semantic_candidates | 4 | 24 | +20 |
| rejected_events | 21 | 71 | +50 |

结论：

- 扩展到 50 个 clip 后，动作覆盖明显提升：有输出的动作从 4 类增加到 16 类。
- VLM 的语义补充能力更明显，`vlm_present_rate` 从 25% 提升到 52%。
- Stage2 规则命中率略降，说明当前规则泛化不足。
- rejected 数量增加，主要来自 drink / phone 规则误报。

## 6. 当前方案结论

当前 Charades batch50 测试说明：

1. **工程链路已经跑通**：50 个 clip 全部完成 Stage2、VLM、融合与聚合。
2. **规则基线可作为初始事件生成器**：对喝水、打电话等人-物交互动作有效，但误报较多。
3. **VLM 对泛化有明显帮助**：能补充规则无法覆盖的语义动作，也能否定一部分规则误报。
4. **规则方法不能覆盖全部 Charades 动作**：尤其是 running、standing_up、sitting_on_floor 等姿态/运动时序动作。
5. **下一阶段需要 TAL / ActionFormer**：当前 ActionFormer 已完成 Charades adapter、split、feature smoke 和 GPU 执行包，待 4090 上生成真实 SlowFast 特征后进入训练 smoke。

## 7. TAL / ActionFormer 当前状态

TAL 侧已经完成：

- `external/actionformer_release/libs/datasets/charades.py`
- `external/actionformer_release/configs/charades_smoke.yaml`
- `external/actionformer_release/configs/charades_slowfast.yaml`
- `scripts/tal/extract_slowfast_features.py`
- `scripts/tal/check_slowfast_backend.py`
- `docs/TAL_4090_EXECUTION_PACKAGE.md`

本机 smoke 结果：

| 指标 | 数值 |
| --- | ---: |
| dataloader smoke total | 50 |
| dataloader ok | 50 |
| feature dim | 2304 |
| feature length range | 38 - 207 |
| total GT segments | 148 |
| ActionFormer sample feats | `[2304, 91]` |
| ActionFormer sample segments | `[5, 2]` |
| train-loss smoke final_loss | 0.30324 |

注意：

- 当前 `features_smoke` 只是格式验证特征，不能代表真实识别性能。
- ActionFormer 尚未训练，不能用 smoke 输出作为模型效果结论。
- 下一步应在 4090 上先跑 `check_slowfast_backend.py --device cuda`，再做 `slowfast-r50 --limit 1`。

## 8. 后续建议

短期建议：

1. 在 4090 上修好 PyTorch CUDA / PyTorchVideo 环境。
2. 跑 SlowFast 后端自检。
3. 先提取 1 个 Charades clip 的真实 SlowFast 特征。
4. limit1 成功后提取 50 个真实特征。
5. 跑真实特征 dataloader smoke。
6. 进入 ActionFormer 1 epoch 训练 smoke。

中期建议：

- 用 ActionFormer 作为 Stage3 时序动作定位模块。
- 将 Stage3 输出接入现有融合逻辑，形成：

```text
Stage2 rules + Stage3 TAL + VLM review + Charades GT comparison
```

预期收益：

- 提高 running、standing_up、sitting_on_floor 等动作覆盖。
- 降低 drink / phone 规则对 VLM 复核的依赖。
- 形成规则基线 vs TAL 模型的可量化 A/B 对比。
