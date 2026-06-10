# Stage3 时序动作定位模型方案

## 背景

当前 Stage2 行为识别基于**手工规则**：`_score_drinking` 和 `_score_phone_call` 两个 `if/elif` 分支，覆盖 2 类动作（喝水、打电话）。在 2026-06-08 的 20 视频批量评估中暴露了三个核心问题（见 `项目进度.md` §31）：

1. **覆盖率不足**：20 个 clip 中 11 个动作没有对应规则（看笔记本、坐、站、走、跑、看电视、吃饭、做饭、走过门等），Stage2 全部报 0 事件。
2. **COCO 物体误检爆量**：EHTB6 上 Stage2 报 9 个 `talking_on_phone`（`cell phone` 类过敏感），VLM 全部纠正。
3. **短促动作漏检**：8BG1T 喝水只持续 0.8s，全视频均匀抽帧 VLM 也没确认。

**手工规则的可解释性强，但表达能力有限**——每加一条新规则需要新代码、新阈值、新测试，扩到 10+ 条规则时维护成本迅速上升。

**方案 C**：用预训练时序动作定位（TAL）模型替代/补充手工规则，给定未裁剪视频直接输出"动作类别 + 起止时间"。论文笔记 `papers/03_ActionFormer_读书笔记.md` 中 ActionFormer 在 THUMOS-14 上 tIoU=0.5 达到 **71.0% mAP**，远超 BMN 的 38.5%。

## 目标

1. **替换规则基线**：用 ActionFormer 在 20 视频 batch 上达到 ≥ 当前规则+VLM 融合的 mAP。
2. **泛化到 157 类**：覆盖 Charades 全部动作类别，零手工规则。
3. **保留融合链路**：ActionFormer 输出直接对接 `fuse_stage2_vlm_events.py`，VLM 仍是 post-hoc 验证。
4. **可解释对比**：规则基线和 ActionFormer 基线并存，论文里做 A/B。

## 非目标

- 实时推理（≤ 1 FPS）：本项目是研究原型，不要求实时
- 跨摄像头人员重识别
- 3D 姿态估计
- 训练比 Charades 更大的数据集（暂用 50 视频 pilot subset）

## 候选模型对比

| 模型 | 范式 | 骨干 | 锚框 | 论文 THUMOS-14 mAP@0.5 | 官方代码 | 备注 |
| --- | --- | --- | --- | ---: | --- | --- |
| **ActionFormer** | 单阶段 | Transformer | 否 | **71.0** | [happyharrycn/actionformer_release](https://github.com/happyharrycn/actionformer_release) | **本方案选用** |
| BMN | 两阶段 | 1D conv | 否 | 38.8 | [uiuctml/BMN-Boundary-Matching-Network](https://github.com/uiuctml/BMN-Boundary-Matching-Network) | 备选 |
| AFSD | 单阶段 | 1D conv | 否 | 55.5 | [MilkClouds/AFSD](https://github.com/MilkClouds/AFSD) | 备选 |

**选 ActionFormer 的理由**：
- 极简设计（序列标注 + 边界回归），单 GPU 训练 4–8 小时可收敛
- 论文在 THUMOS-14 / ActivityNet / EPIC-Kitchens 三个 benchmark 上 SOTA
- 官方代码支持直接加载预训练权重，Charades 上微调工作量小
- Charades 全量 49k 时间段标注格式与 ActionFormer 输入几乎一致，零格式转换

## 完整架构

```
未裁剪视频 (mp4)
    ↓
┌─────────────────────────────────────┐
│ ① 视频特征提取（SlowFast，R50 8x8）│   ← 一次性，离线
│    视频帧 → 时序特征 X[T, 2304]     │      输出: <video_id>.npy
│    缓存到 data/charades/features/    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ ② ActionFormer 模型（加载 Charades  │
│    50 视频 fine-tune 后的权重）      │
│    输入：X[T, 2304]                  │
│    ├ 投影层（2 层 1D conv）          │
│    ├ 7 层局部自注意力 Transformer    │
│    ├ 6 级特征金字塔（步长 1,2,4,8,16,32）│
│    └ 双头解码器：                    │
│       - 分类头：C 路 sigmoid（C=157）│
│       - 回归头：到 onset/offset 距离 │
└─────────────────────────────────────┘
    ↓
  行为时间段 [(start_s, end_s, action, conf), ...]
    ↓
┌─────────────────────────────────────┐
│ ③ Soft-NMS + 类别过滤               │   ← 复用现有思路
│    - 阈值过滤 conf < 0.1             │      新增: scripts/tal_postprocess.py
│    - Soft-NMS 阈值 0.5              │
│    - 按 canonical action 映射        │
└─────────────────────────────────────┘
    ↓
  events.json（与 Stage2 格式兼容）
    ↓
┌─────────────────────────────────────┐
│ ④ 复用 fuse_stage2_vlm_events.py    │   ← 零改动
│    - 读 events.json                  │
│    - 与 Charades 标签对齐            │
│    - 与 VLM 复核融合                 │
└─────────────────────────────────────┘
    ↓
  fused_events.json（最终输出）
```

## 数据准备

### Charades 50 视频 pilot subset

复用 `data/charades/charades_pilot_subset_50.csv` 和 `data/charades/charades_train_pilot_actions.csv`。

| 资产 | 状态 | 来源 |
| --- | --- | --- |
| 视频（480p mp4） | ✅ 已下 50/50 | `download_charades_remote_zip.py` |
| 时间段标注 | ✅ 已有 | `data/charades/annotations/Charades/` |
| ActionFormer 格式 | ✅ 已生成 | `data/charades/tal_annotations.json` |
| Smoke 特征缓存 | ✅ 已生成 | `data/charades/features_smoke/*.npz` |
| 真实 SlowFast 特征 | ❌ 待提取 | 3060/4090 上执行 |

### ActionFormer 输入格式

每行一个时间窗口：

```csv
video_id,start_seconds,end_seconds,action_id,action_name
8BG1T,7.6,43.7,drinking_from_a_cup_glass_bottle,Drinking from a cup/glass/bottle
```

ActionFormer 内部需要：

```json
{
  "8BG1T": {
    "duration": 45.46,
    "annotations": [
      {"segment": [7.6, 43.7], "label_id": 12, "label": "drinking_from_a_cup_glass_bottle"}
    ]
  }
}
```

`scripts/prepare_charades_tal.py` 负责生成这个 JSON，输出到 `data/charades/tal_annotations.json`。

### 训练 / 验证 / 测试划分

| 集合 | 视频数 | 用途 |
| --- | ---: | --- |
| Train | 35 | fine-tune 主体 |
| Val | 5 | 早停 + 超参调优 |
| Test | 10 | 最终评估（与现有 20 视频 batch 重叠部分保留） |

**注意**：20 视频 batch 中的 `8BG1T` / `OINMN` / `024PD` 必须放进 **Test 集**，作为与规则基线的直接对比基准。

## 训练策略

### 预训练权重

ActionFormer 官方提供 THUMOS-14 / ActivityNet / EPIC-Kitchens 的预训练权重。**Charades 与 EPIC-Kitchens 视觉域最接近**（都是日常活动、室内场景），优先用 EPIC-Kitchens 权重初始化。

### 微调超参

| 超参 | 数值 | 备注 |
| --- | ---: | --- |
| 优化器 | AdamW | lr=1e-4, weight_decay=1e-4 |
| 学习率调度 | Cosine | warm-up 5 epoch |
| Batch size | 8 (per GPU) | 特征序列长度不等，用 padding |
| Epoch | 30 | 早停 patience=5 |
| 损失 | Focal Loss (分类) + L1 (回归) | 与原论文一致 |
| 数据增强 | Temporal jitter ±10% | 防止过拟合 |

### 计算资源

- **训练**：单 GPU（A100/V100）4–8 小时；CPU 不可行
- **推理**：单 GPU 0.5–1 秒/视频；CPU 10–20 秒/视频
- **特征提取**：单 GPU 2–4 小时一次性

## 与 VLM 融合的集成

ActionFormer 输出的 `events.json` 与 Stage2 输出的 `events.json` **格式兼容**（都是 `{action, start_seconds, end_seconds, confidence, ...}` 列表）。`fuse_stage2_vlm_events.py` 零修改即可消费。

集成方式：在 `run_charades_clip_batch.py` 中加一个 `--use-tal` 开关，开启后跳过 Stage2，直接跑 ActionFormer 推理。

```python
# run_charades_clip_batch.py 伪代码
if args.use_tal:
    events = run_actionformer(clip_path, tal_model, feature_cache)
    save_events_json(events, stage2_dir / "events.json")  # 复用同一路径
else:
    run_stage2_subprocess(...)  # 现有逻辑
```

VLM 和 Charades alignment 链路完全不用动。

## 实施阶段

### Phase 1：环境 + 推理 smoke test（3–5 天）

**目标**：在已有 3 个视频（8BG1T / OINMN / 024PD）上跑通 ActionFormer 推理。

- 装 PyTorch + mmaction 系列依赖
- 克隆 `actionformer_release`，跑通 THUMOS-14 demo
- 下载预训练权重（EPIC-Kitchens 或 THUMOS-14）
- 用现有 3 个视频的特征，验证能输出时间段
- 写 `scripts/tal_infer_smoke.py`（单视频推理脚本）

**交付物**：
- `scripts/tal_infer_smoke.py`（~50 行）
- 3 个视频的 ActionFormer 输出 JSON
- 一份**对比表**：ActionFormer 检出 vs 规则检出 vs Charades 标注

### Phase 2：数据 + 训练（5–7 天）

**目标**：用 Charades 50 视频 fine-tune ActionFormer。

- `scripts/prepare_charades_tal.py`（~100 行）：Charades CSV → ActionFormer JSON
- `scripts/tal/extract_slowfast_features.py`：50 个 clip 离线提特征
- 改 `actionformer_release/libs/datasets/charades.py`（或新建）
- 跑 fine-tune，监控 val mAP
- 选最佳 checkpoint 导出 `outputs/tal_model/charades_50v_best.pth`

**交付物**：
- `data/charades/tal_annotations.json`
- `data/charades/features_smoke/*.npz`（50 个 smoke 文件，格式验证）
- `data/charades/features/*.npz`（50 个真实 SlowFast 文件，GPU 机器生成）
- `outputs/tal_model/charades_50v_best.pth`
- 训练日志 + loss/mAP 曲线

### Phase 2.1：特征格式 smoke test（已完成）

已新增 `scripts/tal/extract_slowfast_features.py`。当前默认后端是 `cv2-smoke`，用于本机快速验证格式，不用于训练。

输出 schema：

```text
<clip_id>.npz
└── feats: float32, shape = T x 2304
```

该格式与 ActionFormer EPIC-Kitchens 风格 loader 的 `.npz` + `feats` 键一致。真实 SlowFast 后端只需要保持同样输出 schema，即可复用后续训练/推理链路。

已对 batch50 执行：

```bash
./.conda/bin/python scripts/tal/extract_slowfast_features.py \
  outputs/charades_clips_50.csv \
  --output-dir data/charades/features_smoke \
  --summary-output outputs/tal_features_smoke_50_summary.json \
  --feature-manifest-output data/charades/features_smoke_manifest.json
```

结果：

```json
{
  "total": 50,
  "status_counts": {
    "exists": 3,
    "extracted": 47
  },
  "feature_dims": [2304],
  "total_feature_frames": 3743,
  "backend": "cv2-smoke"
}
```

### Phase 2.2：dataloader smoke test（已完成）

已新增 `scripts/tal/dataloader_smoke.py`，验证：

```text
outputs/charades_clips_50.csv
+ data/charades/tal_annotations.json
+ data/charades/features_smoke_manifest.json
+ data/charades/features_smoke/*.npz
-> ActionFormer-style data dict
```

校验内容：

- `.npz` 中 `feats` 可读取。
- `feats` 从 `T x C` 转成 `C x T`。
- Charades 全视频秒级标注按 clip 起止时间截断成 clip-local segment。
- clip-local segment 按 ActionFormer 公式转换到 feature grid。
- labels 与 `tal_annotations.json` 的 `label_id` 对齐。

执行：

```bash
./.conda/bin/python scripts/tal/dataloader_smoke.py \
  outputs/charades_clips_50.csv \
  --annotations data/charades/tal_annotations.json \
  --feature-manifest data/charades/features_smoke_manifest.json \
  --summary-output outputs/tal_dataloader_smoke_50_summary.json
```

结果：

```json
{
  "total": 50,
  "status_counts": {
    "ok": 50
  },
  "feature_dims_c": [2304],
  "min_feature_length_t": 38,
  "max_feature_length_t": 207,
  "total_segments": 148
}
```

这说明本机 smoke 特征、Charades 标注、clip offset 已经能组成 ActionFormer 需要的数据结构。下一步是把这套逻辑沉淀为正式 `charades.py` dataset adapter，并在 GPU 机器上替换为真实 SlowFast 特征。

### Phase 3：集成 + CLI（3–5 天）

**目标**：把 ActionFormer 接入主 CLI，可被 batch driver 调用。

- `src/video_analyst/tal.py`（~200 行）：模型加载、推理、后处理
- `src/video_analyst/cli.py` 新增子命令 `stage3-tal`
- `scripts/run_charades_clip_batch.py` 加 `--use-tal` 开关
- 单元测试 `tests/test_tal_postprocess.py`（mock 推理结果）

**交付物**：
- `video-analyst stage3-tal video.mp4 -o outputs/tal` 跑通
- `run_charades_clip_batch.py --use-tal` 在 20 视频上跑通

### Phase 4：评估 + 对比（3–5 天）

**目标**：在 20 视频 batch 上做规则 vs ActionFormer 的 A/B 对比。

- 跑两次 batch：一次用规则，一次用 ActionFormer
- `scripts/aggregate_charades_fusion.py` 加 `--source` 区分
- 输出三张表：
  1. 动作级别：ActionFormer vs 规则 在 20 个动作上的 precision/recall
  2. 视频级别：每个视频的事件数、融合状态分布
  3. 速度对比：单 clip 推理耗时

**交付物**：
- `outputs/charades_clip_batch_20_tal/aggregation_report.md`
- `outputs/charades_clip_batch_20_tal/comparison_table.md`（规则 vs TAL）

## 成功标准

| 指标 | 当前基线（规则 + VLM） | Phase 4 目标（ActionFormer + VLM） |
| --- | ---: | ---: |
| Stage2 命中率 | 9/20 = 45% | ≥ 13/20 = 65% |
| 融合后 Final 事件数 | 5 | ≥ 8 |
| 融合后 Rejected 事件数 | 21 | ≤ 10（VLM 抑制需求降低） |
| 单 clip 推理耗时 | ~30s (Stage2) + ~10s (VLM) | ~5s (TAL) + ~10s (VLM) |
| Charades 类别覆盖 | 2/20 动作 | ≥ 10/20 动作 |
| mAP@0.5（动作级别） | 不可计算（仅 2 类） | ≥ 30%（粗略下界） |

## 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
| --- | ---: | --- | --- |
| 50 视频太少，模型过拟合 | 高 | 中 | 数据增强（temporal jitter）+ 强预训练权重 + dropout |
| Charades 类别不平衡（吃饭 754 / 跑 0） | 中 | 中 | Focal Loss + 类别加权 |
| ActionFormer 官方代码与新 PyTorch 不兼容 | 中 | 高 | 用 `conda env` 固定 Python 3.8 + PyTorch 1.10 |
| 单 GPU 训练时间超 8 小时 | 低 | 中 | 用 35 视频 train 集（不加 val 做消融） |
| 集成到现有 CLI 时格式冲突 | 低 | 低 | 复用 Stage2 `events.json` 格式，零冲突 |
| ActionFormer 在 Charades 上 mAP 不如 THUMOS-14 | 中 | 中 | Charades 标注更稀疏（每视频 5–6 个动作），预期 30–40% mAP |

## 实施时间线

```text
Week 1   Phase 1：环境 + 推理 smoke test
Week 2-3 Phase 2：数据 + 训练
Week 3-4 Phase 3：集成 + CLI
Week 4-5 Phase 4：评估 + 对比
Week 5   写项目总结 + 进度文件 §32
```

**总投入**：~1 个月（每周 10–15 小时）。

## 与现有资产的关系

| 资产 | 复用 / 替代 |
| --- | --- |
| `data/charades/charades_pilot_subset_50.csv` | ✅ 复用（视频清单） |
| `data/charades/charades_train_pilot_actions.csv` | ✅ 复用（时间段标注） |
| `data/charades/videos_480p/` | ✅ 复用（视频源） |
| `data/charades/clips_480p/` | ⚠️ 部分复用（ActionFormer 倾向全视频而非 clip，clip 仅用于规则基线对比） |
| `data/charades/features_smoke/` | ✅ 复用（本机 smoke 格式验证，不训练） |
| `src/video_analyst/stage2.py` | ⚠️ 保留（作为基线对比项） |
| `scripts/review_video_with_vlm.py` | ✅ 复用（VLM 阶段零改动） |
| `scripts/fuse_stage2_vlm_events.py` | ✅ 复用（输入格式兼容） |
| `scripts/aggregate_charades_fusion.py` | ⚠️ 小改（加 `--source` 参数） |
| `scripts/run_charades_clip_batch.py` | ⚠️ 小改（加 `--use-tal` 开关） |
| `tests/` | ⚠️ 新增 `test_tal_postprocess.py`（~5 测试） |

## 短期决定

由于本方案需要 1 个月投入，建议**先完成以下短期工作再启动**：

1. 跑方案 A 加 5–7 条规则，把规则基线在 50 视频上的 mAP 做到 ~50%
2. 写一份"规则基线 vs ActionFormer"的明确对比假设（如：ActionFormer 在 hold/look phone 上显著优于规则）
3. 申请 1 块 GPU 资源（云 GPU 或本地）

启动条件：上述 3 项 + 周末时间 20+ 小时。

## 引用

```bibtex
@inproceedings{zhang2022actionformer,
  title={ActionFormer: Localizing Moments of Actions with Transformers},
  author={Zhang, Chen-Lin and Wu, Jianxin and Li, Yin},
  booktitle={ECCV},
  year={2022}
}

@inproceedings{lin2019bmn,
  title={BMN: Boundary-Matching Network for Temporal Action Proposal Generation},
  author={Lin, Tianwei and Liu, Xiao and Li, Xin and Ding, Errui and Wen, Shilei},
  booktitle={ICCV},
  year={2019}
}

@inproceedings{feichtenhofer2019slowfast,
  title={SlowFast Networks for Video Recognition},
  author={Feichtenhofer, Christoph and Fan, Haoqi and Malik, Jitendra and Girshick, Ross},
  booktitle={ICCV},
  year={2019}
}
```

## 相关文档

- [视频人员行为分析课题方案.md](../视频人员行为分析课题方案.md) §4.7
- [Charades.md](Charades.md) — Charades 数据集接入
- [VLM_REVIEW.md](VLM_REVIEW.md) — VLM 复核模块
- [项目进度.md §31](../项目进度.md) — 20 视频批量评估结果（基线）
- `papers/03_ActionFormer_读书笔记.md` — ActionFormer 论文笔记
- `papers/05_SlowFast_读书笔记.md` — SlowFast 论文笔记

## 2026-06-09 本机官方 adapter smoke 更新

当前 Phase 2 已先完成官方 ActionFormer 数据适配 smoke，不等待 GPU：

- `external/actionformer_release/libs/datasets/charades.py`
- `external/actionformer_release/configs/charades_smoke.yaml`
- `scripts/tal/actionformer_charades_smoke.py`

已验证：

```text
outputs/charades_clips_50.csv
+ data/charades/tal_annotations.json
+ data/charades/features_smoke_manifest.json
+ data/charades/features_smoke/*.npz
-> make_dataset("charades", ...)
-> LocPointTransformer forward
```

结果：

- dataset length: 50
- sample feats: `[2304, 91]`
- sample segments: `[5, 2]`
- model params: 2,295,328
- train-loss smoke: `final_loss = 0.30324`
- eval smoke: postprocess 输出 103 个随机候选 segment

注意：

- `features_smoke` 只是 `cv2-smoke` 格式特征，不用于训练。
- 真实训练仍需要 Linux GPU 上的 SlowFast/R50 特征。
- 当前 adapter 和 smoke config 已证明 ActionFormer 官方 dataset registry、segment grid、model forward 三者可以连通。

## 2026-06-09 SlowFast 后端与 split 更新

已继续完成真实训练前置：

- `scripts/tal/extract_slowfast_features.py` 支持 `--backend slowfast-r50`。
- `scripts/tal/build_charades_splits.py` 可从 `outputs/charades_clips_50.csv` 生成 50 视频 split。
- `external/actionformer_release/configs/charades_slowfast.yaml` 已准备。
- `external/actionformer_release/libs/datasets/charades.py` 支持 `split_folder` 过滤。

50 视频 split：

```json
{
  "train": 32,
  "val": 7,
  "test": 11,
  "must_test": ["024PD", "8BG1T", "OINMN"]
}
```

真实 SlowFast 后端约定：

```bash
python scripts/tal/extract_slowfast_features.py \
  outputs/charades_clips_50.csv \
  --backend slowfast-r50 \
  --output-dir data/charades/features \
  --summary-output outputs/tal_features_slowfast_50_summary.json \
  --feature-manifest-output data/charades/features_manifest.json
```

输出仍为：

```text
<clip_id>.npz
└── feats: float32, shape = T x 2304
```

`slowfast-r50` 后端会在 PyTorchVideo SlowFast R50 最终分类 projection 前捕获 pooled 2304 维特征。该后端需要 Linux GPU 机器安装 `pytorchvideo`；本机只回归了 `cv2-smoke` 后端和 shape/helper 测试。

新增 [TAL_4090_EXECUTION_PACKAGE.md](TAL_4090_EXECUTION_PACKAGE.md)，把 4090 上的执行顺序固定为：

1. `check_slowfast_backend.py --device cuda`
2. `extract_slowfast_features.py --backend slowfast-r50 --limit 1`
3. 50 clip 完整特征提取
4. 真实特征 dataloader smoke
5. `charades_slowfast.yaml` train-loss smoke
6. 1 epoch 训练 smoke
