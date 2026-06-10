# 多模态大模型复核方案

## 目标

在现有 Stage2 的 YOLO 物体检测、人员检测、姿态估计和规则事件基础上，增加一个 Stage2.5 复核层：

```text
视频 -> Stage2 候选事件/时间段 -> 抽关键帧 -> 多模态大模型复核 -> vlm_review.json -> 最终事件融合
```

这个模块不替代 YOLO/Pose，而是用于：

- 复核低置信度事件，例如 `8BG1T` 中疑似误报的打电话。
- 补检 YOLO 漏检的语义动作，例如 `OINMN` / `024PD` 的手机相关动作。
- 为 Charades 这种时间段级标注生成语义判断结果。

## 硅基流动接口

模型接口采用 OpenAI 兼容的 Chat Completions 格式：

```text
https://api.siliconflow.cn/v1/chat/completions
```

模型列表接口：

```text
https://api.siliconflow.cn/v1/models
```

本项目不会把 API key 写入源码或可提交文档。推荐使用本地 `.env` 文件，且 `.env` 已加入 `.gitignore`：

```bash
SILICONFLOW_API_KEY="你的 key"
```

也可以在本机 shell 设置环境变量：

```bash
export SILICONFLOW_API_KEY="你的 key"
```

然后检查账号可用视觉模型：

```bash
./.conda/bin/python scripts/list_siliconflow_models.py \
  --vision-only \
  --output outputs/siliconflow_vision_models.json
```

## 候选模型

实际可用模型以 `scripts/list_siliconflow_models.py` 查询结果为准。当前优先考虑：

| 模型 | 用途建议 |
| --- | --- |
| `Qwen/Qwen3-VL-8B-Instruct` | 默认首选，成本和速度较适合关键帧复核 |
| `Qwen/Qwen3-VL-32B-Instruct` | 更强复核，适合小批量或疑难样本 |
| `Qwen/Qwen3-VL-30B-A3B-Instruct` | 折中选择，可作为备选 |
| `GLM-5V-Turbo` | 视觉语义复核备选 |
| `Qwen/Qwen2.5-VL-72B-Instruct` | 若账号可用，可用于高质量复核或最终实验对比 |

当前账号通过 `.cn` 域名实际查询到的视觉模型：

- `PaddlePaddle/PaddleOCR-VL-1.5`
- `Qwen/Qwen3-VL-30B-A3B-Instruct`
- `Qwen/Qwen3-VL-30B-A3B-Thinking`
- `Qwen/Qwen3-VL-32B-Instruct`
- `Qwen/Qwen3-VL-32B-Thinking`
- `Qwen/Qwen3-VL-8B-Instruct`
- `Qwen/Qwen3-VL-8B-Thinking`
- `Qwen/Qwen3-VL-Embedding-8B`
- `Qwen/Qwen3-VL-Reranker-8B`

## 已新增脚本

```text
scripts/review_video_with_vlm.py
scripts/list_siliconflow_models.py
scripts/compare_charades_stage2_vlm.py
tests/test_review_video_with_vlm.py
tests/test_list_siliconflow_models.py
tests/test_compare_charades_stage2_vlm.py
```

`review_video_with_vlm.py` 功能：

- 从视频或指定时间段抽取关键帧。
- 生成硅基流动 Chat Completions 请求 payload。
- 支持 `--dry-run`，只生成关键帧和请求，不调用 API。
- 若环境变量 `SILICONFLOW_API_KEY` 存在且没有 `--dry-run`，则调用接口。
- 输出 `vlm_request_payload.json`、`vlm_response_raw.json`、`vlm_review.json`、`vlm_summary.json`。

`compare_charades_stage2_vlm.py` 功能：

- 输入 Charades manifest、Stage2 输出目录和 VLM review。
- 对齐标准动作：`drinking_water`、`talking_on_phone`、`holding_phone`、`looking_at_phone`、`holding_drink_container`。
- 输出每个动作的 Charades / Stage2 / VLM 三方状态。
- 给出初版融合状态和建议。

## Dry-run 记录

已对三个 Charades 视频生成关键帧和请求 payload：

```bash
./.conda/bin/python scripts/review_video_with_vlm.py \
  data/charades/videos_480p/8BG1T.mp4 \
  -o outputs/charades_8BG1T_vlm_review \
  --frame-count 6 \
  --actions drinking_water talking_on_phone holding_phone looking_at_phone \
  --dry-run
```

```bash
./.conda/bin/python scripts/review_video_with_vlm.py \
  data/charades/videos_480p/OINMN.mp4 \
  -o outputs/charades_OINMN_vlm_review \
  --frame-count 6 \
  --actions talking_on_phone holding_phone looking_at_phone \
  --dry-run
```

```bash
./.conda/bin/python scripts/review_video_with_vlm.py \
  data/charades/videos_480p/024PD.mp4 \
  -o outputs/charades_024PD_vlm_review \
  --frame-count 6 \
  --actions talking_on_phone holding_phone looking_at_phone \
  --dry-run
```

输出：

- `outputs/charades_8BG1T_vlm_review/frames/`
- `outputs/charades_8BG1T_vlm_review/vlm_request_payload.json`
- `outputs/charades_OINMN_vlm_review/frames/`
- `outputs/charades_OINMN_vlm_review/vlm_request_payload.json`
- `outputs/charades_024PD_vlm_review/frames/`
- `outputs/charades_024PD_vlm_review/vlm_request_payload.json`

## 真实调用命令

设置 `.env` 或环境变量后，去掉 `--dry-run` 即可真实调用：

```bash
export SILICONFLOW_API_KEY="你的 key"

./.conda/bin/python scripts/review_video_with_vlm.py \
  data/charades/videos_480p/OINMN.mp4 \
  -o outputs/charades_OINMN_vlm_review_live \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --frame-count 6 \
  --actions talking_on_phone holding_phone looking_at_phone
```

## 当前 API 测试结果

已将用户提供的新 key 写入本地 `.env`，文件权限为 `600`，并确认脚本能够从 `.env` 读取 key。

使用 `.com` 域名测试时返回 `401 Unauthorized`。切换为 `.cn` 域名后，模型列表查询成功。

模型列表测试命令：

```bash
./.conda/bin/python scripts/list_siliconflow_models.py \
  --vision-only \
  --output outputs/siliconflow_vision_models.json
```

`.com` 域名返回结果：

```json
{
  "status": "failed",
  "error": "http_error",
  "status_code": 401,
  "reason": "Unauthorized",
  "body": "\"Api key is invalid\""
}
```

结论：

- 当前 key 对 `.cn` 域名有效。
- 后续硅基流动 API 默认使用 `https://api.siliconflow.cn`。

## OINMN 真实 VLM 复核

已使用 `Qwen/Qwen3-VL-8B-Instruct` 对 `OINMN` 做真实复核：

```bash
./.conda/bin/python scripts/review_video_with_vlm.py \
  data/charades/videos_480p/OINMN.mp4 \
  -o outputs/charades_OINMN_vlm_review_live \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --frame-count 6 \
  --actions talking_on_phone holding_phone looking_at_phone
```

输出文件：

- `outputs/charades_OINMN_vlm_review_live/vlm_request_payload.json`
- `outputs/charades_OINMN_vlm_review_live/vlm_response_raw.json`
- `outputs/charades_OINMN_vlm_review_live/vlm_review.json`
- `outputs/charades_OINMN_vlm_review_live/vlm_summary.json`

VLM 结果：

| 动作 | 是否存在 | 置信度 | 支持帧 |
| --- | --- | ---: | --- |
| `talking_on_phone` | 是 | 0.90 | 5 |
| `holding_phone` | 是 | 0.95 | 2, 3, 5 |
| `looking_at_phone` | 是 | 0.90 | 2, 3, 5 |

结论：

- Stage2 在 `OINMN` 中没有检出 `cell phone`，因此事件为 0。
- VLM 能从关键帧中判断手机相关动作存在。
- 这验证了 Stage2.5 对手机类漏检样本有补检价值。

## 三方对比结果

已生成：

- `outputs/charades_8BG1T_comparison.json`
- `outputs/charades_OINMN_comparison.json`
- `outputs/charades_024PD_comparison.json`

### 8BG1T

汇总：

- Stage2 命中：2
- VLM 命中：2
- Charades 标签：1

关键结论：

- `drinking_water`：Charades 有标注，Stage2 命中，但 VLM 均匀抽帧未确认，状态为 `needs_temporal_review`。
- `talking_on_phone`：Stage2 命中但 VLM 否定，状态为 `possible_false_positive`。
- `holding_phone` / `looking_at_phone`：VLM 高置信命中，但无 Charades/Stage2 支持，状态为 `vlm_candidate_event`。

说明：`8BG1T` 的 VLM 均匀抽帧没有抽到 Stage2 检出的喝水瞬间，因此后续应围绕 Stage2 事件时间加密抽帧。

### OINMN

汇总：

- Stage2 命中：0
- VLM 命中：3
- Charades 标签：2

关键结论：

- `holding_phone`：Charades 有标注，Stage2 漏检，VLM 命中，状态为 `vlm_recovered_label`。
- `talking_on_phone`：Charades 有标注，Stage2 漏检，VLM 命中，状态为 `vlm_recovered_label`。
- `looking_at_phone`：VLM 命中，状态为 `vlm_candidate_event`。

### 024PD

汇总：

- Stage2 命中：0
- VLM 命中：2
- Charades 标签：1

关键结论：

- `holding_phone`：Charades 有标注，Stage2 漏检，VLM 命中，状态为 `vlm_recovered_label`。
- `looking_at_phone`：VLM 命中，状态为 `vlm_candidate_event`。

## 初版融合状态

| 状态 | 含义 |
| --- | --- |
| `confirmed_event` | Stage2 和 VLM 均命中，保留事件 |
| `possible_false_positive` | Stage2 命中但 VLM 否定，建议降置信或人工复核 |
| `vlm_recovered_label` | Stage2 漏检但 VLM 与 Charades 一致，可生成语义候选事件 |
| `vlm_candidate_event` | VLM 高置信命中但无 Charades/Stage2 支持，可作为候选事件 |
| `needs_temporal_review` | Charades/Stage2 有支持但 VLM 均匀抽帧未确认，需要围绕事件加密抽帧 |
| `missed_label` | Charades 有标注，但 Stage2 和 VLM 均未确认 |

## 事件中心抽帧复核

已增强 `scripts/review_video_with_vlm.py`：

- 支持 `--events-json` 读取 Stage2 `events.json`。
- 支持 `--event-index` 选择事件。
- 支持 `--event-context-seconds` 在事件前后加入上下文。
- 若未显式传入 `--actions`，会自动使用事件中的 `action`。
- prompt 中加入动作定义，例如 `drinking_water` 表示从杯、瓶、玻璃杯、易拉罐等容器饮用任意饮品，不要求字面上一定是水。

### 8BG1T 喝饮品事件

Stage2 事件：

- 行为：`drinking_water`
- 时间：11.833 秒到 12.667 秒
- Stage2 置信度：0.7083

事件中心 VLM 命令：

```bash
./.conda/bin/python scripts/review_video_with_vlm.py \
  data/charades/videos_480p/8BG1T.mp4 \
  -o outputs/charades_8BG1T_event0_drinking_vlm_review_live_v2 \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --frame-count 6 \
  --events-json outputs/charades_8BG1T_stage2/events.json \
  --event-index 0 \
  --event-context-seconds 1.0
```

VLM 结果：

- `drinking_water`：存在
- 置信度：0.70
- 支持帧：0、1、2、3、4、5
- 可见物体：`can`

结论：

- 事件中心抽帧将原来的 `needs_temporal_review` 变成可确认事件。
- 这说明短促动作不能只用整段均匀抽帧复核。

### 8BG1T 打电话事件

Stage2 事件：

- 行为：`talking_on_phone`
- 时间：12.367 秒到 13.167 秒
- Stage2 置信度：0.6436

事件中心 VLM 命令：

```bash
./.conda/bin/python scripts/review_video_with_vlm.py \
  data/charades/videos_480p/8BG1T.mp4 \
  -o outputs/charades_8BG1T_event1_phone_vlm_review_live \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --frame-count 6 \
  --events-json outputs/charades_8BG1T_stage2/events.json \
  --event-index 1 \
  --event-context-seconds 1.0
```

VLM 结果：

- `talking_on_phone`：不存在
- 置信度：0.0
- 证据：人物拿着的是饮品容器，不是手机。

结论：

- 该 Stage2 事件保持 `possible_false_positive` 判断。
- VLM 能帮助抑制 COCO `cell phone` 误检造成的规则误报。

## 融合输出层

已新增融合脚本：

- `scripts/fuse_stage2_vlm_events.py`
- `tests/test_fuse_stage2_vlm_events.py`

脚本输入：

- `compare_charades_stage2_vlm.py` 生成的 comparison JSON。
- 可选的事件中心 VLM review 目录、`vlm_summary.json` 或 `vlm_review.json`。

脚本输出：

- `final_events`：Stage2 事件被 VLM 确认后的最终事件。
- `semantic_candidates`：Stage2 漏检，但 Charades/VLM 或高置信 VLM 支持的候选语义事件。
- `pending_events`：Stage2 命中但还缺少事件中心复核的事件。
- `rejected_events`：Stage2 命中但被 VLM 否定的事件。
- `fused_events`：以上四类事件的统一列表。

### 运行命令

`8BG1T` 带入两个事件中心 VLM 复核：

```bash
./.conda/bin/python scripts/fuse_stage2_vlm_events.py \
  --comparison outputs/charades_8BG1T_comparison.json \
  --event-review outputs/charades_8BG1T_event0_drinking_vlm_review_live_v2 \
  --event-review outputs/charades_8BG1T_event1_phone_vlm_review_live \
  --output outputs/charades_8BG1T_fused_events.json
```

`OINMN` 和 `024PD` 使用三方 comparison 直接融合：

```bash
./.conda/bin/python scripts/fuse_stage2_vlm_events.py \
  --comparison outputs/charades_OINMN_comparison.json \
  --output outputs/charades_OINMN_fused_events.json

./.conda/bin/python scripts/fuse_stage2_vlm_events.py \
  --comparison outputs/charades_024PD_comparison.json \
  --output outputs/charades_024PD_fused_events.json
```

### 融合结果

| 视频 | 最终事件 | 语义候选 | 待复核 | 否定事件 |
| --- | ---: | ---: | ---: | ---: |
| `8BG1T` | 1 | 2 | 0 | 1 |
| `OINMN` | 0 | 3 | 0 | 0 |
| `024PD` | 0 | 2 | 0 | 0 |

关键结论：

- `8BG1T` 的 `drinking_water` 已从 `needs_temporal_review` 升级为 `confirmed_event`。
- `8BG1T` 的 `talking_on_phone` 被事件中心 VLM 复核否定，进入 `rejected_events`。
- `OINMN` 的 `holding_phone` 和 `talking_on_phone` 由 Charades + VLM 补回为语义候选。
- `024PD` 的 `holding_phone` 由 Charades + VLM 补回为语义候选。

当前验证：

```text
Ran 48 tests in 0.007s
OK
```

## 下一步

1. 把 Charades 的标注时间段切成短 clip，再跑 Stage2 + VLM 复核，减少完整视频处理成本。
2. 给 `semantic_candidates` 增加时间定位：从全视频均匀抽帧升级为围绕 Charades 标注窗口抽帧。
3. 扩大 Charades 小批量，统计最终事件、候选事件、否定事件的类别分布。

## 修复记录

更新时间：2026-06-09

已根据项目审查结果修复融合链路中的关键问题：

- `compare_charades_stage2_vlm.py` 支持 `--stage2-time-offset-seconds`，当 Stage2 跑在切片 clip 上时，可把事件时间平移回原视频时间。
- comparison 输出中的 Stage2 事件会保留 `stage2_event_index`，用于稳定映射事件中心 VLM 的 `event:<index>`。
- `fuse_stage2_vlm_events.py` 改为优先使用事件自带的 `stage2_event_index`，避免因为动作排序导致 VLM 复核错配。
- `fuse_stage2_vlm_events.py` 会在输出中保留 `clip_start_seconds`、`clip_end_seconds`、`time_offset_seconds` 和 `time_coordinate`。
- `aggregate_charades_fusion.py` 已改为按每条事件自己的 `action` 统计，避免一份多动作 fused 文件全部归到第一个动作。
- `run_charades_clip_batch.py` 已修复自定义 `--api-key-env` 的传递，并把 VLM/compare/fuse 错误写入最终记录。

已重新生成三条样本输出：

- `outputs/charades_8BG1T_comparison.json`
- `outputs/charades_OINMN_comparison.json`
- `outputs/charades_024PD_comparison.json`
- `outputs/charades_8BG1T_fused_events.json`
- `outputs/charades_OINMN_fused_events.json`
- `outputs/charades_024PD_fused_events.json`

当前验证：

```text
Ran 66 tests in 0.023s
OK
```

## Batch Pending 事件中心复核

更新时间：2026-06-09

已新增并运行 pending 事件中心复核脚本：

- `scripts/review_pending_events.py`
- `tests/test_review_pending_events.py`

脚本功能：

- 扫描 batch root 下所有 `fused_events.json`。
- 找到 `pending_events`。
- 对每个 pending 事件调用 `review_video_with_vlm.py --events-json ... --event-index ...`。
- 将事件中心 VLM 输出写入 `<clip_dir>/event_reviews/event_<index>_<action>/`。
- 重新调用 `fuse_stage2_vlm_events.py` 融合该 clip。

已对 `outputs/charades_clip_batch_20` 执行：

```bash
./.conda/bin/python scripts/review_pending_events.py \
  --batch-root outputs/charades_clip_batch_20 \
  --frame-count 6 \
  --event-context-seconds 1.0
```

结果：

```json
{
  "pending_events_found": 1,
  "completed": 1,
  "review_failed": 0,
  "fused_ok": 1,
  "vlm_present": 1
}
```

关键变化：

- `8BG1T_c106_t5.60_45.46` 中的 `drinking_water` pending 被事件中心 VLM 确认为存在。
- batch20 聚合结果中 `final_events` 从 5 增加到 6。
- `pending_events` 从 1 降为 0。
- 输出中保留原视频时间和 clip 时间：
  - 原视频时间：17.433 秒到 18.267 秒。
  - clip 内时间：11.833 秒到 12.667 秒。

重新聚合：

```bash
./.conda/bin/python scripts/aggregate_charades_fusion.py \
  --batch-root outputs/charades_clip_batch_20 \
  --clips-manifest outputs/charades_clips_20.csv
```

聚合结果：

| 指标 | 数值 |
| --- | ---: |
| total_clips_processed | 20 |
| total_fused_events | 31 |
| final_events | 6 |
| semantic_candidates | 4 |
| pending_events | 0 |
| rejected_events | 21 |

当前验证：

```text
Ran 68 tests in 0.031s
OK
```

## Canonical Action 映射扩展

更新时间：2026-06-09

已扩展 Charades 到项目内部 action 的映射，覆盖 batch20 / pilot subset 中的 20 个动作：

- 手机类：`holding_phone`、`looking_at_phone`、`talking_on_phone`
- 杯/瓶类：`drinking_water`、`holding_drink_container`、`taking_drink_container`、`putting_drink_container`
- 姿态/移动类：`sitting_down`、`standing_up`、`sitting_on_chair`、`sitting_on_sofa`、`sitting_on_floor`、`walking_through_doorway`、`running`
- 室内活动类：`using_laptop`、`watching_laptop`、`watching_tv`、`cooking`、`eating`

已更新：

- `scripts/compare_charades_stage2_vlm.py`
- `scripts/review_video_with_vlm.py`
- `tests/test_compare_charades_stage2_vlm.py`
- `docs/Charades.md`

已刷新 batch20 输出：

```bash
./.conda/bin/python scripts/run_charades_clip_batch.py \
  outputs/charades_clips_20.csv \
  --charades-manifest data/charades/charades_train_pilot_actions.csv \
  --output-root outputs/charades_clip_batch_20 \
  --skip-stage2 \
  --frame-count 6

./.conda/bin/python scripts/aggregate_charades_fusion.py \
  --batch-root outputs/charades_clip_batch_20 \
  --clips-manifest outputs/charades_clips_20.csv
```

刷新后 `batch_summary.json` 中的 `by_canonical_action` 已不再使用大部分原始 `c051/c052/...` ID：

```json
{
  "eating": 2,
  "sitting_down": 1,
  "standing_up": 1,
  "sitting_on_chair": 1,
  "walking_through_doorway": 1,
  "running": 1,
  "using_laptop": 1,
  "watching_laptop": 1,
  "cooking": 1,
  "taking_drink_container": 1,
  "putting_drink_container": 1,
  "sitting_on_sofa": 1,
  "sitting_on_floor": 1,
  "watching_tv": 1
}
```

当前验证：

```text
Ran 70 tests in 0.025s
OK
```

## Batch20 聚合报告口径修复

更新时间：2026-06-09

`outputs/charades_clip_batch_20/aggregation_report.md` 原先的 per-action 表格只用
`outputs/charades_clips_20.csv` 中抽样时选中的主 action 作为 `GT (clips)`，但融合链路实际会读取同一视频内的全部 Charades 标注。

这会造成一个误导性现象：例如只抽到 1 个 `drinking_water` 主 action clip，但同一批视频里还有其他片段也带有 `drinking_water` 标注，同时 Stage2 会产生多个候选事件，于是旧报告可能出现类似 `recovery=7/1` 的显示。

现在已拆成两个口径：

- `Selected clips`：本次抽样 manifest 中，以该 action 作为主 action 选出的 clip 数。
- `Charades GT clips`：从每个 clip 的 `comparison.json` 中读取的 Charades 全量标注命中数。
- `supported_events`：融合后被保留的事件数，即 `final_events + semantic_candidates`。

已更新：

- `scripts/aggregate_charades_fusion.py`
- `tests/test_aggregate_charades_fusion.py`
- `outputs/charades_clip_batch_20/aggregation_report.json`
- `outputs/charades_clip_batch_20/aggregation_report.md`

重新聚合结果：

```json
{
  "total_clips_processed": 20,
  "total_fused_events": 31,
  "unique_videos": 20,
  "unique_actions": 19,
  "unique_actions_with_outputs": 4,
  "overall_bucket_counts": {
    "final_events": 6,
    "semantic_candidates": 4,
    "pending_events": 0,
    "rejected_events": 21
  }
}
```

报告中 `drinking_water` 现在显示为：

```text
Selected clips=1, Charades GT clips=2, Final=6, Semantic=1, Rejected=10,
Status=F=6 S=1 P=0 R=10 supported_events=7, gt_clips=2
```

当前验证：

```text
Ran 72 tests in 0.050s
OK
```
