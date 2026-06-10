# ActionLens

ActionLens 是一个面向“视频中谁在什么时间做了什么”的研究原型项目。当前代码已实现第一阶段和第二阶段规则基线：视频读取、人员检测、ByteTrack 多目标跟踪、物体检测、姿态估计、简单行为事件输出和标注视频输出。

English summary: **Video-based human action analysis with rules, VLM review, and Temporal Action Localization experiments.**

## 环境

本项目使用本地 conda 环境：

```bash
conda activate /Users/tiu/project/video-analyst/.conda
```

如果需要重新安装依赖：

```bash
./.conda/bin/python -m pip install -e .
```

如果需要重建 conda 环境：

```bash
conda env create -p ./.conda -f environment.yml
```

## 第一阶段命令

查看视频元信息：

```bash
./.conda/bin/video-analyst probe path/to/video.mp4
```

运行人员检测和 ByteTrack 跟踪：

```bash
./.conda/bin/video-analyst stage1 path/to/video.mp4 -o outputs/stage1
```

常用参数：

```bash
./.conda/bin/video-analyst stage1 path/to/video.mp4 \
  -o outputs/stage1 \
  --model yolo11n.pt \
  --conf 0.25 \
  --iou 0.5 \
  --imgsz 640
```

## 输出文件

`stage1` 会在输出目录生成：

- `metadata.json`：视频 FPS、帧数、宽高、时长。
- `detections.json`：逐帧人员检测框、置信度和 track ID。
- `tracks.json`：按人员 ID 聚合后的轨迹。
- `summary.json`：本次分析摘要。
- `annotated.mp4`：带人员框和 ID 的标注视频。

## 第二阶段命令

运行物体检测、姿态估计和规则行为识别：

```bash
./.conda/bin/video-analyst stage2 path/to/video.mp4 -o outputs/stage2
```

常用参数：

```bash
./.conda/bin/video-analyst stage2 path/to/video.mp4 \
  -o outputs/stage2 \
  --det-model yolo11n.pt \
  --pose-model yolo11n-pose.pt \
  --conf 0.25 \
  --iou 0.5 \
  --imgsz 640 \
  --min-event-frames 3 \
  --min-event-seconds 0.3 \
  --max-gap-frames 6
```

`stage2` 当前支持两个规则行为：

- 喝水：`cup` 或 `bottle` 同时靠近人员头部和手腕。
- 打电话：`cell phone` 靠近人员头部。

`stage2` 会在输出目录生成：

- `person_boxes.json`：带 track ID 的人员框。
- `objects.json`：杯子、水瓶、手机等目标物体检测结果。
- `poses.json`：匹配到人员 ID 的姿态关键点。
- `frame_action_scores.json`：逐帧行为规则命中结果。
- `events.json`：合并后的行为事件。
- `events.csv`：行为事件表格。
- `summary.json`：本次分析摘要。
- `stage2_annotated.mp4`：第二阶段标注视频。

事件合并参数：

- `--min-event-frames`：行为至少命中多少帧才输出事件。
- `--min-event-seconds`：行为持续时间至少多少秒才输出事件。
- `--max-gap-frames`：同一行为中间允许缺失多少帧仍继续合并。

## 生成测试视频

安装 Ultralytics 后，可以用其示例图片生成一段短测试视频：

```bash
./.conda/bin/python scripts/make_sample_video.py -o data/sample_bus.mp4 --seconds 2 --fps 5
./.conda/bin/video-analyst stage1 data/sample_bus.mp4 -o outputs/sample_bus
./.conda/bin/video-analyst stage2 data/sample_bus.mp4 -o outputs/sample_bus_stage2
```

## 后续方向

当前第二阶段已经接入规则基线，后续可以继续扩展：

- 更稳定的行为样例视频和阈值标定。
- 使用 Grounding DINO 支持 `printer` 等开放类别物体。
- 输出关键帧截图，便于人工复核每个事件。
- 接入多模态大模型复核低置信度候选片段。

## AVA 数据准备

下载 AVA 标注后，可生成全量动作 manifest：

```bash
./.conda/bin/python scripts/prepare_ava.py \
  data/ava/annotations/ava_val_v2.2.csv \
  --output data/ava/ava_val_all_actions.csv
```

生成小型泛化评估子集：

```bash
./.conda/bin/python scripts/build_ava_eval_subset.py \
  data/ava/ava_val_all_actions.csv \
  --samples-per-action 10 \
  --max-per-video-action 2 \
  --output data/ava/ava_val_eval_subset.csv
```

下载一个 AVA 短片段：

```bash
./.conda/bin/python scripts/download_ava_clips.py \
  data/ava/ava_val_eval_subset.csv \
  --limit 1 \
  --output-dir data/ava/clips
```

对齐 AVA 标注和 `stage2` 输出：

```bash
./.conda/bin/python scripts/evaluate_ava_alignment.py \
  data/ava/ava_val_eval_subset.csv \
  data/ava/clips/5BDj0ow5hnA_t0997_p114_a15_answer_phone.mp4 \
  outputs/ava_clip_answer_phone_stage2
```

批量下载、运行 `stage2` 并汇总 AVA 对齐指标：

```bash
./.conda/bin/python scripts/run_ava_batch_eval.py \
  data/ava/ava_val_eval_subset.csv \
  --limit 3 \
  --clips-dir data/ava/clips \
  --output-root outputs/ava_batch_eval_limit3
```

从批量评估结果提取通用动作特征：

```bash
./.conda/bin/python scripts/extract_ava_action_features.py \
  --batch-summary outputs/ava_batch_eval_limit3/batch_summary.json \
  --output outputs/ava_batch_eval_limit3/ava_action_features.csv \
  --json-output outputs/ava_batch_eval_limit3/ava_action_features.json
```
