# Charades 接入记录

## 目标

Charades 用于在 AVA 和 NTU 之外补充室内日常活动数据。它更适合做视频级/时间段级行为泛化验证，例如喝水、拿手机、打电话、吃东西、坐下、站起、走动、使用电脑等。

它和当前项目的关系：

- 优点：室内日常场景多，动作-物体组合丰富，标注包小且可直接下载。
- 优点：动作有开始和结束时间，适合训练或评估时间段行为定位。
- 限制：标注是视频时间段级，不包含人员框、person ID 或逐人轨迹。
- 限制：视频包较大，480p 版本约 13GB，原始视频约 55GB，不建议一开始整包下载。

## 官方信息

官方页面：https://prior.allenai.org/projects/charades

标注包下载地址：

```text
https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades.zip
```

数据集说明要点：

- 9848 个室内日常活动视频。
- 157 个动作类别。
- 约 66,500 个动作时间段标注。
- 15 类室内场景。
- 标注文件包含训练集和测试集 CSV。
- 数据许可为非商业研究用途；不能向第三方再分发数据。

## 本地文件

已下载并解压标注包：

```text
data/charades/Charades.zip
data/charades/annotations/Charades/
```

关键文件：

- `Charades_v1_train.csv`
- `Charades_v1_test.csv`
- `Charades_v1_classes.txt`
- `Charades_v1_objectclasses.txt`
- `Charades_v1_verbclasses.txt`
- `Charades_v1_mapping.txt`
- `license.txt`

## 标注格式

`Charades_v1_train.csv` 和 `Charades_v1_test.csv` 的关键字段：

- `id`：视频 ID。
- `subject`：拍摄主体 ID。
- `scene`：室内场景。
- `quality`：视频质量评分，7 为最高。
- `relevance`：视频与脚本相关性评分，7 为最高。
- `verified`：是否通过人工验证。
- `script`：拍摄脚本。
- `objects`：视频中涉及的对象，分号分隔。
- `descriptions`：人工描述。
- `actions`：动作时间段，格式为 `class start end`，多个动作以分号分隔。
- `length`：视频时长，单位秒。

示例：

```text
c015 0.00 32.00;c107 0.00 32.00
```

表示同一个视频中同时存在：

- `c015`：Holding a phone/camera
- `c107`：Holding a cup/glass/bottle of something

## 已新增脚本

```text
scripts/prepare_charades.py
scripts/build_charades_pilot_subset.py
scripts/download_charades_remote_zip.py
tests/test_prepare_charades.py
tests/test_build_charades_pilot_subset.py
tests/test_download_charades_remote_zip.py
```

`prepare_charades.py` 功能：

- 解析 Charades 训练集或测试集 CSV。
- 将视频级动作字段拆成一行一个动作时间段。
- 合并动作类别、对象类别、动词类别和 mapping。
- 支持按动作 ID 筛选。
- 支持 `--verified-only` 和 `--min-quality` 过滤。
- 输出 CSV、JSON 和 summary JSON。

`build_charades_pilot_subset.py` 功能：

- 从 pilot manifest 中均衡抽样视频。
- 按动作类别轮询抽样，避免单一动作过度集中。
- 限制同一场景最大样本数。
- 为每个动作时间段生成带前后文的 clip 起止时间。

`download_charades_remote_zip.py` 功能：

- 从官方 480p 远程 zip 中按视频 ID 抽取 mp4。
- 使用 HTTP Range，不需要下载完整 15.2GB 压缩包。
- 输出每个视频的下载状态和本地路径。

## 运行命令

生成训练集全量动作 manifest：

```bash
./.conda/bin/python scripts/prepare_charades.py \
  data/charades/annotations/Charades/Charades_v1_train.csv \
  --output data/charades/charades_train_all_actions.csv \
  --json-output data/charades/charades_train_all_actions.json \
  --summary-output data/charades/charades_train_summary.json
```

生成测试集全量动作 manifest：

```bash
./.conda/bin/python scripts/prepare_charades.py \
  data/charades/annotations/Charades/Charades_v1_test.csv \
  --output data/charades/charades_test_all_actions.csv \
  --summary-output data/charades/charades_test_summary.json
```

生成当前项目 pilot 子集：

```bash
./.conda/bin/python scripts/prepare_charades.py \
  data/charades/annotations/Charades/Charades_v1_train.csv \
  --actions c015 c016 c017 c018 c019 c051 c052 c059 c065 c097 c106 c107 c108 c109 c110 c123 c125 c129 c132 c147 c150 c151 c154 c156 \
  --verified-only \
  --min-quality 5 \
  --output data/charades/charades_train_pilot_actions.csv \
  --json-output data/charades/charades_train_pilot_actions.json \
  --summary-output data/charades/charades_train_pilot_summary.json
```

## 当前生成结果

训练集全量：

- 动作时间段：49,809
- 视频数量：7,811
- 动作类别：157
- 场景数量：16
- 平均动作时长：12.9193 秒

测试集全量：

- 动作时间段：16,691

pilot 子集：

- 过滤条件：已验证，质量分数不低于 5。
- 动作时间段：12,086
- 视频数量：4,717
- 动作类别：24
- 场景数量：16
- 平均动作时长：13.8995 秒

pilot 子集高频动作：

| 动作 ID | 动作名称 | 数量 |
| --- | --- | ---: |
| c097 | Walking through a doorway | 1149 |
| c154 | Someone is standing up from somewhere | 1040 |
| c059 | Sitting in a chair | 982 |
| c106 | Drinking from a cup/glass/bottle | 923 |
| c015 | Holding a phone/camera | 855 |
| c107 | Holding a cup/glass/bottle of something | 813 |
| c156 | Someone is eating something | 754 |
| c151 | Someone is going from standing to sitting | 740 |
| c016 | Playing with a phone/camera | 548 |

## Canonical action 映射

已将当前 pilot / batch20 中使用的 Charades 动作 ID 映射为项目内部统一 action 名称：

| Charades ID | Charades 名称 | Canonical action | 中文名称 |
| --- | --- | --- | --- |
| `c015` | Holding a phone/camera | `holding_phone` | 拿手机 |
| `c016` | Playing with a phone/camera | `looking_at_phone` | 看手机 |
| `c019` | Talking on a phone/camera | `talking_on_phone` | 打电话 |
| `c051` | Watching a laptop or something on a laptop | `watching_laptop` | 看笔记本电脑 |
| `c052` | Working/Playing on a laptop | `using_laptop` | 使用笔记本电脑 |
| `c059` | Sitting in a chair | `sitting_on_chair` | 坐在椅子上 |
| `c065` | Eating a sandwich | `eating` | 吃东西 |
| `c097` | Walking through a doorway | `walking_through_doorway` | 穿过门口 |
| `c106` | Drinking from a cup/glass/bottle | `drinking_water` | 喝水 |
| `c107` | Holding a cup/glass/bottle of something | `holding_drink_container` | 拿杯/瓶 |
| `c109` | Putting a cup/glass/bottle somewhere | `putting_drink_container` | 放下杯/瓶 |
| `c110` | Taking a cup/glass/bottle from somewhere | `taking_drink_container` | 拿起杯/瓶 |
| `c123` | Sitting on sofa/couch | `sitting_on_sofa` | 坐在沙发上 |
| `c125` | Sitting on the floor | `sitting_on_floor` | 坐在地上 |
| `c132` | Watching television | `watching_tv` | 看电视 |
| `c147` | Someone is cooking something | `cooking` | 做饭 |
| `c150` | Someone is running somewhere | `running` | 跑步 |
| `c151` | Someone is going from standing to sitting | `sitting_down` | 坐下 |
| `c154` | Someone is standing up from somewhere | `standing_up` | 站起 |
| `c156` | Someone is eating something | `eating` | 吃东西 |

实现位置：

- `scripts/compare_charades_stage2_vlm.py`
- `scripts/review_video_with_vlm.py`

说明：

- `c065` 和 `c156` 都归并到 `eating`，便于后续统计和 VLM prompt 复用。
- 旧 VLM 输出如果仍然返回 `c156` 这类原始 ID，comparison 会自动 canonicalize。

## 50 个视频候选子集

已从 pilot manifest 中抽取 50 个候选视频：

```bash
./.conda/bin/python scripts/build_charades_pilot_subset.py \
  data/charades/charades_train_pilot_actions.csv \
  --max-videos 50 \
  --samples-per-action 4 \
  --max-per-scene 8 \
  --min-duration 2.0 \
  --context-seconds 2.0 \
  --output data/charades/charades_pilot_subset_50.csv \
  --json-output data/charades/charades_pilot_subset_50.json \
  --summary-output data/charades/charades_pilot_subset_50_summary.json
```

输出文件：

- `data/charades/charades_pilot_subset_50.csv`
- `data/charades/charades_pilot_subset_50.json`
- `data/charades/charades_pilot_subset_50_summary.json`

结果：

- 视频数量：50
- 动作类别：20
- 场景数量：16
- 平均 clip 时长：36.8912 秒

## 小批量视频下载

官方视频包：

- `Charades_v1.zip`：约 54.7GiB
- `Charades_v1_480.zip`：约 15.2GiB

已确认远程 480p zip 支持 HTTP Range，因此可以按视频 ID 抽取 mp4，不需要整包下载。

当前下载命令：

```bash
./.conda/bin/python scripts/download_charades_remote_zip.py \
  data/charades/charades_pilot_subset_50.csv \
  --limit 3 \
  --output-dir data/charades/videos_480p \
  --summary-output data/charades/charades_download_limit3_summary.json
```

已下载视频：

| 视频 ID | 本地路径 | 大小 |
| --- | --- | ---: |
| 8BG1T | `data/charades/videos_480p/8BG1T.mp4` | 1,960,542 bytes |
| OINMN | `data/charades/videos_480p/OINMN.mp4` | 1,807,269 bytes |
| 024PD | `data/charades/videos_480p/024PD.mp4` | 2,256,986 bytes |

## Stage2 试跑记录

### 8BG1T

测试视频：

```text
data/charades/videos_480p/8BG1T.mp4
```

Charades 标注：

- `c106` Drinking from a cup/glass/bottle
- 标注时间段：7.6 秒到 43.7 秒
- 视频描述：人物坐到沙发上，喝杯中饮品，并看电视大笑。

视频元信息：

```json
{
  "fps": 30.0,
  "frame_count": 1365,
  "width": 480,
  "height": 318,
  "duration_seconds": 45.5
}
```

运行命令：

```bash
./.conda/bin/video-analyst stage2 \
  data/charades/videos_480p/8BG1T.mp4 \
  -o outputs/charades_8BG1T_stage2 \
  --det-model yolo11n.pt \
  --pose-model yolo11n-pose.pt \
  --conf 0.25 \
  --imgsz 640
```

结果：

- 目标物体：31
- 姿态结果：1293
- 行为事件：2
- 标注视频：`outputs/charades_8BG1T_stage2/stage2_annotated.mp4`

识别事件：

| 行为 | 起止时间 | 置信度 | 备注 |
| --- | --- | ---: | --- |
| 喝水 | 11.833 秒到 12.667 秒 | 0.7083 | 命中 Charades `c106` 标注时间段 |
| 打电话 | 12.367 秒到 13.167 秒 | 0.6436 | 可能是 `cell phone` 误检导致 |

物体检测统计：

- `cup`：17
- `cell phone`：14

结论：

- 当前 Stage2 可以在 Charades 的 `c106` 视频中命中喝水事件。
- 事件时间段比 Charades 标注短很多，说明当前规则偏向“实际举杯靠近头部”的瞬时动作，而 Charades 标注覆盖更长的语义活动段。
- 手机误报说明 COCO 物体检测在低分辨率室内视频上会产生混淆，后续需要结合 Charades 标签和动作上下文抑制无关事件。
- 直接处理完整 45 秒视频耗时较长，后续应优先按 `clip_start_seconds` 和 `clip_end_seconds` 切片后再运行 Stage2。

### OINMN

测试视频：

```text
data/charades/videos_480p/OINMN.mp4
```

Charades 标注：

- `c015` Holding a phone/camera
- `c019` Talking on a phone/camera
- 标注时间段：0.0 秒到约 46.0 秒

视频元信息：

```json
{
  "fps": 29.97002997002997,
  "frame_count": 1357,
  "width": 480,
  "height": 270,
  "duration_seconds": 45.2786
}
```

输出目录：

```text
outputs/charades_OINMN_stage2
```

Stage2 结果：

- 目标物体：0
- 姿态结果：1359
- 行为事件：0
- 标注视频：`outputs/charades_OINMN_stage2/stage2_annotated.mp4`

结论：

- 姿态链路正常。
- 当前 YOLO COCO 检测没有检出 `cell phone`，所以 `talking_on_phone` 规则无法触发。
- 这说明 Charades 手机类动作不能只依赖 COCO `cell phone` 物体检测，后续需要更泛化的动作/语义复核或更强的开放词汇检测。

### 024PD

测试视频：

```text
data/charades/videos_480p/024PD.mp4
```

Charades 标注：

- `c015` Holding a phone/camera
- 标注时间段：0.0 秒到 60.0 秒
- 视频描述：人物翻杂志，同时打电话并拍照。

视频元信息：

```json
{
  "fps": 29.97002997002997,
  "frame_count": 1770,
  "width": 480,
  "height": 270,
  "duration_seconds": 59.059
}
```

输出目录：

```text
outputs/charades_024PD_stage2
```

Stage2 结果：

- 目标物体：209
- 姿态结果：1763
- 行为事件：0
- 标注视频：`outputs/charades_024PD_stage2/stage2_annotated.mp4`

物体检测统计：

- `bottle`：208
- `cup`：1

结论：

- 姿态链路正常。
- 当前检测器没有检出 `cell phone`，反而将画面中的物体大量识别为 `bottle`。
- 虽然存在 `bottle/cup` 检测，但没有满足“靠近头部和手腕”的喝水规则，因此没有误触发喝水事件。
- 该样本进一步说明：Charades 手机相关动作需要引入更强的物体识别或视频语义模型，单靠 COCO 物体类别不足。

## 和当前 Stage2 的关系

可直接对照的现有行为：

- `c106` Drinking from a cup/glass/bottle -> `drinking_water`
- `c015` Holding a phone/camera -> 手机相关候选
- `c019` Talking on a phone/camera -> `talking_on_phone`

可用于扩展泛化能力的行为：

- `c059` Sitting in a chair
- `c097` Walking through a doorway
- `c150` Someone is running somewhere
- `c151` Someone is going from standing to sitting
- `c154` Someone is standing up from somewhere
- `c156` Someone is eating something
- `c051` Watching a laptop or something on a laptop
- `c052` Working/Playing on a laptop

## 下一步建议

1. 新增 Charades clip 切片脚本，按 `clip_start_seconds` 和 `clip_end_seconds` 从已下载视频中切出短片段。
2. 对已下载的 3 个视频生成短 clip 并批量运行 Stage2。
3. 新增 Charades 时间段评估脚本，计算 Stage2 事件与 Charades 标注的 overlap、coverage、precision-like 指标。
4. 继续按需下载 20 到 50 个视频，而不是下载完整视频包。
