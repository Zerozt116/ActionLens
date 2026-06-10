# AVA Actions 接入计划

## 目标

在 NTU RGB+D 120 审批期间，先使用 AVA Actions 做人员级时空动作检测研究和样本筛选。当前目标从少数固定行为扩展为尽量泛化的人员行为分析。

AVA 比 NTU 更贴近本项目的核心任务：

- AVA 提供人员框。
- AVA 提供同一人的 `person_id`，可跨相邻帧关联。
- AVA 在 1 秒间隔上提供动作标签。
- 同一人可以有多个动作标签。

## 官方信息

官方页面：https://research.google.com/ava/index.html

下载页面：https://research.google.com/ava/download.html

官方说明要点：

- AVA v2.2 包含 430 个视频。
- 训练集 235 个视频，验证集 64 个视频，测试集 131 个视频。
- 每个视频有 15 分钟片段被标注。
- 标注间隔为 1 秒。
- AVA v2.2 共有 1.62M 动作标签。
- 数据以 CSV 形式提供。
- 数据采用 CC BY 4.0 许可。

## 标注格式

AVA Actions CSV 每一行格式：

```text
video_id,middle_frame_timestamp,x1,y1,x2,y2,action_id,person_id
```

字段含义：

- `video_id`：YouTube 视频 ID。
- `middle_frame_timestamp`：动作对应帧的时间戳，单位为秒。
- `x1,y1,x2,y2`：人员框，归一化到 0 到 1。
- `action_id`：动作类别 ID。
- `person_id`：同一视频中相邻帧可关联的人员 ID。

## 动作标签

AVA v2.2 动作分为三类：

- `PERSON_MOVEMENT`
- `OBJECT_MANIPULATION`
- `PERSON_INTERACTION`

当前系统先支持两种 manifest：

- 全量动作 manifest：用于泛化分析和动作分布统计。
- 目标动作 manifest：用于快速验证喝水、接电话、看手机等具体行为。

与现有第二阶段规则直接相关的动作：

| AVA action_id | 标签 | 本项目参考动作 |
| --- | --- | --- |
| 15 | answer phone | `answer_phone` |
| 27 | drink | `drink` |
| 57 | text on/look at a cellphone | `text_on_look_at_a_cellphone` |
| 62 | work on a computer | `work_on_a_computer` |
| 79 | talk to (e.g., self, a person, a group) | `talk_to` |

## 数据准备流程

下载 AVA v2.2 标注压缩包或单独 CSV：

- `ava_v2.2.zip`
- `ava_train_v2.2.csv`
- `ava_val_v2.2.csv`
- `ava_action_list_v2.2.pbtxt`

生成全量动作 manifest：

```bash
./.conda/bin/python scripts/prepare_ava.py \
  /path/to/ava_train_v2.2.csv \
  --output data/ava_train_all_actions.csv
```

生成目标动作 manifest：

```bash
./.conda/bin/python scripts/prepare_ava.py \
  /path/to/ava_train_v2.2.csv \
  --output data/ava_train_targets.csv \
  --json-output data/ava_train_targets.json
```

筛选喝水和手机相关动作：

```bash
./.conda/bin/python scripts/prepare_ava.py \
  /path/to/ava_train_v2.2.csv \
  --actions 15 27 57 \
  --output data/ava_train_phone_drink.csv
```

## 后续工程路线

第一步：只解析官方 CSV，生成全量动作 manifest 和目标动作 manifest。

第二步：统计动作分布，从动作类型、动作频次和视频覆盖率上选择泛化评估子集。

第三步：下载对应视频片段。AVA 的 `video_id` 是 YouTube ID，需要根据官方数据说明和可用工具获取原始视频或片段。由于公开视频可能失效，后续需要记录下载失败比例。

第四步：对下载到的视频片段运行当前 `stage2`：

```bash
./.conda/bin/video-analyst stage2 /path/to/ava_clip.mp4 -o outputs/ava_clip_stage2
```

第五步：对比：

- AVA 标注框和本项目 YOLO 人员框。
- AVA `person_id` 和本项目 ByteTrack ID。
- AVA `drink` / `answer phone` 标签和本项目 `events.json`。

## 已生成文件

全量动作 manifest：

- `data/ava/ava_train_all_actions.csv`
- `data/ava/ava_val_all_actions.csv`

小型泛化评估子集：

- `data/ava/ava_val_eval_subset.csv`
- `data/ava/ava_val_eval_subset.json`
- `data/ava/ava_val_eval_subset_summary.json`

当前验证集子集规模：

- 160 条标注
- 16 个动作
- 15 个视频
- 覆盖 `PERSON_MOVEMENT`、`OBJECT_MANIPULATION`、`PERSON_INTERACTION`

批量评估脚本：

```bash
./.conda/bin/python scripts/run_ava_batch_eval.py \
  data/ava/ava_val_eval_subset.csv \
  --limit 3 \
  --clips-dir data/ava/clips \
  --output-root outputs/ava_batch_eval_limit3
```

当前小批量结果：

- 请求样本：3
- 下载成功：2
- 下载失败：1
- Stage2 成功：2
- AVA 人框匹配率：1.0
- 平均最佳 IoU：0.9074
- 匹配人员姿态存在率：1.0
- 物体检测率：0.0
- 事件检测率：0.0

扩大到 20 条候选样本后的结果：

```bash
./.conda/bin/python scripts/run_ava_batch_eval.py \
  data/ava/ava_val_eval_subset.csv \
  --limit 20 \
  --clips-dir data/ava/clips \
  --output-root outputs/ava_batch_eval_limit20
```

结果：

- 请求样本：20
- 成功完成：2
- 下载失败：18
- Stage2 失败：0
- AVA 对齐失败：0
- 成功样本匹配率：1.0
- 成功样本平均最佳 IoU：0.9074
- 匹配人员姿态存在率：1.0
- 物体检测率：0.0
- 事件检测率：0.0

失败原因：

- 多数候选视频的 YouTube 源已不可用、地区受限、版权限制，或需要登录/反机器人校验。
- 这说明继续扩大 AVA 时，需要先做“可下载性采样”，跳过不可用视频后再补足每类动作样本。

通用动作特征提取：

```bash
./.conda/bin/python scripts/extract_ava_action_features.py \
  --batch-summary outputs/ava_batch_eval_limit3/batch_summary.json \
  --output outputs/ava_batch_eval_limit3/ava_action_features.csv \
  --json-output outputs/ava_batch_eval_limit3/ava_action_features.json
```

当前特征包括：

- AVA action ID、action name、action type
- AVA/Stage2 人框匹配 IoU
- Stage2 person ID
- 姿态是否存在
- 可见关键点数量
- 人框面积、宽高比、中心点
- 手腕到头部距离
- 当前帧人数
- 最近其他人距离
- 窗口内物体数量、最近物体类别和距离
- Stage2 事件数量

下载脚本：

```bash
./.conda/bin/python scripts/download_ava_clips.py \
  data/ava/ava_val_eval_subset.csv \
  --limit 1 \
  --output-dir data/ava/clips
```

已成功下载示例片段：

```text
data/ava/clips/5BDj0ow5hnA_t0997_p114_a15_answer_phone.mp4
```

并运行第二阶段：

```bash
./.conda/bin/video-analyst stage2 \
  data/ava/clips/5BDj0ow5hnA_t0997_p114_a15_answer_phone.mp4 \
  -o outputs/ava_clip_answer_phone_stage2
```

结果：

- 视频时长：4 秒
- 姿态结果：282
- 目标物体：0
- 行为事件：0

已新增 AVA 对齐评估：

```bash
./.conda/bin/python scripts/evaluate_ava_alignment.py \
  data/ava/ava_val_eval_subset.csv \
  data/ava/clips/5BDj0ow5hnA_t0997_p114_a15_answer_phone.mp4 \
  outputs/ava_clip_answer_phone_stage2 \
  --output outputs/ava_clip_answer_phone_stage2/ava_alignment.json
```

对齐结果：

- AVA 目标框：`[186.88, 432.0, 238.08, 573.12]`
- Stage2 最佳人员框：`[189.484, 435.409, 237.209, 579.307]`
- IoU：0.8739
- 匹配帧：55
- Stage2 人员 ID：40
- 该人员有姿态结果：是
- 窗口内目标物体数量：0
- 行为事件数量：0

结论：当前系统在该 AVA 片段上能准确定位 AVA 标注的人，但对 `answer phone` 这类动作，仅依赖 COCO 的 `cell phone` 物体检测不够稳定，需要后续引入更泛化的动作候选生成或多模态复核。

## 注意事项

- AVA 的动作标签是 1 秒间隔，不是连续逐帧标注。
- AVA 是电影片段，场景复杂，遮挡和镜头变化比 NTU 更接近真实世界，但不一定像固定监控视角。
- AVA 标签是“原子动作”，`drink` 不一定等价于“喝水”，但可作为当前规则的强参考。
- AVA 只提供标注和视频 ID，实际视频获取依赖 YouTube 可用性。
