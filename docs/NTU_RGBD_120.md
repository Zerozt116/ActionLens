# NTU RGB+D 120 接入计划

## 目标

使用 NTU RGB+D 120 中的目标动作样本，验证和调参当前第二阶段规则基线。

优先动作：

- A001：drink water，对应本项目 `drinking_water`
- A028：phone call，对应本项目 `talking_on_phone`
- A029：play with phone/tablet，作为后续手机行为扩展

## 获取方式

官方页面：https://rose1.ntu.edu.sg/dataset/actionRecognition/

数据集下载需要：

1. 注册账号。
2. 提交 request form。
3. 接受 Release Agreement。
4. 等待官方审批。
5. 登录后下载对应模态数据。

## 账号注册字段

官方注册页当前可见字段：

- `Username`：必填，150 字符以内，只允许字母、数字和 `@`、`.`、`+`、`-`、`_`
- `Last name`
- `First name`
- `Institute/Organization Email`：必填，建议使用学校或研究机构邮箱
- `Password`：必填，至少 8 位，不能太常见、不能全数字、不能与个人信息过于相似
- `Password confirmation`：必填

## Request Form 字段

已根据登录后的申请页截图确认字段：

- `Project Title`
- `Title`
- `First name`
- `Last name`
- `Organization Email`
- `Designation`
- `Organization`
- `Country`
- `Supervisor Title`
- `Supervisor First Name`
- `Supervisor Last Name`
- `Supervisor Organization Email`
- `Supervisor Designation`
- `I accept the above Dataset Release Agreement`

建议填写内容：

- `Project Title`：`Video-based Human Behavior Analysis for Person-level Action Detection`
- `Title`：按身份选择，例如学生通常可选 `Mr.` / `Ms.`，博士或教师可选 `Dr.`
- `First name`：建议用拼音名，例如 `Zitao`
- `Last name`：建议用拼音姓，例如 `Lin`
- `Organization Email`：使用学校/机构邮箱
- `Designation`：填写身份，不填学号，例如 `Undergraduate Student`、`Master Student`、`PhD Student` 或 `Research Student`
- `Organization`：学校/机构英文名，例如 `Jinan University`
- `Country`：`China`
- `Supervisor Title`：导师称谓，例如 `Dr.` / `Prof.`
- `Supervisor First Name`：导师英文名或拼音名
- `Supervisor Last Name`：导师英文姓
- `Supervisor Organization Email`：导师机构邮箱
- `Supervisor Designation`：导师职位，例如 `Professor`、`Associate Professor`、`Assistant Professor`、`Lecturer`
- `I accept...`：阅读 Release Agreement 后勾选

建议优先下载：

- RGB videos：用于跑当前 YOLO/ByteTrack/YOLO Pose/规则管线。
- 3D skeletons：用于后续直接做骨架动作分类或校验 YOLO Pose。

暂不优先下载：

- full depth maps
- masked depth maps
- IR data

这些模态体积很大，第一轮验证不需要。

## 文件名解析

NTU 文件名通常包含以下字段：

```text
S001C001P001R001A001_rgb.avi
```

含义：

- `S001`：setup ID
- `C001`：camera ID
- `P001`：performer/person ID
- `R001`：replication ID
- `A001`：action ID

本项目先根据 `Axxx` 筛选目标动作。

## 生成 manifest

下载并解压 RGB 视频后，运行：

```bash
./.conda/bin/python scripts/prepare_ntu_rgbd.py \
  /path/to/nturgbd_rgb \
  --output data/ntu_rgbd_120_targets.csv
```

只筛选喝水、打电话、玩手机：

```bash
./.conda/bin/python scripts/prepare_ntu_rgbd.py \
  /path/to/nturgbd_rgb \
  --actions 1 28 29 \
  --output data/ntu_rgbd_120_targets.csv \
  --json-output data/ntu_rgbd_120_targets.json
```

## 后续验证流程

拿到 manifest 后，可以先抽取少量样本：

```bash
head -20 data/ntu_rgbd_120_targets.csv
```

然后对其中一个 RGB 视频运行：

```bash
./.conda/bin/video-analyst stage2 /path/to/sample.avi -o outputs/ntu_sample_stage2
```

检查：

- `objects.json` 是否检测到杯子、手机等目标。
- `poses.json` 是否稳定输出人员姿态。
- `frame_action_scores.json` 是否逐帧命中目标动作。
- `events.json` 是否合并出正确行为时间段。

## 注意事项

- NTU 是受控实验室场景，与真实监控视频分布不同。
- NTU 的单个样本通常是一段完整动作，不一定有精确动作开始/结束边界。
- 对当前规则系统来说，NTU 更适合做动作存在性验证和阈值调参，不适合直接评估复杂多人跟踪。
