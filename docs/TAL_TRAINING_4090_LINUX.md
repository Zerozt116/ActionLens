# ActionFormer Phase 2 训练手册（Linux + RTX 4090）

本手册面向 **Linux + NVIDIA RTX 4090 (24 GB) 训练 ActionFormer** 的完整流程。Phase 1 smoke test 已在本地 M5 上验证（见 `项目进度.md` §32），本手册承接 Phase 2。

## 0. 硬件 & 资源预估

| 项目 | 规格 |
| --- | --- |
| GPU | NVIDIA RTX 4090 (24 GB VRAM, Compute 8.9, Ada Lovelace) |
| 推荐内存 | 64 GB（不够 32 GB 也能跑） |
| 磁盘 | 训练用 30 GB（含 SlowFast 特征 + checkpoints） |
| 单 epoch 时间（batch=8, R50） | **~1 小时** |
| 30 epoch 总计 | **~30 小时（约 1.5 天）** |
| 50 epoch 总计 | **~50 小时（约 2 天）** |
| 特征提取 50 视频 | **10-20 分钟** |
| 预训练权重下载 | THUMOS14 ≈ 1.5 GB / EPIC-Kitchens ≈ 1 GB |

> 4090 比 3060 快约 **6×**（FP32 算力 82 vs 13 TFLOPS），可以放心用更大的 batch、更大的 backbone。**R101 backbone + batch=16** 在 4090 上单 epoch 约 2 小时，30 epoch 仍只需 2.5 天。

## 1. 环境准备

### 1.1 NVIDIA 驱动

```bash
# 查看现有驱动
nvidia-smi

# 如果没有：Ubuntu 22.04 装最新稳定版
sudo apt update
sudo apt install -y nvidia-driver-550  # 或最新稳定版
sudo reboot

# 重启后验证
nvidia-smi
# 应显示: NVIDIA-SMI 5xx.xx   Driver Version: 550.xx   CUDA Version: 12.x
#          GPU: NVIDIA GeForce RTX 4090
```

驱动版本 ≥ 535 即可（推荐 550+，对 Ada 优化更好）。

### 1.2 CUDA Toolkit

`nvidia-smi` 顶部的 `CUDA Version: 12.x` 是驱动支持的**最高**CUDA 版本。**不一定要单独装 CUDA Toolkit**——PyTorch 自带 CUDA runtime。

但如果你想用 `nvcc` 编译 C++ 扩展（ActionFormer 的 NMS 需要）：

```bash
# 选项 A：装系统级 CUDA Toolkit（推荐，C++ 编译需要）
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run --toolkit --silent --override

# 把 CUDA 加到 PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证
nvcc --version
```

### 1.3 Python 环境

```bash
# 装 miniconda（如果还没）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash
source ~/.bashrc

# 建 conda 环境
conda create -n actionformer python=3.10 -y
conda activate actionformer

# 装 PyTorch CUDA（4090 需要 CUDA 11.8+，推荐 12.1）
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# 验证
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0))"
# 期望: True 12.1 NVIDIA GeForce RTX 4090
```

### 1.4 系统依赖

```bash
sudo apt update
sudo apt install -y build-essential git wget ffmpeg libsm6 libxext6

# build-essential 提供 gcc, g++, make —— C++ NMS 编译需要
# 如果有 NVIDIA Video SDK 解码需求（pytorchvideo 内部用）：
sudo apt install -y nvidia-cuda-toolkit
```

---

## 2. 克隆项目 + 安装 Python 依赖

### 2.1 把项目从 Mac 传到 4090

**方式 A：scp（推荐如果有 SSH）**

```bash
# 在 Mac 上
cd /Users/tiu/project
tar --exclude='.conda' --exclude='outputs' --exclude='.git' \
    -czf video-analyst.tar.gz video-analyst/

scp video-analyst.tar.gz user@4090-box:~/

# 在 4090 上
mkdir -p ~/projects
mv ~/video-analyst.tar.gz ~/projects/
cd ~/projects && tar -xzf video-analyst.tar.gz
cd video-analyst
```

**方式 B：USB / 共享盘**

```bash
# 4090 上
mkdir -p ~/projects
cp /mnt/usb/video-analyst.tar.gz ~/projects/
cd ~/projects && tar -xzf video-analyst.tar.gz
cd video-analyst
```

**方式 C：rsync 增量**

```bash
# Mac 上
rsync -avz --exclude='.conda' --exclude='outputs' --exclude='.git' \
    /Users/tiu/project/video-analyst/ user@4090-box:~/projects/video-analyst/
```

### 2.2 装 Python 依赖

```bash
cd ~/projects/video-analyst
pip install -r requirements.txt
pip install h5py tensorboard pandas joblib einops pytorchvideo
```

### 2.3 编译 C++ NMS

这一步是 ActionFormer 后处理必需的。Linux 上直接 `gcc` 编译就行：

```bash
cd external/actionformer_release/libs/utils
python setup.py install --user
cd ../../..
```

成功输出：

```text
Installed .../site-packages/nms_1d_cpu-0.0.0....whl
```

验证：

```bash
python -c "import nms_1d_cpu; print('nms_1d_cpu OK')"
```

如果 `python setup.py install --user` 报 `UserWarning: pkg-config` 之类的提示，但最终 `Installed ...whl`，仍然算成功。

如果**编译失败**（极端情况，比如 gcc 版本太老），fallback 方案：

```bash
# 用我们写的纯 Python 版本覆盖 nms_1d_cpu
ln -sf ~/projects/video-analyst/scripts/tal/nms_1d_cpu.py \
       ~/miniconda3/envs/actionformer/lib/python3.10/site-packages/nms_1d_cpu.py
python -c "import nms_1d_cpu; print('Python fallback OK')"
```

### 2.4 验证 GPU + ActionFormer 联通

```bash
cd ~/projects/video-analyst
PYTHONPATH=. python scripts/tal/infer_smoke.py --device cuda
```

### 2.5 验证 SlowFast 后端

在处理真实视频之前，先跑合成帧自检：

```bash
python scripts/tal/check_slowfast_backend.py \
  --device cuda \
  --summary-output outputs/slowfast_backend_check_summary.json
```

成功标准：

```text
status = ok
feature_shape = [2304]
feature_dim_ok = true
```

期望输出：

```text
[setup] device = cuda
[model] built LocPointTransformer with 29,251,612 parameters
[forward] elapsed = 50-200 ms on 400 frames  ← CUDA 应该 100ms 左右
[PASS] smoke test succeeded
```

CUDA 上 400 帧 forward 应当 **< 200ms**。M5 CPU 是 10 秒，4090 应该快 50×。

---

## 3. 数据准备

### 3.1 视频文件

把 `data/charades/videos_480p/*.mp4` 同步到 4090。50 视频约 40 MB，scp 几秒就到：

```bash
# Mac 上
scp -r data/charades/videos_480p user@4090-box:~/projects/video-analyst/data/charades/
```

### 3.2 ActionFormer 标注格式

把 Mac 上生成的 `data/charades/tal_annotations.json`（2.9 MB）直接拷过去：

```bash
scp data/charades/tal_annotations.json user@4090-box:~/projects/video-analyst/data/charades/
```

格式兼容，无需重新生成。

### 3.3 视频特征提取

ActionFormer 输入是预提取特征。当前已经新增：

- `scripts/tal/extract_slowfast_features.py`
- `tests/test_extract_slowfast_features.py`

本机已完成 `cv2-smoke` 后端验证。这个后端把每帧 resize 为 32x24 RGB 并 flatten 成 2304 维，只用于确认特征缓存格式，不用于训练。

本机 smoke 命令：

```bash
./.conda/bin/python scripts/tal/extract_slowfast_features.py \
  outputs/charades_clips_50.csv \
  --output-dir data/charades/features_smoke \
  --summary-output outputs/tal_features_smoke_50_summary.json \
  --feature-manifest-output data/charades/features_smoke_manifest.json
```

输出 schema：

```text
<clip_id>.npz
└── feats: float32, shape = T x 2304
```

本机 smoke 结果：

```json
{
  "total": 50,
  "feature_dims": [2304],
  "total_feature_frames": 3743,
  "backend": "cv2-smoke"
}
```

4090 上的真实任务是运行已经接入的真实 SlowFast 后端：

```bash
python scripts/tal/extract_slowfast_features.py \
  outputs/charades_clips_50.csv \
  --backend slowfast-r50 \
  --output-dir data/charades/features \
  --summary-output outputs/tal_features_slowfast_50_summary.json \
  --feature-manifest-output data/charades/features_manifest.json \
  --device cuda
```

真实后端必须保持同样 schema：`.npz` 文件内包含 `feats`，shape 为 `T x 2304`。这样 ActionFormer loader 和后续 TAL 推理接口不需要再改。

建议先跑 limit1：

```bash
python scripts/tal/extract_slowfast_features.py \
  outputs/charades_clips_50.csv \
  --backend slowfast-r50 \
  --output-dir data/charades/features \
  --summary-output outputs/tal_features_slowfast_limit1_summary.json \
  --feature-manifest-output data/charades/features_manifest_limit1.json \
  --device cuda \
  --limit 1 \
  --overwrite
```

如果 summary 里 `feature_dims: [2304]`，再跑完整 50 个。

预计耗时：50 视频 × 30s 平均时长 + SlowFast 推理 5-10× 实时 = **10-20 分钟**（4090 上一遍跑完）。

### 3.3.1 dataloader smoke 验证

在生成真实 SlowFast 特征前，可以先用本机已有 smoke 特征验证 annotation 和 feature grid 对齐：

```bash
python scripts/tal/dataloader_smoke.py \
  outputs/charades_clips_50.csv \
  --annotations data/charades/tal_annotations.json \
  --feature-manifest data/charades/features_smoke_manifest.json \
  --summary-output outputs/tal_dataloader_smoke_50_summary.json
```

本机验证结果：

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

真实 SlowFast 特征生成后，应把 `--feature-manifest` 换成 `data/charades/features_manifest.json` 再跑一次。如果仍然是 `ok: 50`，说明训练前的数据输入链路正常。

### 3.4 数据划分

```bash
python scripts/tal/build_charades_splits.py \
  --annotations data/charades/tal_annotations.json \
  --clips-manifest outputs/charades_clips_50.csv \
  --output-dir data/charades/splits \
  --summary-output data/charades/splits/summary.json
```

当前 Mac 已生成：

```json
{
  "train": 32,
  "val": 7,
  "test": 11,
  "must_test": ["024PD", "8BG1T", "OINMN"]
}
```

如果你把整个项目目录同步到 4090，split 文件无需重新生成。

---

## 4. 训练

### 4.1 下载预训练权重

从官方下载 THUMOS14 / EPIC-Kitchens 预训练权重（建议 EPIC-Kitchens SlowFast，视觉域最接近 Charades）：

- 仓库 README: <https://github.com/happyharrycn/actionformer_release>
- 选 `ego4d_slowfast.yaml` 或 `epic_slowfast_noun.yaml` 配套的预训练权重

存到 `external/actionformer_release/pretrained/`，解压后：

```text
external/actionformer_release/pretrained/
└── epic_slowfast_noun/
    └── model_best.pth.tar
```

### 4.2 写 Charades 配置

`external/actionformer_release/configs/charades_slowfast.yaml` 已经写好，核心配置如下：

```yaml
dataset_name: charades
train_split: ['train']
val_split: ['val']
dataset: {
  json_file: ./data/charades/tal_annotations.json,
  clips_manifest: ./outputs/charades_clips_50.csv,
  feature_manifest: ./data/charades/features_manifest.json,
  split_folder: ./data/charades/splits,
  feat_folder: ./data/charades/features,
  file_prefix: ~,
  file_ext: .npz,
  num_classes: 24,           # 与 tal_annotations.json label_dict 一致
  input_dim: 2304,           # SlowFast R50 特征维度
  feat_stride: 15,           # 30fps / 2 feature_fps
  num_frames: 32,
  default_fps: 30,
  trunc_thresh: 0.5,
  crop_ratio: [0.9, 1.0],
  max_seq_len: 256,
}
model: {
  regression_range: [[0, 4], [4, 8], [8, 16], [16, 32], [32, 64], [64, 10000]],
  fpn_type: identity,
  max_buffer_len_factor: 4.0,
  n_mha_win_size: 8,
  n_head: 4,
  embd_dim: 256,
  fpn_dim: 256,
  head_dim: 256,
  embd_kernel_size: 3,
  backbone_arch: [2, 2, 5],
}
opt: {
  learning_rate: 0.0001,
  epochs: 30,
  weight_decay: 0.05,
}
loader: {
  batch_size: 4,
  num_workers: 4,
}
train_cfg: {
  init_loss_norm: 250,
  clip_grad_l2norm: 1.0,
  cls_prior_prob: 0.01,
  center_sample: radius,
  center_sample_radius: 1.5,
  label_smoothing: 0.1,
}
test_cfg: {
  pre_nms_thresh: 0.001,
  pre_nms_topk: 2000,
  roi_size: 4,
  nms_method: soft,
  nms_sigma: 0.5,
  iou_threshold: 0.5,
  min_score: 0.001,
  max_seg_num: 200,
  voting_thresh: 0.5,
}
```

> ⚠️ **R101 backbone 时 batch_size 调到 4**，max_seq_len 调到 1024，否则 OOM。

### 4.3 写 Charades 数据加载器

`external/actionformer_release/libs/datasets/` 下新建 `charades.py`：

```python
from .epic_kitchens import EpicKitchensDataset
from .datasets import register_dataset

@register_dataset("charades")
class CharadesDataset(EpicKitchensDataset):
    """Charades reuses EpicKitchens loader (same segment-annotation format)."""
    pass
```

然后在 `external/actionformer_release/libs/datasets/datasets.py` 末尾加：

```python
from . import charades  # noqa: F401
```

### 4.4 启动训练

```bash
cd ~/projects/video-analyst/external/actionformer_release
python ./train.py ./configs/charades_slowfast.yaml --output charades_50v
```

训练启动后创建 `ckpt/charades_50v/`：

- `logs/` — TensorBoard 日志
- `ckpt.pth.tar` — checkpoint
- `config.yaml` — 训练配置备份

**后台跑 + 监控**：

```bash
# 起后台训练
nohup python ./train.py ./configs/charades_slowfast.yaml --output charades_50v \
    > /tmp/charades_train.log 2>&1 &
TRAIN_PID=$!
echo "Training PID: $TRAIN_PID"

# 跟日志
tail -f /tmp/charades_train.log

# 监控 GPU
watch -n 2 nvidia-smi

# TensorBoard
tensorboard --logdir=ckpt/charades_50v/logs --bind_all &
```

### 4.5 时间预估（4090）

| 配置 | 单 epoch | 30 epoch |
| --- | ---: | ---: |
| R50 + batch=8 + max_seq=2304 | ~1 h | ~30 h |
| R50 + batch=16 + max_seq=2304 | ~50 min | ~25 h |
| R101 + batch=4 + max_seq=1024 | ~2 h | ~60 h |

推荐：先跑 **R50 + batch=8 + 10 epoch** 验证管线（**~10 小时**），确认 mAP 上来了再决定是否跑满 30 epoch。

### 4.6 评估

```bash
cd ~/projects/video-analyst/external/actionformer_release
python ./eval.py ./configs/charades_slowfast.yaml ./ckpt/charades_50v
```

期望输出：

```text
=== Results on Charades ===
mAP@0.50: 0.30+
mAP@0.75: 0.15+
Average: 0.20+
```

参考：ActionFormer 论文在 Charades 全量上 mAP=21.5%。50 video 子集数据更少，预期值接近或略低（避免过拟合）/略高（小数据集上的常见现象）。

---

## 5. 导出权重 + 传回 Mac

### 5.1 提取 best checkpoint

```bash
cd ~/projects/video-analyst/external/actionformer_release

# 看 best epoch
grep "Best" /tmp/charades_train.log

# 直接用最后一个（每隔 N epoch 自动保存的）
ls ckpt/charades_50v/
cp ckpt/charades_50v/ckpt.pth.tar ~/transfer/charades_50v_best.pth.tar
```

### 5.2 传回 Mac

```bash
# 从 4090 到 Mac
scp ~/transfer/charades_50v_best.pth.tar user@mac:~/Downloads/

# 或者 USB / 网盘
```

### 5.3 在 Mac 上做推理对比

回到 `video-analyst/` 根目录：

```bash
# 写 scripts/tal/infer_real.py（Phase 3 任务）
PYTHONPATH=. python scripts/tal/infer_real.py \
  --weights ~/Downloads/charades_50v_best.pth.tar \
  --config external/actionformer_release/configs/charades_slowfast.yaml \
  --video data/charades/clips_480p/8BG1T_c106_t5.60_45.46.mp4
```

输出与 Stage2 同格式的 `events.json`，直接接 `fuse_stage2_vlm_events.py`。

---

## 6. 多卡训练（可选）

如果你机器上有 **多张 4090**，ActionFormer 支持 DataParallel / DistributedDataParallel：

```bash
# 单机多卡（2 张 4090）
python -m torch.distributed.launch --nproc_per_node=2 \
    ./train.py ./configs/charades_slowfast.yaml --output charades_50v_2gpu
```

2×4090 上 R50 + batch=8×2=16，单 epoch ~30 分钟，30 epoch **~15 小时**。

---

## 7. 常见问题

### 7.1 找不到 CUDA

```text
RuntimeError: Found no NVIDIA driver on your system.
```

```bash
nvidia-smi  # 看驱动
nvcc --version  # 看 CUDA Toolkit
```

如果 `nvidia-smi` 看不到：装驱动（§1.1）。如果 `nvcc` 找不到：装 CUDA Toolkit（§1.2）。

### 7.2 PyTorch CUDA 不匹配

```text
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

PyTorch 编译时用的 CUDA 跟当前驱动不匹配。重新装：

```bash
pip uninstall torch torchvision
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121
```

### 7.3 C++ NMS 编译失败

```text
error: command 'gcc' failed
```

```bash
sudo apt install -y build-essential
# 再编一次
cd external/actionformer_release/libs/utils
python setup.py install --user
```

或用我们写的纯 Python fallback（§2.3）。

### 7.4 SlowFast 模型加载失败

```text
ImportError: No module named 'pytorchvideo'
```

```bash
pip install pytorchvideo
```

### 7.5 显存 OOM

```text
RuntimeError: CUDA out of memory.
```

- 减小 `loader.batch_size`（8 → 4）
- 减小 `dataset.max_seq_len`（2304 → 1024）
- 减小 `model.backbone_arch`（R50 → MobileNet）
- 关其他 GPU 进程：`nvidia-smi` 查 PID，`kill -9 PID`

### 7.6 训练时 mAP 一直是 0

- 检查 `num_classes` = 24（不是 20 或 157）
- 检查特征 `*.npy` 形状是 `(2304, T)` 不是 `(T, 2304)`
- 检查 `dataset.default_fps` = 30
- 检查 `loader.batch_size` ≥ 1 且 `> 0`

### 7.7 训练中断想恢复

```bash
python ./train.py ./configs/charades_slowfast.yaml --output charades_50v \
    --resume ckpt/charades_50v
```

### 7.8 WSL 限制没了，4090 还有什么坑？

几乎没了。Linux + CUDA + 4090 是 PyTorch 官方测试平台之一。常见的真坑是：

- **冷启动慢**：第一次 import `torch` + `pytorchvideo` 约 5-10 秒，模型加载 2-3 秒
- **磁盘 IO**：如果数据在机械盘 / NAS 上，特征提取会成瓶颈。**NVMe SSD 上 50 视频 10 分钟**
- **CUDA 版本不匹配**：见 §7.2

---

## 8. 阶段交付物清单

| 输出 | 位置 |
| --- | --- |
| Charades smoke 特征（已完成，仅格式验证） | `data/charades/features_smoke/*.npz` (50 files) |
| Charades SlowFast 特征（待 GPU） | `data/charades/features/*.npz` (50 files) |
| ActionFormer 训练配置 | `external/actionformer_release/configs/charades_slowfast.yaml` |
| smoke 训练配置（已完成） | `external/actionformer_release/configs/charades_smoke.yaml` |
| 数据加载器（已完成） | `external/actionformer_release/libs/datasets/charades.py` |
| 官方 adapter smoke 脚本（已完成） | `scripts/tal/actionformer_charades_smoke.py` |
| 训练 checkpoint | `external/actionformer_release/ckpt/charades_50v/ckpt.pth.tar` |
| 训练日志 | `external/actionformer_release/ckpt/charades_50v/logs/` |
| TensorBoard 摘要 | 通过 `tensorboard --logdir=ckpt/charades_50v/logs` |
| 评估结果 | `external/actionformer_release/ckpt/charades_50v/results.json` |

完成后回 Mac：
- 把 `ckpt.pth.tar` 拷回 → `external/actionformer_tal_weights/`
- 写 `scripts/tal/infer_real.py`
- 在 20 视频 batch 上做 规则 vs TAL A/B 对比

---

## 9. 时间线建议

```text
Day 1 上午    NVIDIA 驱动 + PyTorch + 验证（1h）
Day 1 下午    数据传输 + SlowFast 特征提取（30min）
Day 2 上午    写 charades_slowfast.yaml + 真实特征 smoke（1h）
Day 2 下午    跑 1 epoch smoke（~1h）
Day 3-4       跑 10-30 epoch 完整训练（10-30h）
Day 5         评估 + 导出 + 传回 Mac（1h）
Day 5-6       在 Mac 上做 A/B 对比 + 写 Section 33
```

**总投入**：5-6 天（其中 3-4 天是 GPU 训练等待时间，可以并行做其他事）。

---

## 10. 相关文档

- [TAL_MODEL.md](TAL_MODEL.md) — 方案 C 总体设计
- [TAL_4090_EXECUTION_PACKAGE.md](TAL_4090_EXECUTION_PACKAGE.md) — 4090 逐步执行包
- [Charades.md](Charades.md) — Charades 数据集接入
- [项目进度.md §31](../项目进度.md) — 规则基线 20 视频评估
- [项目进度.md §32](../项目进度.md) — Phase 1 smoke test
- `papers/03_ActionFormer_读书笔记.md` — ActionFormer 论文笔记
- `papers/05_SlowFast_读书笔记.md` — SlowFast 论文笔记

## 11. 2026-06-09 Mac 端已完成的训练前置项

这些项已经不需要在 4090 上重复实现：

- `external/actionformer_release/libs/datasets/charades.py` 已注册 `make_dataset("charades", ...)`。
- `external/actionformer_release/configs/charades_smoke.yaml` 已用 smoke 特征跑通。
- `external/actionformer_release/configs/charades_slowfast.yaml` 已准备。
- `scripts/tal/actionformer_charades_smoke.py` 已验证训练态 loss 和评估态 postprocess。
- `scripts/tal/extract_slowfast_features.py --backend slowfast-r50` 已接入，等待 Linux GPU 验证。
- `scripts/tal/build_charades_splits.py` 已生成 50 视频 split。
- `outputs/actionformer_charades_smoke_train_summary.json`
- `outputs/actionformer_charades_smoke_eval_summary.json`

已通过的本机 smoke：

```text
dataset_length = 50
sample feats_shape_cxt = [2304, 91]
sample segments_shape = [5, 2]
train final_loss = 0.30324
eval prediction segments = [103, 2]
```

4090 机器上的下一步不是重新写 adapter，而是：

1. 安装/验证 SlowFast 依赖。
2. 先跑 `extract_slowfast_features.py --backend slowfast-r50 --limit 1 --device cuda`。
3. limit1 成功后提取 50 个真实 `.npz` 特征。
4. 用真实 `features_manifest.json` 跑 dataloader smoke。
5. 先跑 1 epoch smoke，再决定是否开始完整训练。
