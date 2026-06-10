# InternVideo2 论文读书笔记

**论文标题：** InternVideo2: Scaling Foundation Models for Multimodal Video Understanding

**来源：** arXiv:2403.15377v4 / OpenGVLab, 上海 AI 实验室, 南京大学, 中科院深圳先进院 (2024)

**作者：** Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Chenting Wang, Guo Chen 等

**论文地址：** https://arxiv.org/abs/2403.15377

**代码地址：** https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo2

---

## 一、论文解决什么问题

构建通用的视频基础模型（Video Foundation Model）面临三大挑战：

1. **时空表征学习**：视频不同于图像，需要同时建模空间外观和时间动态，如何从零学习高质量的时空特征是基础问题。
2. **多模态语义对齐**：视频天然包含视觉、音频、语音等多模态信息，如何有效对齐这些模态的语义，使模型具备跨模态理解能力。
3. **开放式推理能力**：仅做分类或检索不够，模型还需支持视频对话、过程推理等开放式任务。

此前的工作（如 InternVideo、UMT、VideoPrism）通常只整合其中的两种学习范式（如掩码重建 + 对比学习），InternVideo2 进一步将**掩码重建、跨模态对比学习、下一 token 预测**三种范式统一到渐进式训练框架中，并将视频编码器扩展至 **6B 参数**。

---

## 二、核心方法：三阶段渐进训练

InternVideo2 的核心思路是**渐进式训练（progressive training）**，三个阶段依次递进，后一阶段以前一阶段的模型为初始化：

### Stage 1：未掩码视频 token 重建

- **目标**：学习基础时空感知能力
- **方法**：使用两个专家教师模型（InternViT-6B 多模态教师 + VideoMAEv2-g 运动感知教师）指导视频编码器做 token 级重建
- **掩码策略**：逐帧 mask 掉 80% 的 token，仅对齐未掩码部分的输出（MSE 损失）
- **对齐细节**：InternViT 的最后 6 层 + VideoMAEv2 的最后 4 层 + InternViT 的最终输出 token，分别通过可学习 MLP 与视频编码器对应层对齐
- **训练后**：丢弃所有投影层，仅保留基础编码器
- **设计要点**：双教师策略使编码器既具备多模态友好性（InternViT），又增强时间敏感性（VideoMAEv2），在 SthSthV2 等时序敏感任务上效果显著

### Stage 2：视频-音频-语音-文本对齐

- **目标**：通过跨模态对比学习对齐视频与其他模态的语义
- **编码器配置**：
  - 视频编码器：Stage 1 输出的 ViT（巨大）
  - 音频编码器：12 层 Transformer，BEATs 初始化（90M）
  - 文本编码器：BERT-Large 的前 19 层
  - 多模态解码器：BERT-Large 的后 5 层 + 交叉注意力
- **损失函数**：L = L_CON（跨模态对比）+ L_MAC（跨模态匹配）+ L_MLM（掩码语言建模）
- **两步骤训练**：
  1. **对齐阶段**：冻结音频编码器，mask 视频 token，对齐视觉-音频-文本
  2. **后预训练阶段**：冻结视频编码器（6B 参数太大），不 mask，联合对齐所有模态

### Stage 3：下一 token 预测

- **目标**：增强视频对话和开放式推理能力
- **方法**：通过 QFormer 将 InternVideo2 连接到开源 LLM（Mistral-7B），构建 VideoChat2 系统
- **高清后训练**：将视频分成最多 6 个 224×224 子视频 + 1 个全局缩放子视频，先 8 帧训练 1 epoch，再 16 帧训练 1 epoch
- **参数更新**：视频编码器 + QFormer 全参数更新，LLM 使用 LoRA 更新

---

## 三、数据贡献

### 3.1 K-Mash（视频仅数据，Stage 1）

从 Kinetics、SthSthV2、Moments in Time、ActivityNet、HACS 等动作识别数据集中筛选，加上 YouTube 额外视频，共 **200 万视频**，无标签。

### 3.2 InternVid2（多模态视频数据，Stage 2）

- **规模**：100M 视频 + 视频-音频-语音描述
- **时间一致性**：使用 AutoShot（基于时序语义变化）替代 FFmpeg 的 SceneDet（基于像素差异）做视频切分，生成语义完整的片段
- **VidCap 标注系统**：
  - 视频描述器：基于 InternVid 的视频描述管线
  - 音频描述器：基于 VideoChat + BEATs + WavCaps 训练的 QFormer
  - 语音描述器：WhisperV2-large
  - 融合：Vicuna-1.5 LLM 将单模态描述融合为跨模态描述（AV 和 AVS 两种）
- 每个视频自动生成 5 种描述（A、V、S、AV、AVS）

### 3.3 指令微调数据（Stage 3）

基于 MVBench 更新版本（1.9M 样本，34 个来源），减少 WebVid/CoCo 的 caption 数据，增加 S-MiT 数据以提高多样性。高清阶段加入 GPT-4 标注数据及 PerceptionTestQA、TVQA、NTU-RGB-D 等。

### 3.4 总数据量

| 阶段 | 数据规模 |
|------|---------|
| Stage 1 | 2M 视频（K-Mash） |
| Stage 2 | 300M 图像-文本 + 50M 视频-文本 + 50M 视频-音频-语音-文本 |
| Stage 3 | 2.1M 指令数据 |

---

## 四、模型架构细节

### 4.1 视频编码器 ViT-6B

- 输入：稀疏采样 8 帧，14×14 空间下采样
- Patch embedding：kernel 1×14×14，stride 1×14×14，输出 3200 token（Table 20）
- 注意力池化（Attention Pooling）
- 位置编码：可学习的 3D 正弦-余弦初始化
- 编码器：48 层 Transformer（2048 维通道，12800 维 MLP）
- 总参数：约 **6B**

### 4.2 训练资源

| 阶段 | GPU | 时间 |
|------|-----|------|
| Stage 1 | 256 × A100 | 18 天 |
| Stage 2 | 256 × A100 | 14 天 |
| Stage 3 | 64 × A100 | 3 天 |

使用 DeepSpeed 和 FlashAttention 加速。

---

## 五、实验结果

### 5.1 动作识别（端到端微调）

| 数据集 | InternVideo2-6B | 此前 SOTA |
|--------|----------------|-----------|
| K400 | **92.1** | 91.1 (InternVideo, ensemble) |
| K600 | **91.9** | 91.3 (InternVideo, ensemble) |
| K700 | **85.9** | 84.0 (InternVideo, ensemble) |
| SthSthV2 | **77.5** | 77.3 (MVD) |
| MiT | **51.2** | 49.0 (CoCa-g) |
| ANet | **95.9** | 94.7 (UniFormerV2) |
| HACS | **97.0** | 95.4 (UniFormerV2) |

仅用 16 帧 × 224 分辨率即达到 SOTA，此前方法通常需要更高分辨率（576）或模型集成。

### 5.2 时间动作定位（TAL）

使用 ActionFormer 作为检测头，InternVideo2-6B 第 7 层特征：

| 数据集 | InternVideo2-6B mAP |
|--------|---------------------|
| THUMOS14 | **72.0** |
| HACS | **43.3** |
| ActivityNet | **41.2** |
| FineAction | **27.7** |

四个数据集均为最高 mAP。1B 到 6B 的扩展在大多数数据集上带来显著提升。

### 5.3 视频检索（零样本 + 微调）

在 6 个基准上评估（MSR-VTT, LSMDC, DiDeMo, MSVD, ANet, VATEX），InternVideo2-6B 在几乎所有 T2V 和 V2T 指标上均领先。例如微调 MSR-VTT T2V R@1 达到 **62.8**（此前 UMT-L 为 58.8）。

### 5.4 视频时间定位

使用 CG-DETR 作为 grounding 头：

| 数据集 | 指标 | InternVideo2-6B |
|--------|------|-----------------|
| QVHighlight | mAP | **49.24** |
| Charades-STA | R1@0.5 | **70.03** |

### 5.5 视频对话

VideoChat2-HD-F16 配置（InternVideo2 + Mistral-7B）：

| 基准 | 得分 |
|------|------|
| MVBench | **67.2** |
| Egoschema | 60.0 |
| Perception Test | **63.4** |

MVBench 和 Perception Test 上超过 GPT-4V 和 Gemini 1.0 Ultra。

### 5.6 音频任务

音频检索（AudioCaps R@1: **55.2**）、音频 QA（ClothoAQA: **30.14**）、音频分类（ESC-50: **98.6**）均达到 SOTA，说明跨模态对比学习对各模态互有裨益。

---

## 六、关键消融实验

### 6.1 模型规模的影响

将视频编码器从 1B 扩展到 6B，零样本动作识别和视频检索分别提升 +1.4% 和 +1.9%，验证了扩展的有效性。

### 6.2 Stage 1 教师选择

仅用多模态教师（CLIP/InternViT）vs 多模态 + 运动感知教师（VideoMAEv2）：双教师策略在 SthSthV2 上显著提升，凸显运动感知教师对时序建模的重要性。

### 6.3 Stage 2 音频编码器的影响

仅引入音频编码器并与视频-文本编码器联合训练时效果最好；引入语音编码器反而有一定损害。

### 6.4 时间分割与描述质量

- AutoShot（语义切分）比 SceneDet（像素切分）将 MSR-VTT 零样本 T2V R@1 提升约 **7 个点**
- 视频-音频-语音融合描述比单一视频描述提升约 2.4 个点

### 6.5 QFormer 指令注入

在 Stage 3 训练中向 QFormer 注入问题文本会导致域外性能下降（Egoschema -2.9），说明在大规模 VideoLLM 中这种做法容易过拟合。

---

## 七、对本课题的启示

### 7.1 视频特征提取的参考

InternVideo2 在 TAL 实验中直接使用第 7 层特征作为 ActionFormer 的输入，验证了其作为通用视频特征提取器的有效性。课题中 Stage 3 的 SlowFast 特征可考虑替换为 InternVideo2 特征，预期可带来 TAL 性能的显著提升（THUMOS14 mAP 从 69.5 提升至 72.0）。

### 7.2 时间分割的重要性

AutoShot 替代 SceneDet 在下游检索上提升巨大，说明视频数据的预处理质量对模型性能至关重要。课题中 Charades 数据集的 50 个视频片段在 TAL 训练前也应确保语义完整性。

### 7.3 视频对话能力的参考

InternVideo2-Chat 在动作序列识别、混淆动作识别、时序计数、意外动作推理等定性案例中展现了优于 GPT-4V 和 Gemini Pro 的能力。这些能力正是课题"大模型复核"模块所需要的：对检测到的行为进行二次验证和推理。

### 7.4 与课题方案的对接

| 论文内容 | 课题对应 |
|---------|---------|
| TAL 特征提取（Table 7） | Stage 3 时间动作定位的 SlowFast 特征替代 |
| 视频对话系统 VideoChat2 | Stage 2.5 大模型复核的候选方案 |
| 视频-音频-文本对齐 | 多模态信息融合的参考范式 |
| 渐进式训练（三阶段） | 分阶段构建视频理解系统的工程参考 |

### 7.5 局限性

论文自承 InternVideo2 没有引入新的架构设计，而是对现有技术的扩展和数据处理改进。固定输入分辨率（224）、固定采样率（8/16 帧）、高度压缩的 token 仍限制了细粒度视频理解。这些局限在课题中也需注意。

---

## 八、核心 Takeaway

1. **渐进式训练统一三种学习范式**：掩码重建 → 跨模态对比 → 下一 token 预测，三阶段依次递进是构建大规模视频基础模型的有效路径。
2. **数据质量比数据数量更重要**：AutoShot 语义切分 + VidCap 多模态标注系统，仅改进数据处理即可带来显著下游提升。
3. **双教师策略的互补性**：多模态教师（InternViT）和运动感知教师（VideoMAEv2）各有所长，二者结合使编码器既多模态友好又时间敏感。
4. **6B 视频编码器的扩展性**：在 60+ 任务上达到 SOTA，验证了大规模视频编码器的泛化能力。
5. **TAL 特征可直接替换 SlowFast**：InternVideo2 特征 + ActionFormer 的组合在四个 TAL 基准上均创新高，对课题有直接参考价值。

---

## 九、引用

```bibtex
@article{wang2024internvideo2,
  title={InternVideo2: Scaling Foundation Models for Multimodal Video Understanding},
  author={Wang, Yi and Li, Kunchang and Li, Xinhao and Yu, Jiashuo and He, Yinan and Wang, Chenting and Chen, Guo and Pei, Baoqi and Zheng, Rongkun and Yan, Ziang and Xu, Jilan and Wang, Zun and Shi, Yansong and Jiang, Tianxiang and Li, Songze and Zhang, Hongjie and Huang, Yifei and Qiao, Yu and Wang, Yali and Wang, Limin},
  journal={arXiv preprint arXiv:2403.15377},
  year={2024}
}
```
