# Video-LLaVA 论文读书笔记

**论文标题：** Video-LLaVA: Learning United Visual Representation by Alignment Before Projection

**来源：** arXiv:2311.10122 (2023)

**作者：** Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munang Ning, Peng Jin, Li Yuan (北京大学深圳研究生院, 鹏城实验室, PandaVilla Tech)

**论文地址：** https://arxiv.org/abs/2311.10122

**代码地址：** https://github.com/PKU-YuanGroup/Video-LLaVA

---

## 一、论文解决什么问题

现有的大型视觉语言模型（LVLM）通常只能处理单一视觉模态：要么只处理图像（如 LLaVA、MiniGPT-4），要么只处理视频（如 Video-ChatGPT）。少数尝试同时处理图像和视频的方法（如 VideoChat、Video-LLaMA）使用了共享视觉编码器，但由于图像和视频的固有差异，性能远不如专用模型。另一些方法（如 X-LLM、Macaw-LLM）为每种模态分配独立编码器，但同样表现不佳。

论文指出，根本问题在于**投影前的模态未对齐（misalignment before projection）**：

```
问题范式（现有方法）:
  Image Encoder → 图像特征 (空间 A) ──┐
                                       ├→ 共享 Projection → LLM
  Video Encoder → 视频特征 (空间 B) ──┘
  
  图像和视频特征在不同空间 → LLM 难以从多个弱投影层学习交互
```

当图像和视频特征处于不同的特征空间时，LLM 很难从若干简单的投影层中学到跨模态的交互。这一现象与 ALBEF 和 ViLT 中讨论的"融合前对齐"问题类似。

---

## 二、核心方法：投影前对齐 + 联合训练

### 2.1 统一视觉表示

Video-LLaVA 的关键洞察是：**在投影到 LLM 之前，先将图像和视频对齐到统一的视觉特征空间**。

实现方式是利用 LanguageBind 编码器家族。LanguageBind 以 OpenCLIP 为基础初始化（天然对齐图像和语言），然后用 VIDAL-10M 的 300 万视频-文本对将视频表示对齐到语言空间。由于图像和视频都与语言空间对齐，它们**自然而然地收敛到统一的视觉特征空间**（emergent alignment）。

```
Video-LLaVA 范式:
  LanguageBind Image Encoder ──┐
                                ├→ 统一视觉空间 → 共享 Projection → LLM
  LanguageBind Video Encoder ──┘
  
  编码器已将图像/视频对齐到语言空间 → 投影层只需处理统一特征
```

### 2.2 联合训练

对齐之后，Video-LLaVA 在图像和视频数据上进行**联合训练**，每个 batch 中同时包含图像和视频样本（随机混合）。这使得 LLM 从统一视觉表示中学习多模态推理能力。

---

## 三、模型架构

```
输入: 图像或视频 + 文本指令
       ↓
  LanguageBind Encoder Zoo
  ├── Image Encoder (OpenCLIP-L/14 初始化)
  └── Video Encoder (LanguageBind-Video-LoRA)
       ↓
  统一视觉特征
       ↓
  Share Projection Layer
  (2 层全连接, GeLU 激活)
       ↓
  与文本 token 拼接
       ↓
  Vicuna-7B v1.5 (LLM)
       ↓
  生成文本回复
```

### 3.1 模型参数

| 组件 | 配置 |
|------|------|
| LLM | Vicuna-7B v1.5 |
| 视觉编码器 | LanguageBind (OpenCLIP-L/14 初始化) |
| 投影层 | 2 层全连接 + GeLU，图像/视频共享 |
| 文本 tokenizer | LLaMA (~32K 词表) |
| 图像分辨率 | 224×224 |
| 视频采样 | 均匀采样 8 帧 |

### 3.2 训练流程

**Stage 1: 理解预训练**

- 目的：让模型获得解读视觉信号的基本能力
- 数据：558K LAION-CC-SBU 图像-文本对 + 702K WebVid 视频-文本对
- 训练：冻结 LLM，仅训练投影层，1 epoch，batch size 256，lr=1e-3

**Stage 2: 指令微调**

- 目的：让模型根据不同指令生成对应回复
- 数据：665K LLaVA 1.5 图像指令数据 + 100K Video-ChatGPT 视频指令数据
- 训练：解冻 LLM，端到端微调，batch size 128，lr=2e-5（正文提及两阶段均为 1e-3，附录 Table 7 标注此阶段为 2e-5）

两阶段均使用 AdamW 优化器、cosine 学习率调度、warmup ratio 0.03。

---

## 四、实验与结果

### 4.1 视频理解（零样本视频问答）

| 方法 | LLM | MSVD-QA Acc | MSRVTT-QA Acc | TGIF-QA Acc | ActivityNet-QA Acc |
|------|-----|-------------|--------------|-------------|-------------------|
| VideoChat | 7B | 56.3 | 45.0 | 34.4 | - |
| Video-LLaMA | 7B | 51.6 | 29.6 | - | 12.4 |
| Video-ChatGPT | 7B | 64.9 | 49.3 | 51.4 | 35.2 |
| Chat-UniVi | 7B | 65.0 | 54.6 | 60.3 | 45.8 |
| **Video-LLaVA** | **7B** | **70.7** | **59.2** | **70.0** | **45.3** |

相比 Video-ChatGPT，Video-LLaVA 在四个数据集上分别提升 **+5.8%、+9.9%、+18.6%、+10.1%**。相比 Chat-UniVi（使用更多数据），Video-LLaVA 在 MSVD、MSRVTT、TGIF 上仍有优势。

### 4.2 图像理解（零样本图像问答）

| 方法 | LLM | VQAv2 | GQA | VisWiz | SQA-I | VQAT | POPE | MMB | LLaVA-W | MM-Vet |
|------|-----|-------|-----|--------|-------|------|------|-----|---------|--------|
| InstructBLIP-7B | V-7B | - | 49.2 | 34.5 | 60.5 | 50.1 | - | 36.0 | 60.9 | 26.2 |
| LLaVA-1.5† | L-7B | 72.3 | 56.9 | 47.8 | 67.9 | 49.2 | 83.3 | 59.5 | 63.3 | 25.7 |
| **Video-LLaVA** | **V-7B** | **74.7** | **60.3** | **48.1** | **66.4** | **51.8** | **84.4** | **60.9** | **73.1** | **32.0** |

† LLaVA-1.5† 使用 LanguageBind-Image 编码器的公平对比版本。

Video-LLaVA 在 9 个图像基准中的 8 个超越了使用相同图像编码器的 LLaVA-1.5†，说明**视频联合训练对图像理解也有帮助**。特别是在 LLaVA-Bench (+9.8) 和 MMBench (+1.4) 上提升显著。

### 4.3 物体幻觉评估（POPE）

| 方法 | Adversarial Acc | Popular Acc | Random Acc |
|------|----------------|------------|-----------|
| MiniGPT-4 (13B) | 66.6 | 68.3 | 77.8 |
| Chat-UniVi (7B) | 55.6 | 56.4 | 73.9 |
| LLaVA-1.5† (L-7B) | 84.3 | 79.8 | 85.7 |
| **Video-LLaVA** (7B) | **81.6** | **85.3** | **86.2** |

Video-LLaVA 在 Popular 和 Random 设置下优于 LLaVA-1.5†，说明统一视觉表示有助于减少幻觉。

---

## 五、核心消融实验

### 5.1 投影前对齐的效果

| 图像编码器 | 视觉表示 | VQAv2 | GQA | MMB | LLaVA-W | MM-Vet |
|-----------|---------|-------|-----|-----|---------|--------|
| MAE | Separated | 66.0 | 55.4 | 45.7 | 35.9 | 20.0 |
| CLIP-L/14 | Separated | 74.6 | 59.9 | 60.2 | 68.9 | 30.6 |
| LanguageBind | **United** | **74.7** | **60.3** | **60.9** | **73.1** | **32.0** |

United 表示在所有指标上均优于 Separated 表示。特别是 LLaVA-Bench (+4.2) 和 MM-Vet (+1.4) 提升明显。

### 5.2 联合训练的效果

| 训练方式 | MSVD Acc | MSRVTT Acc | TGIF Acc | ActivityNet Acc |
|---------|----------|-----------|----------|----------------|
| 仅视频 (Video-LLaVA*) | 64.8 | 58.3 | 67.8 | 40.7 |
| 图像+视频联合 | **70.7** | **59.2** | **70.0** | **45.3** |
| 提升 | **+5.9** | **+0.9** | **+2.2** | **+4.6** |

联合训练在所有视频基准上都带来显著提升，其中 MSVD (+5.9%) 和 ActivityNet (+4.6%) 提升最大。这证明了图像和视频数据的互补性。

### 5.3 图像理解也受益于视频联合训练

对比 Video-LLaVA 和仅用图像训练的 LLaVA-1.5†，在 9 个图像基准中有 8 个表现更好。特别是在 POPE（减少幻觉）、LLaVA-Bench（复杂推理）和 MMBench 上的提升，说明视频数据帮助 LLM 更好地理解时序上下文和视觉概念。

---

## 六、关键洞见总结

1. **对齐先于投影（Alignment Before Projection）是核心原则**：在将视觉特征送入 LLM 之前，先确保不同模态处于统一的特征空间。这比在投影后做对齐更有效。
2. **图像和视频的互补性超出预期**：联合训练不仅提升了视频理解，也提升了图像理解（8/9 基准）。这说明统一视觉表示让 LLM 学到了更通用的视觉概念。
3. **简单设计的力量**：Video-LLaVA 的架构极其简洁——共享编码器 + 共享投影 + LLM，没有复杂的模块，但性能超越了所有专用模型。
4. **LanguageBind 编码器的关键作用**：OpenCLIP 天然对齐图像-语言，LanguageBind 进一步对齐视频-语言，这种传递性对齐是实现统一视觉空间的基础。
5. **1 epoch 即可达到优秀性能**：Video-LLaVA 仅训练 1 epoch 就在 15 个基准上取得竞争力结果，说明预对齐大幅降低了学习难度。

---

## 七、与本课题的关联

Video-LLaVA 与本课题中"大模型复核"模块（课题方案 §4.7）以及整体行为理解管线直接相关。

### 7.1 作为行为理解的大模型组件

课题的最终输出是结构化的行为时间线（JSON 格式），其中需要大模型对前端管线（检测→跟踪→特征提取→时间定位）的结果进行语义验证。Video-LLaVA 作为能同时理解图像和视频的 LVLM，非常适合这一角色：

```
前端管线输出:
  "Person P001 在 10:32:15-10:32:45 期间疑似在使用某物体"
         ↓
Video-LLaVA:
  输入: 对应时间段的视频片段 + 查询指令
  输出: "该人员在 10:32:15 开始接近打印机，10:32:20 按下按钮，
         10:32:45 取走打印文件后离开"
```

### 7.2 统一视觉表示对课题的启示

课题涉及多种视觉信号：
- **图像帧**：YOLO 检测、Grounding DINO 开放检测
- **视频片段**：SlowFast 行为识别、BMN/ActionFormer 时间定位
- **视频分割**：SAM 2 的 mask 序列

Video-LLaVA 的经验表明，将这些不同来源的视觉特征对齐到统一空间后，LLM 可以更高效地进行跨模态推理。

### 7.3 与其他论文的关系

```
本课题中的论文协作链:
  AVA (行为定义) → ByteTrack (跟踪) → SlowFast (特征)
       → ActionFormer/BMN (时间定位)
       → ST-HOI (人-物交互) + Grounding DINO (开放检测)
       → SAM 2 (精细分割)
       → Video-LLaVA / InternVideo2 / Qwen2.5-VL (大模型复核)
```

Video-LLaVA 位于管线的最后端，负责将前端所有模块的输出进行语义层面的综合理解和验证。

### 7.4 局限性对课题的影响

Video-LLaVA 的已知局限：
- **长视频理解较弱**：仅均匀采样 8 帧，对 ActivityNet 等长视频基准表现有限。课题中可通过对长视频分段处理来缓解。
- **训练成本高**：8 张 A100-80G 训练 3-4 天。但推理阶段较轻（Vicuna-7B），可在单卡上部署。
- **缺乏时间关系推理**：论文明确指出未来可探索时间戳嵌入，使模型能回答时序关系问题。这正是课题需要的能力（"先做了什么，后做了什么"）。

### 7.5 与后续论文的关系

Video-LLaVA 是课题阅读顺序中第一篇大视觉语言模型论文（第 3 组：Grounding DINO, SAM 2 → 第 4 组：Video-LLaVA, InternVideo2, Qwen2.5-VL）。后续的 InternVideo2 和 Qwen2.5-VL 可以视为 Video-LLaVA 的进一步发展，在视频理解能力上更强。

---

## 八、关键 Takeaways

1. **统一视觉表示是 LVLM 的关键设计原则**：不同视觉模态（图像、视频）应在投影到 LLM 之前对齐到同一特征空间。
2. **图像和视频联合训练互利**：不是零和博弈，而是 1+1>2 的关系。
3. **简单架构 + 好的预训练 > 复杂架构**：Video-LLaVA 用最简单的共享投影层就超越了所有复杂设计。
4. **LanguageBind 是跨模态对齐的有效工具**：通过传递性对齐（图像→语言←视频），实现图像和视频的间接对齐。
5. **仅 8 帧 + 1 epoch 即可达到强性能**：说明好的预训练大幅降低了下游任务的学习成本，这对课题中的高效部署有重要参考价值。

---

## 九、引用

```bibtex
@article{lin2023videollava,
  title={Video-LLaVA: Learning United Visual Representation by Alignment Before Projection},
  author={Lin, Bin and Ye, Yang and Zhu, Bin and Cui, Jiaxi and Ning, Munang and Jin, Peng and Yuan, Li},
  journal={arXiv preprint arXiv:2311.10122},
  year={2023}
}
```
