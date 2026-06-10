# Grounding DINO 论文读书笔记

**论文标题：** Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection

**来源：** ECCV 2024 / arXiv:2303.05499

**作者：** Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, Lei Zhang (清华大学, IDEA, HKUST, CUHK-Shenzhen, Microsoft)

**论文地址：** https://arxiv.org/abs/2303.05499

**代码地址：** https://github.com/IDEA-Research/GroundingDINO

---

## 一、论文解决什么问题

传统目标检测模型只能识别预定义的固定类别（如 COCO 的 80 类），属于"闭集检测"。但实际应用中，用户可能需要检测任意类型的物体（如"打印机"、"咖啡杯"、"红色背包"），这些物体不在训练类别中。

开放集目标检测（Open-Set Object Detection）的目标是：给定一张图像和一段文本（类别名称或描述性表达），检测出图像中由文本指定的任意物体。这需要模型同时具备**物体定位能力**和**文本理解能力**。

现有方法（如 GLIP、OV-DETR）虽然已尝试引入语言信息，但仅在检测器的**部分阶段**进行模态融合（如 GLIP 仅在 Neck 阶段融合，OV-DETR 仅在 Query 初始化阶段融合），导致语言泛化能力受限。Grounding DINO 的核心主张是：**越紧密的跨模态融合，越强的开放集能力**。

---

## 二、核心设计理念

### 2.1 闭集检测器的三阶段模块化视角

论文将闭集目标检测器抽象为三个核心模块：

```
Backbone (特征提取)  →  Neck (特征增强)  →  Head (区域预测)
```

对应地，语言信息可以在三个阶段注入：

| 阶段 | 位置 | 代表方法 |
|------|------|---------|
| Phase A | Neck（特征增强） | GLIP：在 Neck 做早期融合 |
| Phase B | Query 初始化 | OV-DETR：用语言感知的 query |
| Phase C | Head（解码器） | 在 Head 中做跨模态交互 |

论文的关键洞察是：**三个阶段都做融合，效果优于只在部分阶段融合**。而 Transformer-based 检测器 DINO 的逐层设计天然适合与语言模块交互，使得三阶段紧密融合成为可能。

### 2.2 基于 DINO 的架构选择

选择 DINO（而非 Faster R-CNN 等经典检测器）作为基座检测器的原因：

1. DINO 基于 Transformer，其 self-attention 和 cross-attention 机制天然可与语言特征交互。
2. DETR 类检测器的 query 机制便于注入语言引导。
3. DINO 在 COCO 闭集检测上已是最强（62.5 AP），基座够强。

---

## 三、Grounding DINO 架构详解

### 3.1 整体框架

```
输入图像 → Image Backbone (Swin-T/L) → 多尺度图像特征
输入文本 → Text Backbone (BERT-base)  → 文本特征
                    ↓
        ┌─────────────────────┐
        │  Feature Enhancer   │  ← Phase A: Neck 融合
        │  (6 层, 每层含:)     │
        │  - 图像自注意力      │
        │  - 文本自注意力      │
        │  - 图→文交叉注意力   │
        │  - 文→图交叉注意力   │
        └─────────────────────┘
                    ↓
        ┌─────────────────────┐
        │  Language-Guided    │  ← Phase B: Query 初始化
        │  Query Selection    │
        │  (选 top-900 特征)   │
        └─────────────────────┘
                    ↓
        ┌─────────────────────┐
        │  Cross-Modality     │  ← Phase C: Head 解码
        │  Decoder            │
        │  (6 层, 每层含:)     │
        │  - Query 自注意力    │
        │  - 图像交叉注意力    │
        │  - 文本交叉注意力    │  ← 比 DINO 多出的模块
        │  - FFN              │
        └─────────────────────┘
                    ↓
            输出: 物体 bbox + 对应文本短语
```

### 3.2 Feature Enhancer（Phase A）

Feature Enhancer 是 Neck 模块，负责在编码阶段就进行图像和文本的特征融合。每层包含：

1. **Deformable Self-Attention**：增强图像特征（使用可变形注意力降低计算量）。
2. **Vanilla Self-Attention**：增强文本特征。
3. **Image-to-Text Cross-Attention**：文本作为 Query，图像作为 Key/Value，让文本"看到"图像。
4. **Text-to-Image Cross-Attention**：图像作为 Query，文本作为 Key/Value，让图像"理解"文本。

双向交叉注意力使两种模态从编码阶段就开始对齐，这是 GLIP 的核心思想，Grounding DINO 在此基础上增加了更多阶段的融合。

### 3.3 Language-Guided Query Selection（Phase B）

传统 DETR 使用可学习的 object query 或从编码器输出中选择 query。Grounding DINO 设计了语言引导的 query 选择：

1. 计算图像特征 Xᵢ ∈ R^(Nᵢ×d) 和文本特征 Xₜ ∈ R^(Nₜ×d) 的相似度矩阵。
2. 对每个图像 token，取其与所有文本 token 的最大相似度。
3. 选择相似度最高的 Nq=900 个图像 token 作为 decoder query。

公式：I_Nq = Top_Nq(Max_(−1)(Xᵢ · Xₜᵀ))

这确保了被选中的 query 与输入文本高度相关，而非盲目地选择响应最强的区域。

### 3.4 Cross-Modality Decoder（Phase C）

解码器每层比 DINO 多了一个**文本交叉注意力层**：

```
每个 Decoder Layer:
  Query 自注意力 → 图像交叉注意力 → 文本交叉注意力 → FFN
                                    ↑
                             DINO 没有这一层
```

文本交叉注意力让 query 在每层细化过程中持续参考文本信息，从而在最终输出时能准确地将检测框与文本概念对齐。

### 3.5 Sub-Sentence Level 文本特征

这是论文的一个重要技术贡献。现有方法有两种文本表示方式：

| 方式 | 做法 | 缺点 |
|------|------|------|
| Sentence level | 整句编码为一个特征 | 丢失细粒度信息 |
| Word level | 将所有类别名拼接后编码 | 不相关类别之间会互相干扰 |

Grounding DINO 提出 **Sub-Sentence Level**：在拼接的类别名称之间加入 attention mask，阻止不相关类别之间的注意力交互。

```
输入文本: "cat . baseball glove . A cat is sleeping on a table ."

Word level (无 mask):
  cat ←attention→ baseball ←attention→ glove ←attention→ ...
  (不相关类别互相干扰)

Sub-Sentence level (有 mask):
  [cat] | [baseball glove] | [A cat is sleeping on a table]
  子句内部可以 attention，子句之间被 mask 阻断
```

这样既保留了每个词的细粒度特征，又避免了无关类别之间的干扰。

### 3.6 损失函数

- **定位损失**：L1 loss + GIoU loss
- **分类损失**：对比损失（query 与文本 token 的点积 → focal loss）
- **匹配**：匈牙利匹配（分类代价权重 1.0，L1 权重 5.0，GIoU 权重 2.0；对应损失权重为 2.0、5.0、2.0）
- 每层解码器输出和编码器输出后都加辅助损失

---

## 四、预训练策略

Grounding DINO 在三类数据上进行大规模预训练：

### 4.1 检测数据

将检测任务转化为短语定位（phrase grounding）任务：将所有类别名称拼接成文本输入。使用 COCO、Objects365 (O365)、OpenImages (OI)。训练时随机采样类别名称的子集和顺序，增加多样性。

### 4.2 Grounding 数据

使用 GoldG（Flickr30k Entities + Visual Genome）和 RefC（RefCOCO/+/g），这些数据天然包含图像区域与文本短语的对齐标注。

### 4.3 Caption 数据

使用 GLIP 生成的伪标注 caption 数据（Cap4M），利用丰富的图像描述来扩展模型对新概念的泛化能力。

---

## 五、实验与结果

### 5.1 COCO 零样本检测

| 模型 | 骨干 | 预训练数据 | Zero-Shot AP | Fine-Tune AP |
|------|------|-----------|-------------|-------------|
| DyHead-T | Swin-T | - | - | 49.7 |
| DINO (Swin-L) | Swin-L | O365 | 46.2 (闭集→映射) | 62.5 |
| GLIP-L | Swin-L | FourODs, GoldG, Cap24M | 49.8 | 60.8/61.0 |
| **Grounding DINO L** | **Swin-L** | **O365, OI, GoldG** | **52.5** | **62.6/62.7** |

Grounding DINO 在不使用任何 COCO 训练数据的情况下取得 **52.5 AP**，比 GLIP-L 高 2.7 AP，比 DINO 闭集映射高 4.3 AP。

### 5.2 ODinW 零样本检测（35 个数据集）

| 模型 | 骨干 | 参数量 | AP | AP_median |
|------|------|--------|-----|---------|
| MDETR | ENB5 | 169M | 10.7 | 3.0 |
| OWL-ViT | ViT-L/14 | >1243M | 18.8 | 9.8 |
| GLIP-T | Swin-T | 232M | 19.6 | 5.1 |
| DetCLIP | Swin-L | 267M | 24.9 | 18.3 |
| Florence | CoSwinH | ≈841M | 25.8 | 14.3 |
| **Grounding DINO L** | **Swin-L** | **341M** | **26.1** | **18.4** |

Grounding DINO 以 341M 参数（远小于 Florence 的 841M）在 ODinW 上创下新纪录 **26.1 AP**，且 AP_median 最高（18.4），说明在不同数据集上表现更一致。

### 5.3 LVIS 长尾物体检测

| 模型 | 预训练数据 | AP | AP_r / AP_c / AP_f |
|------|-----------|-----|---------------------|
| GLIP-T (C) | O365, GoldG | 24.9 | 17.7/19.5/31.0 |
| DetCLIPv2 | O365, GoldG, CC15M | 40.4 | 36.0/41.7/40.0 |
| **Grounding DINO L** | **多数据源** | **33.9** | **22.2/30.7/38.8** |

Grounding DINO L 总体 AP 高于 GLIP-T 但低于 DetCLIPv2（后者训练数据规模更大）。在同骨干（Swin-T）对比下，Grounding DINO 在稀有类别上弱于 GLIP（AP_r 14.4 vs 17.7），这是 DETR 类架构的共性局限。微调后 Grounding DINO-T 以 52.1 AP 超越 DetCLIPv2-T 的 50.7 AP，显示了良好的可扩展性。

### 5.4 Referring Expression Comprehension (REC)

| 模型 | 预训练数据 | RefCOCO val | RefCOCO+ val | RefCOCOg val |
|------|-----------|------------|-------------|-------------|
| TransVG | 无 | 81.02 | 64.82 | 68.67 |
| MDETR | GoldG, RefC | 86.75 | 79.52 | 81.64 |
| GLIP-T | O365, GoldG, Cap4M | 50.42 | 49.50 | 66.09 |
| **Grounding DINO T** | **O365, GoldG** | **50.41** | **51.40** | **67.46** |
| **Grounding DINO T** | **O365, GoldG, RefC + 微调** | **89.19** | **81.09** | **84.15** |

零样本时 Grounding DINO 优于 GLIP，加入 RefC 数据微调后大幅超越所有方法。

### 5.5 消融实验（O365 预训练，Swin-T 骨干）

| 模型变体 | COCO Zero-Shot | COCO Fine-Tune | LVIS Zero-Shot |
|---------|---------------|----------------|----------------|
| 完整模型 | 46.7 | 56.9 | 16.1 |
| w/o encoder fusion | 45.8 (-0.9) | 56.1 (-0.8) | 13.1 (-3.0) |
| 静态 query (非语言引导) | 46.3 (-0.4) | 56.6 (-0.3) | 13.6 (-2.5) |
| w/o text cross-attention | 46.1 (-0.6) | 56.3 (-0.6) | 14.3 (-1.8) |
| word-level (非 sub-sentence) | 46.4 (-0.3) | 56.6 (-0.3) | 15.6 (-0.5) |

Encoder fusion（Phase A）贡献最大，language-guided query selection（Phase B）和 text cross-attention（Phase C）次之，sub-sentence level 文本特征也有稳定贡献。**四个模块缺一不可**，验证了三阶段紧密融合的设计理念。

---

## 六、关键洞见总结

1. **紧密融合优于松散融合**：在 Neck、Query 初始化和 Decoder 三个阶段都进行跨模态交互，效果显著优于仅在单一阶段融合。消融实验清楚地展示了每个阶段的独立贡献。
2. **Transformer 架构是跨模态融合的天然平台**：self-attention 和 cross-attention 的统一范式使得图像和文本可以在任意层级交互，这是传统 CNN 检测器难以做到的。
3. **Sub-Sentence Level 文本表示的巧思**：通过 attention mask 阻断不相关类别之间的交互，是一个简单但有效的设计，在不增加计算量的情况下提升了性能。
4. **零样本能力来自大规模多样化预训练**：检测数据 + grounding 数据 + caption 数据的组合使模型具备了对新概念的强大泛化能力。
5. **DETR 类架构的长尾局限**：在 LVIS 等长尾分布数据集上，DETR 类方法在稀有类别上的表现仍有待提升，这是未来需要解决的问题。

---

## 七、与本课题的关联

Grounding DINO 与本课题中"开放词汇目标检测"模块（课题方案 §4.2）直接相关，为检测工作场景中可能出现的任意物体提供了关键技术支撑。

### 7.1 开放词汇检测的必要性

课题的目标场景是办公/生活环境中的人员行为分析。工作场景中可能出现的物体远超预定义类别列表：不仅有常见物体（椅子、桌子、电脑），还有特定物体（打印机、咖啡机、白板、文件夹等）。传统闭集检测器（如 YOLO 训练在 COCO 80 类上）无法覆盖所有可能出现的物体，而 Grounding DINO 可以通过文本输入灵活指定检测目标。

### 7.2 在课题技术路线中的位置

```
视频帧 → 两路检测:
  ├── YOLO (闭集, 快速) → 检测常见物体和人
  └── Grounding DINO (开放集, 灵活) → 检测特定/罕见物体
          ↓
  ByteTrack 跟踪 → 关联检测结果到轨迹
          ↓
  SlowFast + ST-HOI → 动作识别 + 人-物交互检测
```

### 7.3 与 ST-HOI 的互补

ST-HOI 关注"人与物体之间的交互关系"，但其交互类别受限于 VidHOI 的 50 个预定义谓词。Grounding DINO 则可以检测任意物体，为 ST-HOI 提供更丰富的物体候选。例如，当 ST-HOI 检测到"人-使用-?"时，Grounding DINO 可以识别出具体的物体（如"打印机"、"微波炉"），从而生成更精确的三元组。

### 7.4 与行为时间线的结合

课题的最终输出是结构化行为时间线：

```json
{
  "person_id": "P001",
  "action": "使用打印机",
  "interacting_object": "打印机",
  "start_time": "10:32:15",
  "end_time": "10:32:45",
  "confidence": 0.87
}
```

其中"interacting_object"字段需要开放词汇的物体检测能力——这正是 Grounding DINO 的价值所在。

### 7.5 Referring Expression 的应用

Grounding DINO 不仅支持类别名检测（如"printer"），还支持描述性表达检测（如"the red cup on the left side of the table"）。这在课题中可用于更精细的场景理解，例如区分"左边的人"和"右边的人"。

### 7.6 部署建议

- 使用 Grounding DINO T（Swin-T, 172M 参数）作为默认模型，在精度和速度之间取得平衡。
- 对于需要检测的特定物体，维护一个动态更新的文本提示列表（如"printer, coffee machine, whiteboard, ..."）。
- Grounding DINO 与 YOLO 并行运行：YOLO 负责高速检测常见物体，Grounding DINO 负责检测长尾物体。

---

## 八、关键 Takeaways

1. **三阶段融合是关键**：Neck（特征增强）+ Query Selection（查询初始化）+ Decoder（解码）三个阶段都融入语言信息，实现了最强的开放集检测能力。
2. **52.5 AP 零样本 COCO 的里程碑意义**：不需要任何 COCO 训练数据就达到 52.5 AP，说明大规模预训练 + 紧密融合可以实现强大的零样本泛化。
3. **模型紧凑高效**：Grounding DINO T 仅 172M 参数（远小于 Florence 的 841M），但在 ODinW 上超越所有竞品，说明架构设计比单纯的模型规模更重要。
4. **Sub-Sentence Level 是简单而有效的设计**：通过 attention mask 避免不相关类别的干扰，在不增加计算量的情况下提升性能。
5. **REC 能力扩展了应用范围**：不仅能检测类别，还能理解描述性表达（如"the person wearing a red hat"），为更精细的场景理解提供了可能。

---

## 九、引用

```bibtex
@article{liu2023grounding,
  title={Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection},
  author={Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and Li, Feng and Zhang, Hao and Yang, Jie and Jiang, Qing and Li, Chunyuan and Yang, Jianwei and Su, Hang and Zhu, Jun and Zhang, Lei},
  journal={ECCV},
  year={2024}
}
```
