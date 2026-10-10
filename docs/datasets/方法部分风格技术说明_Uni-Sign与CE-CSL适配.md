# 方法部分风格技术说明：Uni-Sign 与 CE-CSL 适配

本说明采用更接近论文 `Method` 部分的写法，对当前项目的技术路线进行结构化描述。  
目标不是介绍使用步骤，而是从方法设计角度说明：

1. 输入是什么
2. 表示如何构造
3. 模型如何建模
4. 输出如何生成
5. 为什么当前选择 `CE-CSL + pose-only + SLT` 作为第一阶段主线

---

## 1. 问题定义

我们考虑连续手语视频到自然语言文本的映射问题。  
给定一段手语视频序列：

\[
V = \{v_t\}_{t=1}^{T}
\]

其中 \(v_t\) 表示第 \(t\) 帧图像，\(T\) 表示视频长度。  
目标是学习一个条件生成模型，将输入视频映射到目标文本序列：

\[
Y = \{y_n\}_{n=1}^{N}
\]

其中 \(Y\) 可以表示自然语言句子，在 `SLT` 任务中对应中文文本输出。

在当前项目中，我们不直接对原始像素流进行端到端建模，而是优先采用姿态驱动的结构化表示。  
因此，原始视频首先被转换为关键点时序：

\[
P = \{p_t\}_{t=1}^{T}
\]

其中 \(p_t\) 由人体、左右手、面部等子区域关键点共同组成。

---

## 2. 输入表示构造

### 2.1 姿态序列提取

对输入视频 \(V\)，首先利用姿态估计器对每一帧提取关键点。  
当前项目使用的结构化关键点由以下几个部分组成：

- body
- left hand
- right hand
- face

在工程实现中，这些关键点被组织为多个分支张量，而不是直接扁平化为单一向量。  
这样做的动机在于：

1. 保留不同身体区域的结构语义
2. 降低不同局部区域之间的特征混淆
3. 提高对手部细粒度动作的建模能力

### 2.2 归一化与时序组织

关键点坐标经过基于图像宽高的归一化处理后，以帧序列形式保存到 `pose_format/*.pkl` 中。  
对于每个样本，最终可得到：

\[
P = \{P^{body}, P^{left}, P^{right}, P^{face}\}
\]

其中每个分支均为时间维上的序列表示。

### 2.3 监督信号

对于 `CE-CSL`，每个样本同时具有：

- 中文文本标注
- gloss 序列标注

因此样本可以统一表示为：

\[
\mathcal{D} = \{(P_i, G_i, Y_i)\}_{i=1}^{M}
\]

其中：

- \(P_i\) 为第 \(i\) 个样本的姿态序列
- \(G_i\) 为 gloss 序列
- \(Y_i\) 为自然语言句子

在当前阶段，我们重点研究 `SLT`，因此训练目标主要围绕 \(P_i \rightarrow Y_i\) 建立。

---

## 3. 数据适配与样本组织

### 3.1 数据集适配原则

`Uni-Sign` 原始框架主要围绕既有公开数据集构建。  
为了将 `CE-CSL` 纳入统一训练与推理流程，当前采用了“最小侵入式适配”策略：

1. 保持原有训练框架不变
2. 将 `CE-CSL` 标签转换为 Uni-Sign 可读格式
3. 保持视频路径、pose 路径与标签索引一致
4. 在配置层注册 `CE_CSL` 数据集分支

### 3.2 标签表示

`CE-CSL` 的原始标注来自 `label_cleaned/*.csv`，其中包含：

- `Number`
- `Translator`
- `Chinese Sentences`
- `Gloss_Clean`

转换后，每个样本被组织为统一字典结构，核心字段包括：

- `name`
- `text`
- `gloss`
- `video_path`

这样做的目的是将数据集层和模型层解耦，使后续训练器不必关心原始 CSV 的具体格式。

### 3.3 样本加载机制

在 [datasets.py](E:\project\xiaozhi\Uni-Sign\datasets.py) 中，`S2T_Dataset` 负责将标签、pose 和可选 RGB 数据组织成：

- `src_input`
- `tgt_input`

其中：

- `src_input` 表示源模态输入
- `tgt_input` 表示目标文本监督

这种组织方式本质上与标准的序列到序列建模范式一致。

---

## 4. 模型架构

### 4.1 总体结构

当前系统采用“手语编码器 + 文本生成器”的两阶段耦合结构。  
更具体地说，模型可以抽象为：

\[
Y = \mathcal{G}(\mathcal{E}(P))
\]

其中：

- \(\mathcal{E}(\cdot)\) 表示手语输入编码器
- \(\mathcal{G}(\cdot)\) 表示文本生成器

在当前实现中：

- \(\mathcal{E}\) 由 Uni-Sign 主体承担
- \(\mathcal{G}\) 由 `mT5` 承担

### 4.2 手语编码器

手语编码器的职责是将多分支姿态输入映射为统一的高层语义表示。  
从功能上看，这一模块需要解决三个问题：

1. 对局部关键点序列进行时序建模
2. 对不同身体区域信息进行融合
3. 将非语言输入转换为适合语言生成器消费的表征

当前项目同时支持：

- pose-only
- RGB + pose

两种设置。  
考虑到 `CE-CSL` 当前已经完成了 pose 链路的稳定接入，而 RGB 路线仍会引入更高的数据和工程复杂度，因此第一阶段采用 `pose-only` 更合理。

### 4.3 文本生成器

在得到手语编码器输出之后，项目使用 `mT5` 作为条件文本生成器。  
这意味着模型不是仅仅输出一个固定类别或 gloss 片段，而是通过生成式建模直接预测完整文本序列。

从技术上看，这一设计有两个重要意义：

1. 将手语翻译建模为条件序列生成问题
2. 充分利用大规模预训练语言模型的表达能力

因此，系统在方法范式上属于典型的多模态 encoder-decoder 架构。

---

## 5. 训练策略

### 5.1 当前推荐训练目标

结合当前数据适配状态与工程稳定性，第一阶段最推荐的训练配置为：

- dataset: `CE_CSL`
- task: `SLT`
- modality: `pose-only`

原因在于：

1. `CE-CSL` 具备清晰的文本监督
2. pose 数据已经能够稳定生成与读取
3. 该路径最适合先验证中文连续手语到文本的可学习性

### 5.2 微调而非从零训练

当前训练不是从随机初始化开始，而是从已有 Uni-Sign checkpoint 出发继续微调。  
这种策略可以表示为：

\[
\theta^{*} = \arg \min_{\theta} \sum_i \mathcal{L}(f_{\theta}(P_i), Y_i)
\]

其中模型参数 \(\theta\) 的初始值来自预训练或既有微调权重，而不是随机初始化。

这一策略的优点是：

1. 降低训练成本
2. 更快获得可用结果
3. 更适合中等规模领域数据集

### 5.3 训练入口

当前训练主流程由 [fine_tuning.py](E:\project\xiaozhi\Uni-Sign\fine_tuning.py) 控制。  
训练器负责：

- 构建训练集 / 验证集 / 测试集
- 初始化模型
- 加载 checkpoint
- 计算损失
- 进行评估
- 保存 `best_checkpoint.pth`

从工程实现上看，训练框架已具备任务通用性，`CE_CSL` 的接入主要体现在数据配置和数据读取层，而不需要完全重写训练器。

---

## 6. 推理机制

### 6.1 最小前向验证

为了验证 `CE-CSL` 是否已经被成功纳入当前框架，我们采用最小前向测试：

1. 读取一个 `CE-CSL` 样本
2. 加载 pose 输入
3. 加载 checkpoint 与 `mT5`
4. 调用模型生成文本

这种测试方式的意义在于：

- 将环境问题与方法问题分离
- 快速验证数据、模型、权重是否闭环
- 为后续微调提供前置可用性保证

### 6.2 演示型推理

除最小前向测试外，项目还提供视频级推理入口。  
其流程可以写为：

\[
V \rightarrow P \rightarrow \mathcal{E}(P) \rightarrow \mathcal{G}(\mathcal{E}(P)) \rightarrow \hat{Y}
\]

这一路径更适合比赛展示或原型演示。

---

## 7. 当前系统的技术优势与局限

### 7.1 优势

当前方案的主要优点包括：

1. 姿态驱动表示稳定，便于迁移到新数据集
2. 框架本身支持统一任务建模
3. 通过 `mT5` 引入生成式语言能力
4. 已具备接入中文连续手语数据集的工程基础

### 7.2 当前局限

当前最主要的局限并不在于模型无法运行，而在于：

1. 现有权重并非为 `CE-CSL` 专门训练
2. 因而当前预测存在明显域偏差
3. `RGB + pose` 路线尚未作为 `CE-CSL` 的第一阶段主线 fully 验证

也就是说，当前系统的下一步关键任务不是“证明能不能跑”，而是“提高对 `CE-CSL` 的任务适配性”。

---

## 8. 为什么当前优先选择 CE-CSL pose-only SLT

这个选择不是随意的，而是基于方法和工程稳定性共同决定的。

### 8.1 从任务角度

`CE-CSL` 具有文本监督，因此最自然的目标就是 `SLT`。  
相较于单纯 `CSLR`，`SLT` 更能体现连续手语到语言理解的最终价值。

### 8.2 从表示角度

当前 pose 路线已经完成：

- 标签转换
- pose 生成
- 数据加载
- 最小前向

因此，pose-only 已经是闭环最完整、最稳定的路径。

### 8.3 从实验推进角度

如果一开始就采用 `RGB + pose`，会同时增加：

- 数据准备复杂度
- 显存压力
- 调参难度
- 排错难度

因此第一阶段优先选择：

- `CE_CSL + pose-only + SLT`

是更符合研究实践的路线。

---

## 9. 面向后续论文写作的技术表述

如果将当前方法浓缩为论文风格的概括，可以写成：

**We formulate Chinese continuous sign language translation as a pose-driven sequence-to-text problem. Given an input sign video, we first extract structured human pose sequences including body, hand, and facial keypoints. These multi-branch temporal representations are then encoded by the Uni-Sign backbone and further coupled with an mT5-based text generator to produce natural language sentences. To adapt the framework to CE-CSL, we introduce a lightweight dataset adaptation pipeline including label conversion, pose generation, and dataset registration, enabling direct fine-tuning on Chinese sign language data.**

对应中文可以写成：

**我们将中文连续手语翻译任务建模为一种基于姿态驱动的序列到文本生成问题。给定输入手语视频，首先提取包含身体、左右手和面部关键点的结构化姿态时序表示，然后由 Uni-Sign 主干网络进行手语特征编码，并进一步与基于 mT5 的文本生成器耦合，从而输出自然语言句子。为适配 CE-CSL 数据集，我们设计了轻量化的数据集接入流程，包括标签转换、姿态生成与数据集注册，使现有统一框架能够直接在中文连续手语数据上进行微调。**

---

## 10. 当前最准确的技术结论

结合你现在的项目状态，可以给出如下技术结论：

1. 当前系统已经完成 `CE-CSL` 在 Uni-Sign 框架中的基本接入
2. 当前最小前向闭环已经打通
3. 当前方法的主表示为姿态序列，而非纯像素端到端输入
4. 当前模型范式本质上是多模态编码与生成式语言解码耦合
5. 当前性能瓶颈主要来自域偏差与缺乏 `CE-CSL` 专属微调，而不是框架结构错误
6. 因此，下一步最合理的方法路线是围绕 `pose-only + SLT` 开展 `CE-CSL` 微调与评估
