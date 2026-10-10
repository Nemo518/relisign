# PPT风格：彩色大模块中文版框架图

这张图更适合比赛答辩或项目汇报中的“大模块展示页”，强调系统分层、业务逻辑和落地方向。

```mermaid
flowchart TB
    A["1. 数据输入层\n手语视频\nCE-CSL 数据集\n中文标注 / Gloss 标注"]:::layer1
    B["2. 预处理层\n视频读取\n姿态提取\n关键点归一化\n标签转换"]:::layer2
    C["3. 数据组织层\nlabels.train/dev/test\npose_format\nS2T_Dataset\nbatch 构造"]:::layer3
    D["4. 模型理解层\nUni-Sign 编码器\n时序建模\n多分支融合\n可选 RGB + Pose"]:::layer4
    E["5. 语言生成层\nmT5 解码器\n手语到中文文本生成"]:::layer5
    F["6. 应用输出层\n中文翻译结果\nGloss 结果\n比赛演示\n系统推理"]:::layer6

    A --> B --> C --> D --> E --> F

    G["训练主线\nfine_tuning.py\n加载 checkpoint\nCE-CSL 微调"]:::side
    H["推理主线\nonline_inference.py\ntest_cecsl_forward.py"]:::side
    I["当前推荐路线\nCE-CSL + pose-only + SLT"]:::focus

    C --> G
    D --> G
    E --> H
    D --> H
    I --> G
    I --> H

    classDef layer1 fill:#dff2ff,stroke:#2f80c1,stroke-width:2,color:#16324a;
    classDef layer2 fill:#e8f8e6,stroke:#4c9a57,stroke-width:2,color:#204428;
    classDef layer3 fill:#fff7df,stroke:#c79b2c,stroke-width:2,color:#5b4610;
    classDef layer4 fill:#ffe8ea,stroke:#d35b74,stroke-width:2,color:#5b1f2a;
    classDef layer5 fill:#efe7ff,stroke:#7b61d1,stroke-width:2,color:#32225f;
    classDef layer6 fill:#eafaf2,stroke:#50a36d,stroke-width:2,color:#1e4b2a;
    classDef side fill:#f5f5f5,stroke:#888,stroke-width:1.6,color:#333;
    classDef focus fill:#fff0f0,stroke:#e54848,stroke-width:2.5,color:#7a1e1e;
```

## 使用建议

这张图适合放在 PPT 的“技术路线”或“系统架构”页，讲解时可以按下面顺序说：

1. 最上层是数据输入，包括手语视频和 `CE-CSL` 中文标注
2. 第二层先做姿态提取和标签转换
3. 第三层把数据整理成 Uni-Sign 可以直接读取的格式
4. 第四层由 Uni-Sign 完成手语时序理解和多分支融合
5. 第五层由 `mT5` 把手语特征翻译成中文文本
6. 最下层输出比赛演示结果、中文翻译结果和后续应用接口

当前你们项目最适合强调的点是：

- 不只是做“识别”
- 而是在做“连续手语理解与文本翻译”
- 并且已经把中文连续手语数据 `CE-CSL` 接入到了统一框架中
