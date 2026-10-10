import type { JudgePersona, Section } from "@/types/presentation";

export const SECTION_ORDER: Section[] = [
  { key: "live", label: "实时演示", seconds: 35, accent: "#0B3954" },
  { key: "xai", label: "模型归因", seconds: 55, accent: "#087E8B" },
  { key: "compare", label: "模型对比", seconds: 40, accent: "#C08429" },
  { key: "arch", label: "技术架构", seconds: 35, accent: "#5C677D" },
  { key: "scene", label: "应用场景", seconds: 30, accent: "#3C6E71" },
];

export const SECTION_LOOKUP: Record<string, Section> = {};
for (const section of SECTION_ORDER) {
  SECTION_LOOKUP[section.key] = section;
}

export const JUDGE_BRIEF: Record<JudgePersona, string> = {
  "综合评委": "建议叙事顺序：问题背景 → 现场演示 → 数据支撑 → 扩展价值。",
  "技术评委": "建议叙事顺序：真实 XAI → 多模型对比 → 实时推理链路 → 架构边界。",
  "应用评委": "建议叙事顺序：使用场景 → 交互体验 → 稳定性指标 → 社会价值。",
};

export const QUOTE_LINES: string[] = [
  "ReliSign —— 可归因、可扩展、可落地的多模态手语翻译方案。",
  "展示层独立、推理层服务化、分析层可追踪，从原型到产品的关键架构决策。",
  "不只展示结果，更展示模型如何判断、在哪一帧最敏感、错误集中在何处。",
  "前后端分离保证最小改动下的现场稳定与后续扩展空间。",
];

export const DECK_SCRIPTS: Record<JudgePersona, Record<string, string[]>> = {
  "综合评委": {
    live: [
      '先展示系统入口和样例视频，立刻把评委注意力拉到“能不能用”。',
      "点出翻译结果、延迟和吞吐三个指标，快速建立可信度。",
      "如果评委追问真实性，再切换到真实推理模式。",
    ],
    xai: [
      "现在不是只看结果，而是看模型在什么时间、什么部位做出判断。",
      "用关键帧截图说明模型关注的动作证据，用混淆矩阵解释当前主要错误类型。",
    ],
    compare: [
      "这里用对比证明我们不是偶然跑得好，而是稳定优于 baseline。",
      "重点讲 BLEU、ROUGE 和 WER 的综合平衡。",
    ],
    arch: [
      '这一页专门回答“为什么这个系统后续还能继续做大”。',
      "展示层、推理层、分析层解耦，是工程可持续性的关键。",
    ],
    scene: [
      "最后落到真实场景，说明教育、医疗、智能家居三条应用线。",
      "把技术价值和社会价值合到一起收尾。",
    ],
  },
  "技术评委": {
    live: [
      "现场先给出端到端链路，证明系统不是离线拼图。",
      "再说明 video → pose/RGB → translation 的完整推理路径。",
    ],
    xai: [
      "这一页重点讲真实 attention、Grad-CAM 和字符级 confusion 都接了真实接口。",
      "关键帧、部位贡献和注意力矩阵一起构成归因分析闭环。",
    ],
    compare: [
      "这里建议直接讲 ablation，对比 pose-only、uncertainty gate 和完整模型。",
      "强调不是单一指标领先，而是多指标综合更稳。",
    ],
    arch: [
      "回答系统边界和模块职责，说明为什么这样拆可以最小改动。",
      "FastAPI 服务化后，React/WebSocket 只是展示层升级，不影响核心模型。",
    ],
    scene: [
      "用场景页承接边缘部署、多端适配和模型压缩。",
      "这是技术路线的延长线，不是另起炉灶。",
    ],
  },
  "应用评委": {
    live: [
      "先让评委看懂界面，再让评委看到结果。",
      "这一页的重点是低门槛、直观、稳定。",
    ],
    xai: [
      "归因分析不是炫技，而是帮助用户和评委理解系统为何可信。",
      "关键帧截图尤其适合应用向评委理解系统的判断依据。",
    ],
    compare: [
      '把复杂的训练实验收束成“效果更好、更稳、更能复现”。',
      "这页只讲最关键的对比结论，不陷入细节。",
    ],
    arch: [
      "架构页用于说明系统未来可以接更多终端和业务，不会推倒重来。",
      "前后端分离意味着产品化成本更低。",
    ],
    scene: [
      "教育、医疗、家居三页就是三条落地路线。",
      "最后一句落到无障碍沟通和社会价值。",
    ],
  },
};

export const DEFAULT_CHECKPOINT =
  "out/stage3_finetuning/best_checkpoint.pth";

export const STAGE_OPTIONS: string[] = [
  "Level 1 快速体验",
  "Level 2 核心功能",
  "Level 3 技术深度",
  "Level 4 创新扩展",
];

export const JUDGE_OPTIONS: JudgePersona[] = ["综合评委", "技术评委", "应用评委"];

export const SPLIT_OPTIONS: string[] = ["test", "dev"];
