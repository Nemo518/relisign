import {
  Cpu,
  GitBranch,
  Layers,
  Lightbulb,
  Zap,
  Eye,
  TrendingUp,
  ShieldCheck,
  Box,
  ArrowRight,
  Sparkles,
} from "lucide-react";
import { cn } from "@/lib/utils";

const pipelineSteps = [
  {
    step: 1,
    title: "视频采集",
    icon: <Eye size={18} />,
    detail: "样例库 / 上传 / 摄像头三种输入源，兼容多分辨率视频流。",
    innovation: "无需外部硬件",
    level: "基础",
    levelColor: "text-slate-400 bg-slate-400/10 border-slate-400/20",
  },
  {
    step: 2,
    title: "姿态提取",
    icon: <Box size={18} />,
    detail: "133 个关键点提取，含手部精细建模与置信度过滤。",
    innovation: "手部精细建模",
    level: "中等",
    levelColor: "text-cyan-400 bg-cyan-400/10 border-cyan-400/20",
  },
  {
    step: 3,
    title: "双流编码",
    icon: <Layers size={18} />,
    detail: "RGB 语义 + Pose 时空双流并行编码，自适应融合。",
    innovation: "多模态融合",
    level: "核心难点",
    levelColor: "text-red-400 bg-red-400/10 border-red-400/20",
  },
  {
    step: 4,
    title: "时序建模",
    icon: <TrendingUp size={18} />,
    detail: "Transformer 捕获长程帧间依赖，建模手语动作时序上下文。",
    innovation: "自注意力上下文",
    level: "核心难点",
    levelColor: "text-red-400 bg-red-400/10 border-red-400/20",
  },
  {
    step: 5,
    title: "文本生成",
    icon: <Cpu size={18} />,
    detail: "自回归解码器逐词生成中文手语翻译，mT5 语义增强。",
    innovation: "mT5 语义增强",
    level: "中等",
    levelColor: "text-cyan-400 bg-cyan-400/10 border-cyan-400/20",
  },
  {
    step: 6,
    title: "归因分析",
    icon: <Lightbulb size={18} />,
    detail: "Attention + Grad-CAM + 混淆矩阵，模型决策可追溯。",
    innovation: "模型归因",
    level: "创新亮点",
    levelColor: "text-purple-400 bg-purple-400/10 border-purple-400/20",
  },
];

const innovations = [
  {
    title: "多模态融合架构",
    description:
      "RGB 视觉 + Pose 骨骼双流并行编码，自适应融合模块实现互补增强。",
    tags: ["RGB 流", "Pose 流", "自适应融合", "消融验证"],
    color: "from-cyan-400 to-teal-500",
    glow: "shadow-cyan-400/20",
  },
  {
    title: "模型归因分析",
    description:
      "Attention 可视化 + Grad-CAM 热力图 + 字符级混淆矩阵，模型决策全程可追溯。",
    tags: ["Attention", "Grad-CAM", "混淆矩阵", "关键帧定位"],
    color: "from-violet-400 to-purple-500",
    glow: "shadow-purple-400/20",
  },
  {
    title: "端到端在线推理",
    description:
      "标准 ~500ms / 轻量 ~200ms / 边缘 ~100ms 三级性能档位，延迟与准确率灵活平衡。",
    tags: ["实时推理", "三级性能", "30 FPS", "摄像头输入"],
    color: "from-emerald-400 to-teal-500",
    glow: "shadow-emerald-400/20",
  },
  {
    title: "三层解耦架构",
    description:
      "展示层 React · 推理层 FastAPI+PyTorch · 分析层 XAI，HTTP API 通信，独立升级扩展。",
    tags: ["前后端分离", "服务化", "可扩展", "ONNX Ready"],
    color: "from-amber-400 to-orange-500",
    glow: "shadow-amber-400/20",
  },
];

const difficultyMatrix = [
  { module: "视频流实时处理", difficulty: 4, desc: "多分辨率适配，摄像头/上传/样例三源统一" },
  { module: "姿态关键点提取", difficulty: 3, desc: "133 点全身+手部，置信度过滤，运动补偿" },
  { module: "双流特征融合", difficulty: 5, desc: "RGB 语义 + Pose 时空，消融实验证明互补性" },
  { module: "Transformer 时序建模", difficulty: 4, desc: "长程依赖捕获，自注意力跨帧建模" },
  { module: "文本自回归解码", difficulty: 4, desc: "逐词生成 + mT5 语义增强 + Beam Search" },
  { module: "模型归因分析", difficulty: 5, desc: "Attention 可视化 + Grad-CAM + 字符级混淆" },
  { module: "多模型对比评估", difficulty: 3, desc: "BLEU-4 / ROUGE-L / WER / Similarity 四维" },
  { module: "三层架构解耦", difficulty: 3, desc: "展示/推理/分析层独立，API 通信，可单独升级" },
];

function DifficultyBar({ level }: { level: number }) {
  return (
    <div className="flex flex-1 gap-1">
      {[1, 2, 3, 4, 5].map((lv) => (
        <div
          key={lv}
          className={cn(
            "h-5 flex-1 rounded transition-all duration-500",
            lv <= level
              ? lv >= 5
                ? "bg-gradient-to-r from-red-500 to-red-400 shadow-sm shadow-red-400/30"
                : lv === 4
                  ? "bg-gradient-to-r from-orange-500 to-amber-400 shadow-sm shadow-amber-400/30"
                  : "bg-gradient-to-r from-cyan-500 to-teal-400 shadow-sm shadow-cyan-400/30"
              : "bg-slate-700/50",
          )}
        />
      ))}
    </div>
  );
}

export function ArchitecturePage() {
  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <h2 className="text-xl font-bold">技术架构与创新难点</h2>
        <span className="tech-badge rounded-md bg-purple-400/10 border border-purple-400/20 px-2.5 py-1 text-[10px] text-purple-400">
          TECHNICAL DEPTH
        </span>
      </div>

      {/* Pipeline */}
      <div className="glass-card scan-lines rounded-3xl p-6">
        <h3 className="mb-5 text-base font-bold flex items-center gap-2">
          <GitBranch size={18} className="text-cyan-400" />
          <span>端到端推理流水线</span>
          <span className="tech-badge ml-auto rounded-md bg-cyan-400/10 border border-cyan-400/20 px-2 py-0.5 text-[10px] text-cyan-400">
            6-STAGE PIPELINE
          </span>
        </h3>
        <div className="grid grid-cols-6 gap-3">
          {pipelineSteps.map((step) => (
            <div key={step.step} className="text-center group">
              <div className="relative mx-auto mb-3">
                <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-cyan-500 to-teal-500 text-white shadow-lg shadow-cyan-500/20 group-hover:shadow-cyan-500/40 transition-shadow mx-auto">
                  {step.icon}
                </div>
                <div className="absolute -top-1 -right-1 flex h-5 w-5 items-center justify-center rounded-full bg-slate-800 border border-cyan-400/30 text-[10px] font-bold text-cyan-400">
                  {step.step}
                </div>
              </div>
              <div className="mb-1.5 text-sm font-bold">{step.title}</div>
              <div
                className={cn(
                  "mb-1.5 inline-block rounded-full border px-2 py-0.5 text-[10px] font-semibold",
                  step.levelColor,
                )}
              >
                {step.level}
              </div>
              <p className="text-[11px] leading-relaxed text-slate-400">
                {step.detail}
              </p>
              <p className="mt-1.5 text-[10px] font-semibold text-cyan-400">
                ✓ {step.innovation}
              </p>
            </div>
          ))}
        </div>
        {/* Flow arrows */}
        <div className="mt-4 flex items-center justify-center gap-1">
          {[0, 1, 2, 3, 4].map((i) => (
            <div key={i} className="flex items-center gap-1">
              <div className="h-px w-10 rounded bg-gradient-to-r from-cyan-400/50 to-teal-400/50" />
              <ArrowRight size={12} className="text-cyan-400/60" />
            </div>
          ))}
        </div>
      </div>

      {/* Innovation Cards */}
      <h3 className="text-lg font-bold flex items-center gap-2">
        <Sparkles size={20} className="text-purple-400" />
        <span>四大技术创新</span>
        <span className="tech-badge rounded-md bg-purple-400/10 border border-purple-400/20 px-2 py-0.5 text-[10px] text-purple-400">
          CORE INNOVATIONS
        </span>
      </h3>
      <div className="grid grid-cols-2 gap-5">
        {innovations.map((item) => (
          <div
            key={item.title}
            className="glass-card group rounded-3xl p-6 transition-all duration-300 hover:-translate-y-1 hover:border-cyan-400/30"
          >
            <div
              className={`mb-4 inline-flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br ${item.color} text-white shadow-lg ${item.glow}`}
            >
              <Lightbulb size={22} />
            </div>
            <h4 className="mb-2 text-base font-bold">{item.title}</h4>
            <p className="mb-3 text-sm leading-relaxed text-slate-400">
              {item.description}
            </p>
            <div className="flex flex-wrap gap-1.5">
              {item.tags.map((tag) => (
                <span
                  key={tag}
                  className="tech-badge rounded-lg bg-slate-700/50 px-2.5 py-1 text-[10px] text-slate-400 group-hover:text-slate-300 transition-colors"
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Difficulty Matrix */}
      <div className="glass-card rounded-3xl p-6">
        <h3 className="mb-4 text-base font-bold flex items-center gap-2">
          <ShieldCheck size={18} className="text-red-400" />
          <span>技术难度矩阵</span>
          <span className="tech-badge rounded-md bg-red-400/10 border border-red-400/20 px-2 py-0.5 text-[10px] text-red-400">
            Lv1–Lv5
          </span>
        </h3>
        <div className="space-y-3">
          {difficultyMatrix.map((item) => (
            <div key={item.module} className="flex items-center gap-4 group">
              <div className="w-36 shrink-0 text-sm font-medium text-slate-300">
                {item.module}
              </div>
              <DifficultyBar level={item.difficulty} />
              <div className="w-10 text-center text-sm font-bold text-slate-200 font-mono">
                Lv{item.difficulty}
              </div>
              <div className="hidden w-72 text-xs text-slate-400 group-hover:text-slate-400 transition-colors lg:block">
                {item.desc}
              </div>
            </div>
          ))}
        </div>
        <div className="mt-4 flex items-center gap-5 text-[11px] text-slate-400">
          <span className="flex items-center gap-1.5">
            <div className="h-3 w-3 rounded bg-gradient-to-r from-cyan-500 to-teal-400 shadow-sm shadow-cyan-400/20" />
            基础/中等
          </span>
          <span className="flex items-center gap-1.5">
            <div className="h-3 w-3 rounded bg-gradient-to-r from-orange-500 to-amber-400 shadow-sm shadow-amber-400/20" />
            较难 (Lv4)
          </span>
          <span className="flex items-center gap-1.5">
            <div className="h-3 w-3 rounded bg-gradient-to-r from-red-500 to-red-400 shadow-sm shadow-red-400/20" />
            核心难点 (Lv5)
          </span>
        </div>
      </div>

      {/* Evolution Roadmap */}
      <div className="glass-card rounded-3xl p-5">
        <h3 className="mb-4 text-base font-bold">系统演进路线</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-700/50 text-slate-400">
                <th className="py-2.5 text-left text-xs font-semibold">阶段</th>
                <th className="py-2.5 text-left text-xs font-semibold">展示层</th>
                <th className="py-2.5 text-left text-xs font-semibold">推理层</th>
                <th className="py-2.5 text-left text-xs font-semibold">分析层</th>
                <th className="py-2.5 text-left text-xs font-semibold">部署</th>
              </tr>
            </thead>
            <tbody>
              {[
                {
                  phase: "当前展示版",
                  presentation: "React + TypeScript",
                  inference: "PyTorch 在线推理",
                  analysis: "实时 XAI 计算",
                  delivery: "本地 HTTP 8000",
                  active: true,
                },
                {
                  phase: "下一阶段",
                  presentation: "React + WebSocket",
                  inference: "ONNX Runtime 加速",
                  analysis: "在线学习分析",
                  delivery: "Docker 容器部署",
                  active: false,
                },
                {
                  phase: "增强阶段",
                  presentation: "移动端适配",
                  inference: "边缘端 NPU",
                  analysis: "联邦学习分析",
                  delivery: "云端 + 边缘协同",
                  active: false,
                },
              ].map((p, i) => (
                <tr
                  key={i}
                  className={cn(
                    "border-b border-slate-700/30 transition-colors",
                    p.active
                      ? "bg-cyan-400/5"
                      : "text-slate-400",
                  )}
                >
                  <td className="py-2.5 font-semibold">
                    <div className="flex items-center gap-2">
                      {p.active && <div className="pulse-dot" />}
                      {p.phase}
                    </div>
                  </td>
                  <td className="py-2.5">{p.presentation}</td>
                  <td className="py-2.5">{p.inference}</td>
                  <td className="py-2.5">{p.analysis}</td>
                  <td className="py-2.5">
                    <span className={cn(
                      "tech-badge rounded-md px-2 py-0.5 text-[10px]",
                      p.active
                        ? "bg-cyan-400/10 border border-cyan-400/20 text-cyan-400"
                        : "bg-slate-700/30 text-slate-400",
                    )}>
                      {p.delivery}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Architecture Principle */}
      <div className="holo-card rounded-2xl p-5 border-cyan-400/10">
        <div className="font-bold flex items-center gap-2">
          <Sparkles size={16} className="text-cyan-400" />
          架构设计原则
        </div>
        <p className="mt-1 text-sm leading-relaxed text-slate-400">
          三层完全解耦，HTTP API 通信。展示层 (React / WebSocket / 移动端) 独立升级，不触及推理引擎。
          FastAPI 服务化使推理层可横向扩展，分析层可接入更多评估工具。当前原型可直接演进为产品级系统。
        </p>
      </div>
    </div>
  );
}
