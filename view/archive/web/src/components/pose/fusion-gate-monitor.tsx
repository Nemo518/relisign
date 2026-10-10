import { cn } from "@/lib/utils";
import { ShieldCheck, ArrowRight, Brain, Eye, Hand } from "lucide-react";

interface Props {
  className?: string;
}

const metaInputs = [
  {
    icon: <Hand size={14} />,
    label: "手部置信度",
    desc: "21个手部关键点平均置信度，反映姿态提取质量",
    color: "text-cyan-400",
    bg: "bg-cyan-400/10 border-cyan-400/20",
  },
  {
    icon: <Eye size={14} />,
    label: "选择密度",
    desc: "置信度引导的帧选择策略，控制RGB帧采样密度",
    color: "text-purple-400",
    bg: "bg-purple-400/10 border-purple-400/20",
  },
  {
    icon: <Brain size={14} />,
    label: "手部存在率",
    desc: "视频片段中手部可见帧的比例，判断遮挡程度",
    color: "text-amber-400",
    bg: "bg-amber-400/10 border-amber-400/20",
  },
];

export function FusionGateMonitor({ className }: Props) {
  return (
    <div className={cn("glass-card rounded-2xl p-4", className)}>
      {/* Header */}
      <div className="flex items-center gap-2 mb-4">
        <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-gradient-to-br from-red-500 to-orange-500 text-white shadow">
          <ShieldCheck size={14} />
        </div>
        <div>
          <span className="text-sm font-bold">不确定性感知融合门控</span>
          <span className="ml-2 text-[10px] text-red-400 font-mono">
            UNCERTAINTY GATE
          </span>
        </div>
      </div>

      {/* Mechanism flow diagram */}
      <div className="rounded-xl bg-slate-800/40 border border-slate-700/30 p-4">
        {/* Three meta inputs */}
        <div className="grid grid-cols-3 gap-3 mb-4">
          {metaInputs.map((item) => (
            <div
              key={item.label}
              className={cn("rounded-xl border p-3 text-center", item.bg)}
            >
              <div className={cn("inline-flex items-center gap-1 mb-1", item.color)}>
                {item.icon}
                <span className="text-[10px] font-semibold">{item.label}</span>
              </div>
              <p className="text-[9px] text-slate-400 leading-relaxed">
                {item.desc}
              </p>
            </div>
          ))}
        </div>

        {/* Arrow down */}
        <div className="flex justify-center mb-3">
          <div className="flex items-center gap-1 text-[10px] text-slate-400">
            <ArrowRight size={12} className="rotate-90" />
            <span>3D 元数据拼接 → Conv1D → GELU → Conv1D → Sigmoid</span>
            <ArrowRight size={12} className="rotate-90" />
          </div>
        </div>

        {/* Gate output */}
        <div className="rounded-xl bg-slate-900/60 border border-slate-700/30 p-3">
          <div className="text-center mb-2">
            <span className="text-[10px] text-slate-400">
              门控分数 g ∈ [0, 1]
            </span>
          </div>
          <div className="flex items-center gap-3">
            {/* Pose side */}
            <div className="flex-1 rounded-lg bg-slate-800/60 border border-slate-600/30 p-2.5 text-center">
              <div className="text-[10px] text-slate-500 mb-0.5">
                g → 0 (退回姿态)
              </div>
              <div className="text-xs font-semibold text-slate-300">
                纯 ST-GCN 姿态推理
              </div>
              <div className="text-[9px] text-slate-400 mt-0.5">
                手部遮挡/低质量时自动退回
              </div>
            </div>

            {/* Gate bar */}
            <div className="w-16 shrink-0">
              <div className="relative h-2 rounded-full bg-gradient-to-r from-slate-500 via-amber-400 to-cyan-400 mb-1">
                <div
                  className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 h-4 w-1 rounded-full bg-white shadow"
                  style={{ left: "60%" }}
                />
              </div>
              <div className="flex justify-between text-[8px] text-slate-500">
                <span>姿态</span>
                <span>RGB</span>
              </div>
            </div>

            {/* RGB side */}
            <div className="flex-1 rounded-lg bg-cyan-400/5 border border-cyan-400/20 p-2.5 text-center">
              <div className="text-[10px] text-cyan-400 mb-0.5">
                g → 1 (信任RGB)
              </div>
              <div className="text-xs font-semibold text-cyan-300">
                PGF 跨模态融合增强
              </div>
              <div className="text-[9px] text-slate-400 mt-0.5">
                高质量手部图像时启用RGB增强
              </div>
            </div>
          </div>

          {/* Gate formula */}
          <div className="mt-3 text-center">
            <code className="text-[10px] text-slate-400 bg-slate-900/60 px-3 py-1 rounded font-mono">
              fused = gate_score × PGF(rgb, pose) + (1 − gate_score) × STGCN(pose)
            </code>
          </div>
        </div>
      </div>

      {/* Key insight */}
      <div className="mt-3 rounded-xl bg-red-400/5 border border-red-400/15 p-2.5">
        <div className="flex items-center gap-1.5 text-[10px] text-red-400">
          <ShieldCheck size={12} />
          <span className="font-semibold">核心创新</span>
        </div>
        <p className="mt-1 text-[10px] text-slate-400 leading-relaxed">
          偏置初始化为 <code className="text-slate-300">-2.0</code>（Sigmoid(−2.0)
          ≈ 0.12），使模型先信任稳定的姿态特征，再逐步学习何时融入
          RGB。这是"可靠性优先"的设计哲学：宁可不用，不可错用。
        </p>
      </div>
    </div>
  );
}
