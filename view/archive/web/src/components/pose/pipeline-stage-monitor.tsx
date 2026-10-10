import { useState } from "react";
import { cn } from "@/lib/utils";
import {
  Video,
  Box,
  Layers,
  TrendingUp,
  Cpu,
  Lightbulb,
  ChevronRight,
  Zap,
} from "lucide-react";

interface StageConfig {
  id: string;
  label: string;
  icon: React.ReactNode;
  latencyMs: { standard: number; lightweight: number; ultra: number };
  innovation: string;
  description: string;
  tech: string;
}

const stages: StageConfig[] = [
  {
    id: "capture",
    label: "视频采集",
    icon: <Video size={16} />,
    latencyMs: { standard: 45, lightweight: 28, ultra: 15 },
    innovation: "三源输入",
    description: "样例库/上传/摄像头统一输入，自适应分辨率",
    tech: "MediaRecorder / HTML5 Video",
  },
  {
    id: "pose",
    label: "姿态提取",
    icon: <Box size={16} />,
    latencyMs: { standard: 180, lightweight: 75, ultra: 40 },
    innovation: "133点手部精细",
    description: "全身133关键点提取，手部21点置信度过滤",
    tech: "RTMPose + ONNX Runtime",
  },
  {
    id: "fusion",
    label: "双流编码",
    icon: <Layers size={16} />,
    latencyMs: { standard: 150, lightweight: 55, ultra: 25 },
    innovation: "PGF融合 + 门控",
    description: "RGB语义+Pose时空双流，不确定性门控自适应融合",
    tech: "ST-GCN + EfficientNet + PGF",
  },
  {
    id: "temporal",
    label: "时序建模",
    icon: <TrendingUp size={16} />,
    latencyMs: { standard: 90, lightweight: 28, ultra: 12 },
    innovation: "自注意力上下文",
    description: "Transformer捕获长程帧间依赖，跨帧自注意力",
    tech: "Multi-Head Self-Attention",
  },
  {
    id: "decode",
    label: "文本生成",
    icon: <Cpu size={16} />,
    latencyMs: { standard: 65, lightweight: 20, ultra: 10 },
    innovation: "mT5语义增强",
    description: "自回归解码器逐词生成，mT5语义先验引导",
    tech: "mT5 Decoder + Beam Search",
  },
  {
    id: "xai",
    label: "归因分析",
    icon: <Lightbulb size={16} />,
    latencyMs: { standard: 55, lightweight: 18, ultra: 8 },
    innovation: "全链路可解释",
    description: "Attention + Grad-CAM + 混淆矩阵，决策可追溯",
    tech: "Grad-CAM + Attention Viz",
  },
];

type PerfTier = "standard" | "lightweight" | "ultra";

const tierLabels: Record<PerfTier, { label: string; color: string }> = {
  standard: { label: "标准模式 ~585ms", color: "text-cyan-400" },
  lightweight: { label: "轻量模式 ~224ms", color: "text-emerald-400" },
  ultra: { label: "超轻量 ~110ms", color: "text-amber-400" },
};

interface Props {
  className?: string;
}

export function PipelineStageMonitor({ className }: Props) {
  const [tier, setTier] = useState<PerfTier>("standard");
  const [activeStage, setActiveStage] = useState<string | null>(null);

  const totalLatency = stages.reduce((s, st) => s + st.latencyMs[tier], 0);

  return (
    <div className={cn("glass-card rounded-2xl p-4", className)}>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-gradient-to-br from-purple-500 to-violet-500 text-white shadow">
            <Zap size={14} />
          </div>
          <div>
            <span className="text-sm font-bold">推理流水线监控</span>
            <span className="ml-2 text-[10px] text-purple-400 font-mono">6-STAGE PIPELINE</span>
          </div>
        </div>
        {/* Tier selector */}
        <div className="flex rounded-lg bg-slate-800/60 border border-slate-700/30 p-0.5">
          {(["standard", "lightweight", "ultra"] as PerfTier[]).map((t) => (
            <button
              key={t}
              onClick={() => setTier(t)}
              className={cn(
                "px-2.5 py-1 text-[10px] font-semibold rounded-md transition-all duration-200",
                tier === t
                  ? "bg-gradient-to-r from-cyan-500/20 to-teal-500/20 text-cyan-400 border border-cyan-400/20"
                  : "text-slate-400 hover:text-slate-300",
              )}
            >
              {t === "standard" ? "标准" : t === "lightweight" ? "轻量" : "超轻"}
            </button>
          ))}
        </div>
      </div>

      {/* Pipeline stages */}
      <div className="space-y-1.5">
        {stages.map((stage, idx) => {
          const isActive = activeStage === stage.id;
          const latency = stage.latencyMs[tier];
          const widthPct = (latency / totalLatency) * 100;

          return (
            <div key={stage.id}>
              <button
                className={cn(
                  "w-full rounded-xl border p-3 text-left transition-all duration-200 group",
                  isActive
                    ? "border-cyan-400/30 bg-cyan-400/5"
                    : "border-slate-700/20 bg-slate-800/20 hover:border-slate-600/30 hover:bg-slate-800/30",
                )}
                onClick={() => setActiveStage(isActive ? null : stage.id)}
              >
                <div className="flex items-center gap-3">
                  {/* Stage number */}
                  <div
                    className={cn(
                      "flex h-8 w-8 shrink-0 items-center justify-center rounded-xl text-xs font-bold transition-all",
                      isActive
                        ? "bg-gradient-to-br from-cyan-500 to-teal-500 text-white shadow-lg shadow-cyan-500/20"
                        : "bg-slate-700/50 text-slate-400 group-hover:bg-slate-700 group-hover:text-slate-200",
                    )}
                  >
                    {idx + 1}
                  </div>

                  {/* Stage info */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-semibold text-slate-200">{stage.label}</span>
                      <span className="text-[10px] text-slate-400">{stage.tech}</span>
                    </div>
                    {isActive && (
                      <p className="mt-0.5 text-[11px] text-slate-400">{stage.description}</p>
                    )}
                  </div>

                  {/* Innovation badge */}
                  <span
                    className={cn(
                      "shrink-0 rounded-md border px-2 py-0.5 text-[9px] font-semibold",
                      isActive
                        ? "bg-purple-400/10 border-purple-400/30 text-purple-400"
                        : "bg-purple-400/5 border-purple-400/10 text-purple-400/60",
                    )}
                  >
                    {stage.innovation}
                  </span>

                  {/* Latency bar */}
                  <div className="hidden sm:flex items-center gap-2 w-24 shrink-0">
                    <div className="flex-1 h-1.5 rounded-full bg-slate-700/50 overflow-hidden">
                      <div
                        className={cn(
                          "h-full rounded-full transition-all duration-300",
                          idx === 1
                            ? "bg-gradient-to-r from-orange-500 to-amber-400"
                            : idx === 2
                              ? "bg-gradient-to-r from-red-500 to-rose-400"
                              : "bg-gradient-to-r from-cyan-500 to-teal-400",
                        )}
                        style={{ width: `${Math.max(widthPct * 3, 8)}%` }}
                      />
                    </div>
                    <span className="text-[10px] text-slate-400 font-mono w-10 text-right">
                      {latency}ms
                    </span>
                  </div>

                  {/* Arrow */}
                  {idx < stages.length - 1 && (
                    <ChevronRight size={14} className="text-slate-600 shrink-0" />
                  )}
                </div>

                {/* Expanded detail */}
                {isActive && (
                  <div className="mt-3 pl-11">
                    <div className="flex items-center gap-3 text-[11px]">
                      <span className="text-slate-400">
                        延迟占比：<span className="text-slate-200 font-mono">{widthPct.toFixed(1)}%</span>
                      </span>
                      <span className="text-slate-400">
                        总计：<span className="text-slate-200 font-mono">{latency}ms</span>
                      </span>
                    </div>
                  </div>
                )}
              </button>
            </div>
          );
        })}
      </div>

      {/* Total latency bar */}
      <div className="mt-3 flex items-center gap-3 text-[11px]">
        <span className="text-slate-400">端到端总延迟：</span>
        <span className={cn("font-mono font-bold", tierLabels[tier].color)}>
          {totalLatency}ms
        </span>
        <div className="flex-1 h-2 rounded-full bg-slate-700/50 overflow-hidden">
          {stages.map((stage, idx) => (
            <div
              key={stage.id}
              className="h-full float-left transition-all duration-300"
              style={{
                width: `${(stage.latencyMs[tier] / totalLatency) * 100}%`,
                backgroundColor:
                  idx === 1 ? "#f97316" : idx === 2 ? "#ef4444" : "#22d3ee",
                opacity: 0.7,
              }}
            />
          ))}
        </div>
        <span className={cn("font-mono font-bold", tierLabels[tier].color)}>
          {tierLabels[tier].label.split(" ")[0]}
        </span>
      </div>
    </div>
  );
}
