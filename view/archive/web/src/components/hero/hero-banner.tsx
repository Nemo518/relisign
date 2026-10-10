import { useAppStore } from "@/store/app-store";
import { Sparkles, Cpu, Zap } from "lucide-react";

export function HeroBanner() {
  const { stage, judge, split } = useAppStore();

  return (
    <div className="hero-gradient relative mb-5 overflow-hidden rounded-3xl px-8 py-7 text-white shadow-2xl scan-lines">
      {/* Orbital ring decorations */}
      <div className="orbital-ring absolute -right-20 -top-20 h-64 w-64 opacity-30" />
      <div className="orbital-ring absolute -bottom-32 -left-10 h-80 w-80 opacity-20" style={{ animationDuration: "28s" }} />

      {/* Floating glow orbs */}
      <div className="absolute right-1/4 top-8 h-2 w-2 rounded-full bg-cyan-400 animate-pulse-ring" />
      <div
        className="absolute bottom-6 right-1/3 h-1.5 w-1.5 rounded-full bg-magenta-400 animate-pulse-ring"
        style={{ animationDelay: "0.8s" }}
      />

      {/* Animated scan line */}
      <div className="absolute left-0 right-0 top-0 h-px bg-gradient-to-r from-transparent via-cyan-400/40 to-transparent animate-scan" />

      <div className="relative">
        {/* Title with glow */}
        <div className="flex items-center gap-3 mb-2">
          <div className="flex items-center gap-1.5 rounded-xl bg-cyan-400/10 border border-cyan-400/20 px-3 py-1.5">
            <Cpu size={16} className="text-cyan-400" />
            <span className="text-xs font-semibold text-cyan-400 tracking-wider">AI POWERED</span>
          </div>
          <div className="flex items-center gap-1.5 rounded-xl bg-amber-400/10 border border-amber-400/20 px-3 py-1.5">
            <Zap size={16} className="text-amber-400" />
            <span className="text-xs font-semibold text-amber-400 tracking-wider">REAL-TIME</span>
          </div>
          <div className="flex items-center gap-1.5 rounded-xl bg-purple-400/10 border border-purple-400/20 px-3 py-1.5">
            <Sparkles size={16} className="text-purple-400" />
            <span className="text-xs font-semibold text-purple-400 tracking-wider">EXPLAINABLE</span>
          </div>
        </div>

        <h1 className="text-4xl font-black tracking-tight mb-1">
          <span className="text-glow-cyan">ReliSign</span>
        </h1>
        <p className="max-w-2xl text-base leading-relaxed text-white/70">
          多模态手语翻译 · 模型归因 · 端到端推理
        </p>

        {/* Status indicators */}
        <div className="mt-4 flex flex-wrap gap-3">
          {[
            { label: "展示层级", value: stage, color: "border-cyan-400/30 bg-cyan-400/10 text-cyan-300" },
            { label: "评委视角", value: judge, color: "border-purple-400/30 bg-purple-400/10 text-purple-300" },
            { label: "主数据集", value: split.toUpperCase(), color: "border-amber-400/30 bg-amber-400/10 text-amber-300" },
          ].map((item) => (
            <div
              key={item.label}
              className={`flex items-center gap-2 rounded-xl border px-3.5 py-2 text-xs font-semibold ${item.color}`}
            >
              <span className="pulse-dot" />
              <span className="opacity-70">{item.label}</span>
              <span>{item.value}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
