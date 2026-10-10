import { useMemo } from "react";
import { cn } from "@/lib/utils";
import { Camera, Zap } from "lucide-react";

interface TopFrame {
  frame: number;
  xai_score: number;
  encoder_attention?: number;
  gradcam_combined?: number;
}

interface Props {
  frames: TopFrame[];
  className?: string;
}

export function KeyframeGallery({ frames, className }: Props) {
  if (!frames.length) return null;

  const maxScore = Math.max(...frames.map((f) => f.xai_score), 0.001);

  return (
    <div className={cn("glass-card rounded-2xl p-4", className)}>
      <div className="flex items-center gap-2 mb-3">
        <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-gradient-to-br from-amber-500 to-orange-500 text-white shadow">
          <Camera size={14} />
        </div>
        <div>
          <span className="text-sm font-bold">关键帧定位</span>
          <span className="ml-2 text-[10px] text-amber-400 font-mono">
            TOP-6 KEYFRAMES
          </span>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2.5">
        {frames.slice(0, 6).map((f, i) => {
          const scorePct = ((f.xai_score / maxScore) * 100).toFixed(0);
          const barH = Math.max(12, (f.xai_score / maxScore) * 100);
          return (
            <div
              key={i}
              className="rounded-xl bg-slate-800/40 border border-slate-700/30 overflow-hidden group hover:border-amber-400/30 transition-colors"
            >
              {/* Frame placeholder with index */}
              <div className="relative aspect-video bg-slate-900/80 flex items-center justify-center">
                <div className="text-center">
                  <Camera size={20} className="text-slate-600 mx-auto mb-1" />
                  <span className="text-xs font-mono text-slate-500">
                    F{f.frame}
                  </span>
                </div>
                {/* Rank badge */}
                <div className="absolute top-1.5 left-1.5 flex h-5 w-5 items-center justify-center rounded-full bg-amber-400/20 border border-amber-400/30 text-[10px] font-bold text-amber-400">
                  {i + 1}
                </div>
              </div>

              {/* Score bar */}
              <div className="p-2">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[10px] text-slate-400">归因分数</span>
                  <span className="text-[10px] font-mono font-bold text-amber-400">
                    {scorePct}%
                  </span>
                </div>
                <div className="h-1.5 rounded-full bg-slate-700/50 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-amber-500 to-orange-400 transition-all duration-500"
                    style={{ width: `${barH}%` }}
                  />
                </div>

                {/* Sub-scores */}
                <div className="mt-1.5 flex justify-between text-[9px]">
                  <span className="text-slate-500">
                    Attn {(f.encoder_attention ?? 0).toFixed(2)}
                  </span>
                  <span className="text-slate-500">
                    CAM {(f.gradcam_combined ?? 0).toFixed(2)}
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="mt-2 flex items-center gap-1.5 text-[10px] text-slate-400">
        <Zap size={11} className="text-amber-400" />
        <span>
          XAI Score = 0.55 × Grad-CAM + 0.45 × Encoder Attention。
          前 6 帧贡献了主要的模型决策依据。
        </span>
      </div>
    </div>
  );
}
