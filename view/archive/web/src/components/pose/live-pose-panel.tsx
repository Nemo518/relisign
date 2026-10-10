import { useMemo } from "react";
import type { NormalizedLandmark } from "@mediapipe/tasks-vision";
import { PoseLandmarker } from "@mediapipe/tasks-vision";
import { cn } from "@/lib/utils";
import { Eye, Activity } from "lucide-react";
import type { Connection } from "./live-pose-overlay";

interface Props {
  landmarks: NormalizedLandmark[] | null;
  connections: Connection[];
  className?: string;
}

// MediaPipe 33-landmark body-part grouping
function getGroup(idx: number): "face" | "body" | "left_hand" | "right_hand" {
  if (idx <= 10) return "face";
  if (idx === 11 || idx === 12 || (idx >= 23 && idx <= 32)) return "body";
  if (idx === 13 || idx === 15 || idx === 17 || idx === 19 || idx === 21) return "left_hand";
  return "right_hand";
}

const PART_META: Record<string, { stroke: string; fill: string; label: string; kpRange: string }> = {
  body:      { stroke: "#22d3ee", fill: "#06b6d4", label: "身体/足部", kpRange: "11-12,23-32" },
  left_hand: { stroke: "#c084fc", fill: "#a855f7", label: "左手",       kpRange: "13-21 奇数" },
  right_hand:{ stroke: "#f472b6", fill: "#ec4899", label: "右手",       kpRange: "14-22 偶数" },
  face:      { stroke: "#fbbf24", fill: "#f59e0b", label: "面部",       kpRange: "0-10" },
};

export function LivePosePanel({ landmarks, connections, className }: Props) {
  const hasData = landmarks && landmarks.length >= 33;

  // Per-group stats
  const groupStats = useMemo(() => {
    const stats: Record<string, { count: number; avgConf: number; highConf: number; lowConf: number; midConf: number }> = {};
    for (const key of Object.keys(PART_META)) {
      stats[key] = { count: 0, avgConf: 0, highConf: 0, lowConf: 0, midConf: 0 };
    }
    if (!hasData) return stats;

    for (let i = 0; i < landmarks.length; i++) {
      const group = getGroup(i);
      const conf = landmarks[i].visibility;
      stats[group].count++;
      stats[group].avgConf += conf;
      if (conf >= 0.6) stats[group].highConf++;
      else if (conf >= 0.3) stats[group].midConf++;
      else if (conf > 0) stats[group].lowConf++;
    }
    for (const key of Object.keys(stats)) {
      const s = stats[key];
      if (s.count > 0) s.avgConf /= s.count;
    }
    return stats;
  }, [landmarks, hasData]);

  const overallConf = useMemo(() => {
    if (!hasData) return 0;
    let sum = 0;
    for (const lm of landmarks) sum += lm.visibility;
    return sum / landmarks.length;
  }, [landmarks, hasData]);

  // Build skeleton SVG points from live landmarks
  const svgPoints = landmarks ?? [];

  const pad = 0.04;
  const vb = `${-pad} ${-pad} ${1 + pad * 2} ${1 + pad * 2}`;

  return (
    <div className={cn("glass-card rounded-2xl p-4", className)}>
      {/* ── Header ── */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-gradient-to-br from-purple-500 to-pink-500 text-white shadow">
            <Activity size={14} />
          </div>
          <div>
            <span className="text-sm font-bold">实时姿态监测</span>
            <span className="ml-2 text-[10px] text-purple-400 font-mono">
              {hasData ? `33 KP · LIVE` : "等待检测"}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {hasData ? (
            <span className={cn(
              "rounded px-2 py-0.5 text-[10px] font-mono",
              overallConf >= 0.7
                ? "bg-emerald-400/10 text-emerald-400"
                : overallConf >= 0.4
                  ? "bg-amber-400/10 text-amber-400"
                  : "bg-rose-400/10 text-rose-400",
            )}>
              LIVE · {(overallConf * 100).toFixed(0)}%
            </span>
          ) : (
            <span className="rounded bg-slate-400/10 px-2 py-0.5 text-[10px] text-slate-400 font-mono">
              NO DATA
            </span>
          )}
          <Eye size={12} className={hasData ? "text-emerald-400" : "text-slate-400"} />
          <span className={cn("text-[10px] font-mono", hasData ? "text-emerald-400" : "text-slate-400")}>
            {hasData ? `${landmarks.length} KP` : "0 KP"}
          </span>
        </div>
      </div>

      {/* ── SVG skeleton ── */}
      <div className="relative rounded-xl bg-slate-950/80 border border-slate-700/30 overflow-hidden mb-3">
        {!hasData ? (
          <div className="flex items-center justify-center aspect-[4/3]">
            <div className="text-center">
              <Eye size={28} className="text-slate-500 mx-auto mb-2" />
              <p className="text-[10px] text-slate-400">打开摄像头后自动检测33关键点骨架</p>
              <p className="mt-1 text-[9px] text-slate-500">
                MediaPipe Pose: Body + Face + Hands (33 KP)
              </p>
            </div>
          </div>
        ) : (
          <svg viewBox={vb} className="w-full aspect-[4/3]" preserveAspectRatio="xMidYMid meet">
            <defs>
              <filter id="live-glow-strong" x="-200%" y="-200%" width="500%" height="500%">
                <feGaussianBlur stdDeviation="0.006" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
              <filter id="live-glow-soft" x="-200%" y="-200%" width="500%" height="500%">
                <feGaussianBlur stdDeviation="0.003" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
              <radialGradient id="live-bg" cx="50%" cy="40%" r="50%">
                <stop offset="0%" stopColor="#0f172a" stopOpacity="0.6" />
                <stop offset="100%" stopColor="#020617" stopOpacity="0.95" />
              </radialGradient>
            </defs>

            {/* Background */}
            <rect x={-pad} y={-pad} width={1 + pad * 2} height={1 + pad * 2} fill="url(#live-bg)" />
            {/* Grid */}
            {Array.from({ length: 8 }).map((_, i) => (
              <line key={`gv-${i}`} x1={(i + 1) * 0.125} y1={-pad} x2={(i + 1) * 0.125} y2={1 + pad}
                stroke="#1e293b" strokeWidth="0.0005" opacity="0.4" />
            ))}
            {Array.from({ length: 6 }).map((_, i) => (
              <line key={`gh-${i}`} x1={-pad} y1={(i + 1) * 0.166} x2={1 + pad} y2={(i + 1) * 0.166}
                stroke="#1e293b" strokeWidth="0.0005" opacity="0.4" />
            ))}

            {/* Bones */}
            {connections.map((conn, i) => {
              const a = svgPoints[conn.start];
              const b = svgPoints[conn.end];
              if (!a || !b || a.visibility < 0.08 || b.visibility < 0.08) return null;
              const group = getGroup(conn.start);
              const colors = PART_META[group];
              const minVis = Math.min(a.visibility, b.visibility);
              const w = group === "face" ? 0.0025 : group === "body" ? 0.004 : 0.003;
              return (
                <g key={`lb-${i}`}>
                  <line x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                    stroke={colors.stroke} strokeWidth={w * 2.5}
                    opacity={minVis * 0.2} filter="url(#live-glow-strong)" />
                  <line x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                    stroke={colors.stroke} strokeWidth={w}
                    opacity={0.15 + minVis * 0.85} />
                </g>
              );
            })}

            {/* Keypoints — 3-tier confidence */}
            {svgPoints.map((kp, i) => {
              if (kp.visibility < 0.08) return null;
              const group = getGroup(i);
              const colors = PART_META[group];
              const baseR = group === "face" ? 0.003 : group === "body" ? 0.005 : 0.004;

              if (kp.visibility >= 0.6) {
                return (
                  <g key={`lkp-${i}`}>
                    <circle cx={kp.x} cy={kp.y} r={baseR * 2.5} fill={colors.fill}
                      opacity={0.12} filter="url(#live-glow-strong)" />
                    <circle cx={kp.x} cy={kp.y} r={baseR} fill={colors.fill}
                      opacity={0.5 + kp.visibility * 0.5} />
                    <circle cx={kp.x} cy={kp.y} r={baseR * 0.35} fill="#fff"
                      opacity={0.55 + kp.visibility * 0.3} />
                  </g>
                );
              }
              if (kp.visibility >= 0.3) {
                return (
                  <g key={`lkp-${i}`}>
                    <circle cx={kp.x} cy={kp.y} r={baseR * 1.5} fill={colors.fill}
                      opacity={0.06} filter="url(#live-glow-soft)" />
                    <circle cx={kp.x} cy={kp.y} r={baseR * 0.7} fill={colors.fill}
                      opacity={0.2 + kp.visibility * 0.35} />
                  </g>
                );
              }
              return (
                <g key={`lkp-${i}`}>
                  <circle cx={kp.x} cy={kp.y} r={baseR * 0.6} fill="none"
                    stroke={colors.stroke} strokeWidth={baseR * 0.2}
                    opacity={0.15 + kp.visibility * 0.35} />
                </g>
              );
            })}
          </svg>
        )}
      </div>

      {/* ── Confidence tier legend ── */}
      <div className="flex items-center gap-3 text-[9px] text-slate-400 mb-2">
        <span className="font-semibold text-slate-300">置信度：</span>
        <span className="inline-flex items-center gap-1">
          <span className="h-1.5 w-1.5 rounded-full bg-white/60" />高(≥60%)
        </span>
        <span className="inline-flex items-center gap-1">
          <span className="h-1.5 w-1.5 rounded-full bg-white/25" />中(30-60%)
        </span>
        <span className="inline-flex items-center gap-1">
          <span className="h-1.5 w-1.5 rounded-full border border-white/20 bg-transparent" />低(&lt;30%)
        </span>
      </div>

      {/* ── Per-group stats cards ── */}
      <div className="grid grid-cols-4 gap-2">
        {Object.entries(PART_META).map(([key, meta]) => {
          const stats = groupStats[key];
          if (!stats) return null;
          const barW = hasData ? Math.min(stats.avgConf * 100, 100) : 0;
          return (
            <div key={key} className="rounded-lg border border-slate-700/20 bg-slate-800/30 p-2">
              <div className="flex items-center gap-1.5 mb-1">
                <div className="h-2 w-2 rounded-full shadow" style={{ backgroundColor: meta.fill, boxShadow: `0 0 4px ${meta.fill}` }} />
                <span className="text-[10px] font-semibold text-slate-300">{meta.label}</span>
              </div>
              <div className="text-[9px] text-slate-400 space-y-0.5">
                <div className="flex justify-between">
                  <span>{stats.count}点</span>
                  <span>高{stats.highConf} 中{stats.midConf} 低{stats.lowConf}</span>
                </div>
                <div className="h-1 rounded-full bg-slate-700/50 overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-300"
                    style={{ width: `${barW}%`, backgroundColor: meta.fill, opacity: 0.7 }}
                  />
                </div>
                <div className="text-right text-[8px] text-slate-500">
                  {(stats.avgConf * 100).toFixed(0)}%
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
