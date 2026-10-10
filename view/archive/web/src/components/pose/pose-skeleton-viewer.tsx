import { useMemo } from "react";
import { cn } from "@/lib/utils";
import { BODY_PART_COLORS, getKeypointGroup, getAllBones } from "@/lib/pose-constants";
import type { PoseSequence } from "@/types/api";
import { Eye, Link2, Loader2 } from "lucide-react";

interface Props {
  className?: string;
  frameIndex: number;
  isPlaying: boolean;
  poseSequence: PoseSequence | null;
  isLoading: boolean;
}

export function PoseSkeletonViewer({
  className,
  frameIndex,
  isPlaying,
  poseSequence,
  isLoading,
}: Props) {
  const bones = useMemo(() => getAllBones(), []);

  const hasData = poseSequence && poseSequence.frames && poseSequence.frames.length > 0;

  const totalFrames = poseSequence?.num_frames ?? 0;
  const clampedFrame = totalFrames > 0 ? Math.min(frameIndex, totalFrames - 1) : 0;
  const points: { x: number; y: number; confidence: number }[] =
    hasData && poseSequence!.frames[clampedFrame]?.points
      ? poseSequence!.frames[clampedFrame].points
      : [];
  const fps = poseSequence?.fps ?? 25;
  const timestamp = totalFrames > 0 ? clampedFrame / Math.max(fps, 1) : 0;

  // Per-group stats
  const groupStats: Record<string, { count: number; avgConf: number; highConf: number; lowConf: number }> = {};
  for (const group of ["body", "left_hand", "right_hand", "face"]) {
    const idxs = points.reduce<number[]>((acc, _, i) => {
      if (getKeypointGroup(i) === group) acc.push(i);
      return acc;
    }, []);
    const confs = idxs.map((i) => points[i]?.confidence ?? 0).filter((c) => c > 0);
    groupStats[group] = {
      count: idxs.length,
      avgConf: confs.length > 0 ? confs.reduce((s, c) => s + c, 0) / confs.length : 0,
      highConf: confs.filter((c) => c >= 0.6).length,
      lowConf: confs.filter((c) => c > 0 && c < 0.3).length,
    };
  }

  // Overall confidence for badge color
  const overallConf =
    points.length > 0
      ? points.reduce((s, p) => s + p.confidence, 0) / points.length
      : 0;

  const pad = 0.04;
  const vb = `${-pad} ${-pad} ${1 + pad * 2} ${1 + pad * 2}`;

  return (
    <div className={cn("glass-card rounded-2xl p-4", className)}>
      {/* ── Header ── */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-gradient-to-br from-cyan-500 to-teal-500 text-white shadow">
            <Eye size={14} />
          </div>
          <div>
            <span className="text-sm font-bold">姿态提取可视化</span>
            <span className="ml-2 text-[10px] text-cyan-400 font-mono">
              {hasData ? `${poseSequence!.num_keypoints} KP · ${totalFrames}F` : "等待数据"}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {isLoading && (
            <span className="rounded bg-cyan-400/10 px-2 py-0.5 text-[10px] text-cyan-400 font-mono">FETCHING</span>
          )}
          {hasData && (
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
          )}
          {!hasData && !isLoading && (
            <span className="rounded bg-slate-400/10 px-2 py-0.5 text-[10px] text-slate-400 font-mono">NO DATA</span>
          )}
          <Link2 size={12} className={isPlaying ? "text-emerald-400" : "text-slate-400"} />
          <span className={cn("text-[10px] font-mono", isPlaying ? "text-emerald-400" : "text-slate-400")}>
            F{clampedFrame + 1}/{Math.max(totalFrames, 1)}
          </span>
          <span className="text-[10px] text-slate-500">@{timestamp.toFixed(1)}s</span>
        </div>
      </div>

      {/* ── SVG area ── */}
      <div className="relative rounded-xl bg-slate-950/80 border border-slate-700/30 overflow-hidden">
        {isLoading ? (
          <div className="flex items-center justify-center aspect-[4/3]">
            <div className="text-center">
              <Loader2 size={28} className="animate-spin text-cyan-400 mx-auto mb-2" />
              <p className="text-[10px] text-slate-400 font-mono">RTMPose 133关键点提取中...</p>
            </div>
          </div>
        ) : points.length === 0 ? (
          <div className="flex items-center justify-center aspect-[4/3]">
            <div className="text-center">
              <Eye size={28} className="text-slate-500 mx-auto mb-2" />
              <p className="text-[10px] text-slate-400">
                {hasData ? `Frame ${clampedFrame + 1} has no keypoints` : "选择样例视频即可加载133关键点骨架"}
              </p>
              <p className="mt-1 text-[9px] text-slate-500">
                RTMPose Wholebody: Body(17) + Feet(6) + Face(68) + Hands(2×21)
              </p>
            </div>
          </div>
        ) : (
          <svg viewBox={vb} className="w-full aspect-[4/3]" preserveAspectRatio="xMidYMid meet">
            <defs>
              {/* ── Glow filters ── */}
              <filter id="glow-strong" x="-200%" y="-200%" width="500%" height="500%">
                <feGaussianBlur stdDeviation="0.004" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
              <filter id="glow-soft" x="-200%" y="-200%" width="500%" height="500%">
                <feGaussianBlur stdDeviation="0.002" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
              {/* ── Radial gradient background ── */}
              <radialGradient id="bg-glow" cx="50%" cy="40%" r="50%">
                <stop offset="0%" stopColor="#0f172a" stopOpacity="0.6" />
                <stop offset="100%" stopColor="#020617" stopOpacity="0.95" />
              </radialGradient>
            </defs>

            {/* Background with grid + radial glow */}
            <rect x={-pad} y={-pad} width={1 + pad * 2} height={1 + pad * 2} fill="url(#bg-glow)" />
            {/* Subtle grid */}
            {Array.from({ length: 10 }).map((_, i) => (
              <line key={`gv-${i}`} x1={(i + 1) * 0.1} y1={-pad} x2={(i + 1) * 0.1} y2={1 + pad}
                stroke="#1e293b" strokeWidth="0.0005" opacity="0.5" />
            ))}
            {Array.from({ length: 8 }).map((_, i) => (
              <line key={`gh-${i}`} x1={-pad} y1={(i + 1) * 0.125} x2={1 + pad} y2={(i + 1) * 0.125}
                stroke="#1e293b" strokeWidth="0.0005" opacity="0.5" />
            ))}

            {/* ── Bones ── */}
            {bones.map(({ from, to, group }) => {
              if (from >= points.length || to >= points.length) return null;
              const a = points[from];
              const b = points[to];
              if (!a || !b || a.confidence < 0.05 || b.confidence < 0.05) return null;
              const conf = Math.min(a.confidence, b.confidence);
              const colors = BODY_PART_COLORS[group] ?? BODY_PART_COLORS.body;
              const w = group === "face" ? 0.001 : group === "body" ? 0.0035 : 0.0025;
              return (
                <g key={`b-${from}-${to}`}>
                  {/* Glow layer */}
                  <line
                    x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                    stroke={colors.stroke}
                    strokeWidth={w * 2.5}
                    opacity={conf * 0.25}
                    filter="url(#glow-strong)"
                  />
                  {/* Core line */}
                  <line
                    x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                    stroke={colors.stroke}
                    strokeWidth={w}
                    opacity={0.2 + conf * 0.8}
                  />
                </g>
              );
            })}

            {/* ── Keypoints (3-tier confidence) ── */}
            {points.map((kp, i) => {
              const group = getKeypointGroup(i);
              const colors = BODY_PART_COLORS[group] ?? BODY_PART_COLORS.body;
              const baseR = group === "face" ? 0.0012 : group === "body" ? 0.0035 : 0.0028;

              // Tier 1: high confidence (≥0.6) — filled + glow
              if (kp.confidence >= 0.6) {
                return (
                  <g key={`kp-${i}`}>
                    <circle cx={kp.x} cy={kp.y} r={baseR * 2.2} fill={colors.fill}
                      opacity={0.15} filter="url(#glow-strong)" />
                    <circle cx={kp.x} cy={kp.y} r={baseR} fill={colors.fill}
                      opacity={0.55 + kp.confidence * 0.45} />
                    <circle cx={kp.x} cy={kp.y} r={baseR * 0.4} fill="#ffffff"
                      opacity={0.6} />
                  </g>
                );
              }

              // Tier 2: medium confidence (0.3–0.6) — semi-transparent
              if (kp.confidence >= 0.3) {
                return (
                  <g key={`kp-${i}`}>
                    <circle cx={kp.x} cy={kp.y} r={baseR * 1.4} fill={colors.fill}
                      opacity={0.08} filter="url(#glow-soft)" />
                    <circle cx={kp.x} cy={kp.y} r={baseR * 0.75} fill={colors.fill}
                      opacity={0.25 + kp.confidence * 0.3} />
                  </g>
                );
              }

              // Tier 3: low confidence (0.05–0.3) — hollow ring
              return (
                <g key={`kp-${i}`}>
                  <circle cx={kp.x} cy={kp.y} r={baseR * 0.7} fill="none"
                    stroke={colors.stroke} strokeWidth={baseR * 0.25}
                    opacity={0.2 + kp.confidence * 0.3} />
                </g>
              );
            })}
          </svg>
        )}

        {/* Progress bar */}
        {totalFrames > 0 && (
          <div className="absolute bottom-2 left-3 right-3 h-1 rounded-full bg-slate-700/50 overflow-hidden">
            <div
              className={cn("h-full rounded-full transition-all duration-75", isPlaying
                ? "bg-gradient-to-r from-cyan-500 to-teal-400"
                : "bg-gradient-to-r from-slate-500 to-slate-400")}
              style={{ width: `${((clampedFrame + 1) / totalFrames) * 100}%` }}
            />
          </div>
        )}
      </div>

      {/* ── Legend with confidence tiers ── */}
      <div className="mt-3 space-y-2">
        {/* Confidence tier legend */}
        <div className="flex items-center gap-3 text-[9px] text-slate-400">
          <span className="font-semibold text-slate-300">置信度：</span>
          <span className="inline-flex items-center gap-1">
            <span className="h-1.5 w-1.5 rounded-full bg-white/60" />
            高(≥60%)
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="h-1.5 w-1.5 rounded-full bg-white/25" />
            中(30-60%)
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="h-1.5 w-1.5 rounded-full border border-white/20 bg-transparent" />
            低(&lt;30%)
          </span>
        </div>

        {/* Body part legend */}
        <div className="grid grid-cols-4 gap-2">
          {(["body", "left_hand", "right_hand", "face"] as const).map((key) => {
            const colors = BODY_PART_COLORS[key];
            const stats = groupStats[key];
            const barW = hasData ? Math.min(stats.avgConf * 100, 100) : 0;
            const labels: Record<string, string> = {
              body: "身体/足部",
              left_hand: "左手",
              right_hand: "右手",
              face: "面部",
            };
            return (
              <div key={key} className="rounded-lg border border-slate-700/20 bg-slate-800/30 p-2">
                <div className="flex items-center gap-1.5 mb-1">
                  <div className="h-2 w-2 rounded-full shadow" style={{ backgroundColor: colors.fill, boxShadow: `0 0 4px ${colors.fill}` }} />
                  <span className="text-[10px] font-semibold text-slate-300">{labels[key]}</span>
                </div>
                <div className="text-[9px] text-slate-400 space-y-0.5">
                  <div>{stats.count} 点 · 高{stats.highConf} / 低{stats.lowConf}</div>
                  {/* Confidence bar */}
                  <div className="h-1 rounded-full bg-slate-700/50 overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all"
                      style={{
                        width: `${barW}%`,
                        backgroundColor: colors.fill,
                        opacity: 0.7,
                      }}
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
    </div>
  );
}
