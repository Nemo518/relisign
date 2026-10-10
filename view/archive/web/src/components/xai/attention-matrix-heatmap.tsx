import { useMemo } from "react";
import { cn } from "@/lib/utils";
import { Eye } from "lucide-react";

interface AttentionMatrixRow {
  source_frame: number;
  target_frame: number;
  attention: number;
}

interface Props {
  data: AttentionMatrixRow[];
  className?: string;
}

export function AttentionMatrixHeatmap({ data, className }: Props) {
  const { matrix, maxSrc, maxTgt } = useMemo(() => {
    if (!data.length) return { matrix: [] as number[][], maxSrc: 0, maxTgt: 0 };
    const srcMax = Math.max(...data.map((d) => d.source_frame));
    const tgtMax = Math.max(...data.map((d) => d.target_frame));
    // Resample to at most 28x28 for display
    const size = Math.min(28, Math.max(srcMax, tgtMax) + 1);
    const m: number[][] = Array.from({ length: size }, () => Array(size).fill(0));
    const counts: number[][] = Array.from({ length: size }, () => Array(size).fill(0));

    for (const row of data) {
      const si = Math.round((row.source_frame / Math.max(1, srcMax)) * (size - 1));
      const ti = Math.round((row.target_frame / Math.max(1, tgtMax)) * (size - 1));
      m[ti][si] += row.attention;
      counts[ti][si] += 1;
    }
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        if (counts[y][x] > 0) m[y][x] /= counts[y][x];
      }
    }
    return { matrix: m, maxSrc: srcMax, maxTgt: tgtMax };
  }, [data]);

  const size = matrix.length;
  if (size === 0) return null;

  const cellW = 100 / size;
  const cellH = 100 / size;

  return (
    <div className={cn("glass-card rounded-2xl p-4", className)}>
      <div className="flex items-center gap-2 mb-3">
        <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-gradient-to-br from-violet-500 to-purple-500 text-white shadow">
          <Eye size={14} />
        </div>
        <div>
          <span className="text-sm font-bold">编码器自注意力矩阵</span>
          <span className="ml-2 text-[10px] text-purple-400 font-mono">
            SELF-ATTENTION HEATMAP
          </span>
        </div>
      </div>

      <div className="relative rounded-xl bg-slate-900/60 border border-slate-700/30 overflow-hidden">
        {/* Axes labels */}
        <div className="absolute -left-0.5 top-0 bottom-0 flex items-center">
          <span className="text-[8px] text-slate-500 -rotate-90 whitespace-nowrap tracking-wider">
            目标帧
          </span>
        </div>

        <svg
          viewBox={`0 0 ${size} ${size}`}
          className="w-full aspect-square block"
          preserveAspectRatio="xMidYMid meet"
        >
          {matrix.map((row, y) =>
            row.map((val, x) => {
              // Viridis-like colormap: dark blue → teal → green → yellow
              const t = Math.min(1, Math.max(0, val));
              const r = Math.round(t < 0.5 ? t * 2 * 64 : 200 + t * 55);
              const g = Math.round(t < 0.5 ? 0 : (t - 0.5) * 2 * 160 + 30);
              const b = Math.round(t < 0.3 ? 120 + t * 3.3 * 90 : t < 0.7 ? 180 - (t - 0.3) * 450 : 30);
              const color = `rgb(${Math.round(r)},${Math.round(g)},${Math.round(b)})`;
              return (
                <rect
                  key={`${y}-${x}`}
                  x={x}
                  y={y}
                  width={1}
                  height={1}
                  fill={color}
                  opacity={0.3 + t * 0.7}
                />
              );
            }),
          )}
          {/* Diagonal reference line — strong self-attention along diagonal is expected */}
          <line
            x1={0}
            y1={0}
            x2={size}
            y2={size}
            stroke="white"
            strokeWidth={0.15}
            strokeDasharray="1 2"
            opacity={0.2}
          />
        </svg>

        <div className="absolute bottom-1 left-8 right-2 flex items-center justify-between">
          <span className="text-[8px] text-slate-500">源帧 0 → {maxSrc}</span>
          {/* Color legend */}
          <div className="flex items-center gap-0.5">
            <span className="text-[7px] text-slate-500 mr-1">低</span>
            <div className="h-2 w-2 rounded-sm" style={{ background: "rgb(40,20,100)" }} />
            <div className="h-2 w-2 rounded-sm" style={{ background: "rgb(30,60,140)" }} />
            <div className="h-2 w-2 rounded-sm" style={{ background: "rgb(20,130,110)" }} />
            <div className="h-2 w-2 rounded-sm" style={{ background: "rgb(120,180,40)" }} />
            <div className="h-2 w-2 rounded-sm" style={{ background: "rgb(240,230,30)" }} />
            <span className="text-[7px] text-slate-500 ml-1">高</span>
          </div>
        </div>
      </div>

      <p className="mt-2 text-[10px] text-slate-400 leading-relaxed">
        对角线亮区 = 相邻帧局部注意力，非对角线亮区 = 长程跨帧依赖。
        手语动作的时序上下文越丰富，非对角线响应越显著。
      </p>
    </div>
  );
}
