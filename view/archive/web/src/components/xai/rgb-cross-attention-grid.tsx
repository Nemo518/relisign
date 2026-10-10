import { useMemo } from "react";
import { cn } from "@/lib/utils";
import { Layers } from "lucide-react";

interface RgbCrossRow {
  branch: string;
  patch_x: number;
  patch_y: number;
  attention: number;
}

interface Props {
  data: RgbCrossRow[];
  className?: string;
}

export function RgbCrossAttentionGrid({ data, className }: Props) {
  const branches = useMemo(() => {
    const left = data.filter((d) => d.branch === "left");
    const right = data.filter((d) => d.branch === "right");
    return { left, right };
  }, [data]);

  if (!data.length) return null;

  // Normalize across both branches for consistent coloring
  const allVals = data.map((d) => d.attention);
  const maxVal = Math.max(...allVals, 0.001);

  const renderGrid = (rows: RgbCrossRow[], label: string) => {
    const grid: number[][] = Array.from({ length: 4 }, () => Array(4).fill(0));
    for (const r of rows) {
      if (r.patch_x >= 0 && r.patch_x < 4 && r.patch_y >= 0 && r.patch_y < 4) {
        grid[r.patch_y][r.patch_x] = r.attention;
      }
    }

    return (
      <div className="flex-1">
        <div className="text-center text-[10px] text-slate-400 mb-1.5 font-semibold">{label}</div>
        <svg viewBox="0 0 4 4" className="w-full aspect-square rounded-lg overflow-hidden border border-slate-700/30">
          {grid.map((row, y) =>
            row.map((val, x) => {
              const t = Math.min(1, val / maxVal);
              const r = Math.round(10 + t * 200);
              const g = Math.round(10 + t * 180);
              const b = Math.round(40 + t * 160);
              return (
                <rect
                  key={`${y}-${x}`}
                  x={x}
                  y={y}
                  width={1}
                  height={1}
                  fill={`rgb(${r},${g},${b})`}
                  stroke="rgba(255,255,255,0.1)"
                  strokeWidth={0.05}
                />
              );
            }),
          )}
          {/* Overlay attention values */}
          {grid.map((row, y) =>
            row.map((val, x) => {
              const t = val / maxVal;
              return (
                <text
                  key={`t-${y}-${x}`}
                  x={x + 0.5}
                  y={y + 0.55}
                  textAnchor="middle"
                  fill={t > 0.5 ? "white" : "rgba(255,255,255,0.4)"}
                  fontSize={0.55}
                  fontWeight="bold"
                  fontFamily="monospace"
                >
                  {(val * 100).toFixed(0)}
                </text>
              );
            }),
          )}
        </svg>
      </div>
    );
  };

  return (
    <div className={cn("glass-card rounded-2xl p-4", className)}>
      <div className="flex items-center gap-2 mb-3">
        <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-gradient-to-br from-emerald-500 to-teal-500 text-white shadow">
          <Layers size={14} />
        </div>
        <div>
          <span className="text-sm font-bold">RGB 跨模态注意力</span>
          <span className="ml-2 text-[10px] text-emerald-400 font-mono">
            RGB CROSS-ATTENTION
          </span>
        </div>
      </div>

      <div className="flex gap-4">
        {renderGrid(branches.left, "左手 4×4 Patch")}
        {renderGrid(branches.right, "右手 4×4 Patch")}
      </div>

      <p className="mt-3 text-[10px] text-slate-400 leading-relaxed">
        姿态关键点作为 Query，在 RGB 特征图上通过可变形注意力采样 4×4
        空间网格。亮区 = 模型认为该区域对手语理解最重要。
      </p>
    </div>
  );
}
