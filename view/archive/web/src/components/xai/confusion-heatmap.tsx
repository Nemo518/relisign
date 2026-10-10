import { useMemo } from "react";
import { cn } from "@/lib/utils";

interface ConfusionPair {
  reference_char: string;
  prediction_char: string;
  count: number;
  operation: string;
}

interface Props {
  data: ConfusionPair[];
  className?: string;
}

export function ConfusionHeatmap({ data, className }: Props) {
  const { chars, maxCount, grid } = useMemo(() => {
    if (!data.length) return { chars: [] as string[], maxCount: 0, grid: [] as number[][] };

    // Collect unique characters
    const charSet = new Set<string>();
    for (const d of data) {
      charSet.add(d.reference_char);
      charSet.add(d.prediction_char);
    }
    const chars = Array.from(charSet).sort();
    const maxCount = Math.max(...data.map((d) => d.count), 1);

    // Build NxN matrix
    const n = chars.length;
    const grid: number[][] = Array.from({ length: n }, () => Array(n).fill(0));
    for (const d of data) {
      const ri = chars.indexOf(d.reference_char);
      const ci = chars.indexOf(d.prediction_char);
      if (ri >= 0 && ci >= 0) {
        grid[ri][ci] = d.count;
      }
    }

    return { chars, maxCount, grid };
  }, [data]);

  const n = chars.length;
  if (n === 0) return null;

  const cellSize = Math.max(16, Math.min(28, 400 / n));
  const svgW = n * cellSize + 24;
  const svgH = n * cellSize + 24;

  return (
    <div className={cn("rounded-lg border border-slate-700/30 overflow-hidden bg-slate-900/60", className)}>
      <div className="overflow-auto max-h-64">
        <svg
          viewBox={`0 0 ${svgW} ${svgH}`}
          width={svgW}
          height={svgH}
          className="block"
        >
          {/* Column headers (prediction) */}
          {chars.map((c, i) => (
            <text
              key={`col-${i}`}
              x={12 + i * cellSize + cellSize / 2}
              y={10}
              textAnchor="middle"
              fill="#94a3b8"
              fontSize={Math.max(7, cellSize * 0.4)}
              fontFamily="monospace"
            >
              {c}
            </text>
          ))}
          {/* Row headers (reference) */}
          {chars.map((c, i) => (
            <text
              key={`row-${i}`}
              x={8}
              y={16 + i * cellSize + cellSize / 2}
              textAnchor="end"
              fill="#94a3b8"
              fontSize={Math.max(7, cellSize * 0.4)}
              fontFamily="monospace"
            >
              {c}
            </text>
          ))}

          {/* Heatmap cells */}
          {grid.map((row, y) =>
            row.map((val, x) => {
              const t = val / maxCount;
              const r = Math.round(t < 0.5 ? t * 2 * 100 : 200 + t * 55);
              const g = Math.round(t < 0.5 ? 0 : (t - 0.5) * 2 * 30);
              const b = Math.round(t < 0.3 ? 200 - t * 3.3 * 100 : 30);
              return (
                <rect
                  key={`${y}-${x}`}
                  x={12 + x * cellSize + 0.5}
                  y={12 + y * cellSize + 0.5}
                  width={cellSize - 1}
                  height={cellSize - 1}
                  fill={`rgb(${r},${g},${b})`}
                  opacity={0.2 + t * 0.8}
                  rx={2}
                />
              );
            }),
          )}

          {/* Value labels for non-zero cells */}
          {grid.map((row, y) =>
            row.map((val, x) => {
              if (val === 0) return null;
              const t = val / maxCount;
              return (
                <text
                  key={`v-${y}-${x}`}
                  x={12 + x * cellSize + cellSize / 2}
                  y={12 + y * cellSize + cellSize / 2 + 1}
                  textAnchor="middle"
                  dominantBaseline="central"
                  fill={t > 0.5 ? "white" : "rgba(255,255,255,0.5)"}
                  fontSize={Math.max(6, cellSize * 0.32)}
                  fontFamily="monospace"
                  fontWeight="bold"
                >
                  {val}
                </text>
              );
            }),
          )}
        </svg>
      </div>
      <div className="flex items-center justify-between px-3 py-1.5 border-t border-slate-700/30">
        <span className="text-[9px] text-slate-500">行=参考 列=预测</span>
        <div className="flex items-center gap-0.5">
          <span className="text-[8px] text-slate-500 mr-1">0</span>
          <div className="h-2 w-2 rounded-sm" style={{ background: "rgb(40,0,100)" }} />
          <div className="h-2 w-2 rounded-sm" style={{ background: "rgb(100,0,160)" }} />
          <div className="h-2 w-2 rounded-sm" style={{ background: "rgb(200,20,40)" }} />
          <div className="h-2 w-2 rounded-sm" style={{ background: "rgb(240,100,30)" }} />
          <span className="text-[8px] text-slate-500 ml-1">{maxCount}</span>
        </div>
      </div>
    </div>
  );
}
