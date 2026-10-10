import { PieChart, Pie, Cell, ResponsiveContainer } from "recharts";
import { cn } from "@/lib/utils";

interface Props {
  value: number; // 0-1
  title?: string;
}

export function GaugeChart({ value, title }: Props) {
  const pct = Math.round(value * 100);
  const qualityLabel =
    pct >= 75 ? "稳定" : pct >= 50 ? "可用" : "待优化";

  const data = [
    { name: "value", value: Math.max(pct, 1) },
    { name: "remaining", value: Math.max(100 - pct, 0) },
  ];

  const color =
    pct >= 75 ? "#2dd4bf" : pct >= 50 ? "#fbbf24" : "#fb7185";

  return (
    <div className="rounded-3xl border border-border bg-white p-5 shadow-lg dark:bg-slate-800">
      {title && (
        <h3 className="mb-2 text-base font-bold text-foreground">{title}</h3>
      )}
      <div className="relative">
        <ResponsiveContainer width="100%" height={240}>
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="100%"
              startAngle={180}
              endAngle={0}
              innerRadius="75%"
              outerRadius="95%"
              dataKey="value"
              stroke="none"
            >
              <Cell fill={color} />
              <Cell fill="#e2e8f0" />
            </Pie>
          </PieChart>
        </ResponsiveContainer>
        <div className="absolute bottom-1 left-1/2 -translate-x-1/2 text-center">
          <div className="text-3xl font-extrabold text-foreground">
            {pct}%
          </div>
          <div
            className={cn(
              "mt-1 text-sm font-semibold",
              pct >= 75
                ? "text-teal-500"
                : pct >= 50
                  ? "text-yellow-500"
                  : "text-red-500",
            )}
          >
            文本质量 · {qualityLabel}
          </div>
        </div>
      </div>
    </div>
  );
}
