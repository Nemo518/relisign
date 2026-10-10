import {
  BarChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  Legend,
  ComposedChart,
} from "recharts";

interface ComboData {
  model: string;
  bleu4: number;
  rougeL: number;
  werAdvantage: number;
}

interface Props {
  data: ComboData[];
  title?: string;
}

export function ComboChart({ data, title }: Props) {
  return (
    <div className="rounded-3xl border border-border bg-white p-5 shadow-lg dark:bg-slate-800">
      {title && (
        <h3 className="mb-2 text-base font-bold text-foreground">{title}</h3>
      )}
      <ResponsiveContainer width="100%" height={340}>
        <ComposedChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#94a3b8" strokeOpacity={0.2} />
          <XAxis
            dataKey="model"
            tick={{ fill: "#64748b", fontSize: 12 }}
            stroke="#94a3b8"
            strokeOpacity={0.4}
          />
          <YAxis
            yAxisId="left"
            tick={{ fill: "#64748b", fontSize: 12 }}
            stroke="#94a3b8"
            strokeOpacity={0.4}
            label={{ value: "BLEU / ROUGE", angle: -90, position: "insideLeft", fill: "#64748b" }}
          />
          <YAxis
            yAxisId="right"
            orientation="right"
            tick={{ fill: "#64748b", fontSize: 12 }}
            stroke="#94a3b8"
            strokeOpacity={0}
            label={{ value: "Advantage", angle: 90, position: "insideRight", fill: "#64748b" }}
          />
          <Legend />
          <Bar yAxisId="left" dataKey="bleu4" name="BLEU-4" fill="#38bdf8" radius={[6, 6, 0, 0]} />
          <Bar yAxisId="left" dataKey="rougeL" name="ROUGE-L" fill="#14b8a6" radius={[6, 6, 0, 0]} />
          <Line
            yAxisId="right"
            dataKey="werAdvantage"
            name="WER Advantage"
            stroke="#fbbf24"
            strokeWidth={3}
            dot={{ fill: "#fbbf24", r: 5 }}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
