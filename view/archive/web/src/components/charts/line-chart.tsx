import {
  LineChart as ReLine,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  Legend,
} from "recharts";

interface LineData {
  name: string;
  [key: string]: number | string;
}

interface LineConfig {
  key: string;
  label: string;
  color: string;
}

interface Props {
  data: LineData[];
  lines: LineConfig[];
  title?: string;
  height?: number;
  xLabel?: string;
  yLabel?: string;
}

export function LineChart({
  data,
  lines,
  title,
  height = 340,
  xLabel,
  yLabel,
}: Props) {
  return (
    <div className="rounded-3xl border border-border bg-white p-5 shadow-lg dark:bg-slate-800">
      {title && (
        <h3 className="mb-2 text-base font-bold text-foreground">{title}</h3>
      )}
      <ResponsiveContainer width="100%" height={height}>
        <ReLine data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#94a3b8" strokeOpacity={0.2} />
          <XAxis
            dataKey="name"
            tick={{ fill: "#64748b", fontSize: 12 }}
            label={
              xLabel
                ? { value: xLabel, position: "insideBottom", offset: -5, fill: "#64748b" }
                : undefined
            }
          />
          <YAxis
            tick={{ fill: "#64748b", fontSize: 12 }}
            label={
              yLabel
                ? { value: yLabel, angle: -90, position: "insideLeft", fill: "#64748b" }
                : undefined
            }
          />
          <Legend />
          {lines.map((line) => (
            <Line
              key={line.key}
              type="monotone"
              dataKey={line.key}
              name={line.label}
              stroke={line.color}
              strokeWidth={2.5}
              dot={{ r: 3 }}
              activeDot={{ r: 6 }}
            />
          ))}
        </ReLine>
      </ResponsiveContainer>
    </div>
  );
}
