import {
  BarChart as ReBar,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  Legend,
} from "recharts";

interface BarConfig {
  key: string;
  label: string;
  color: string;
}

interface BarData {
  name: string;
  [key: string]: number | string;
}

interface Props {
  data: BarData[];
  bars: BarConfig[];
  title?: string;
  height?: number;
  horizontal?: boolean;
}

export function BarChart({
  data,
  bars,
  title,
  height = 340,
  horizontal = false,
}: Props) {
  return (
    <div className="rounded-3xl border border-border bg-white p-5 shadow-lg dark:bg-slate-800">
      {title && (
        <h3 className="mb-2 text-base font-bold text-foreground">{title}</h3>
      )}
      <ResponsiveContainer width="100%" height={height}>
        <ReBar
          data={data}
          layout={horizontal ? "vertical" : "horizontal"}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#94a3b8" strokeOpacity={0.2} />
          {horizontal ? (
            <>
              <XAxis type="number" tick={{ fill: "#64748b", fontSize: 12 }} />
              <YAxis
                dataKey="name"
                type="category"
                tick={{ fill: "#64748b", fontSize: 12 }}
                width={120}
              />
            </>
          ) : (
            <>
              <XAxis dataKey="name" tick={{ fill: "#64748b", fontSize: 12 }} />
              <YAxis tick={{ fill: "#64748b", fontSize: 12 }} />
            </>
          )}
          <Legend />
          {bars.map((bar) => (
            <Bar
              key={bar.key}
              dataKey={bar.key}
              name={bar.label}
              fill={bar.color}
              radius={[6, 6, 0, 0]}
            />
          ))}
        </ReBar>
      </ResponsiveContainer>
    </div>
  );
}
