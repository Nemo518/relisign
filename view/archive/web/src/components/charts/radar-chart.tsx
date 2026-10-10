import {
  RadarChart as ReRadar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
} from "recharts";

interface RadarData {
  name: string;
  value: number;
}

interface Props {
  data: RadarData[];
  title?: string;
}

export function RadarChart({ data, title }: Props) {
  return (
    <div className="rounded-3xl border border-border bg-white p-5 shadow-lg dark:bg-slate-800">
      {title && (
        <h3 className="mb-2 text-base font-bold text-foreground">{title}</h3>
      )}
      <ResponsiveContainer width="100%" height={340}>
        <ReRadar data={data} cx="50%" cy="50%" outerRadius="75%">
          <PolarGrid stroke="#94a3b8" strokeOpacity={0.3} />
          <PolarAngleAxis
            dataKey="name"
            tick={{ fill: "#64748b", fontSize: 13 }}
          />
          <PolarRadiusAxis
            domain={[0, 100]}
            tick={{ fill: "#94a3b8", fontSize: 11 }}
          />
          <Radar
            name="系统画像"
            dataKey="value"
            stroke="#2dd4bf"
            fill="#38bdf8"
            fillOpacity={0.18}
            strokeWidth={3}
          />
        </ReRadar>
      </ResponsiveContainer>
    </div>
  );
}
