import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Cell } from "recharts";

interface Props {
  data: { name: string; value: number; color: string }[];
  title?: string;
}

export function FunnelChart({ data, title }: Props) {
  return (
    <div className="rounded-3xl border border-border bg-white p-5 shadow-lg dark:bg-slate-800">
      {title && (
        <h3 className="mb-2 text-base font-bold text-foreground">{title}</h3>
      )}
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={data} layout="vertical" margin={{ left: 50, right: 20 }}>
          <XAxis type="number" tick={{ fontSize: 12 }} />
          <YAxis
            type="category"
            dataKey="name"
            tick={{ fontSize: 13 }}
            width={80}
          />
          <Bar dataKey="value" radius={[0, 8, 8, 0]}>
            {data.map((entry, i) => (
              <Cell key={i} fill={entry.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
