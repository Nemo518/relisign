import { cn } from "@/lib/utils";

interface MetricCardProps {
  title: string;
  value: string;
  note: string;
  className?: string;
}

export function MetricCard({ title, value, note, className }: MetricCardProps) {
  return (
    <div
      className={cn(
        "glass-card rounded-3xl p-5 min-h-[136px] flex flex-col justify-between transition-all duration-300 hover:-translate-y-0.5 hover:border-cyan-400/30",
        className,
      )}
    >
      <div className="text-xs font-semibold text-slate-300 uppercase tracking-wider">{title}</div>
      <div className="text-3xl font-extrabold text-white leading-tight font-mono tracking-tight">
        {value}
      </div>
      <div className="text-xs text-slate-300 leading-relaxed mt-1">
        {note}
      </div>
    </div>
  );
}
