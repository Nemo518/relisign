import type { VideoRecord, BestExampleRecord } from "@/types/api";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { queryKeys } from "@/api/query-keys";
import { useAppStore } from "@/store/app-store";

interface Props {
  value: VideoRecord | null;
  onChange: (v: VideoRecord | null) => void;
}

export function SampleSelector({ value, onChange }: Props) {
  const split = useAppStore((s) => s.split);

  const { data: bestExamples, isLoading } = useQuery({
    queryKey: queryKeys.bestExamples(split),
    queryFn: () => api.bestExamples(split),
  });

  if (isLoading) {
    return <div className="text-sm text-muted-foreground">加载最佳样例视频...</div>;
  }

  const records = bestExamples ?? [];

  if (!records.length) {
    return (
      <div className="text-sm text-muted-foreground">
        当前没有可用的样例视频，请检查外部数据仓中的 CE-CSL 视频是否已经就位。
      </div>
    );
  }

  return (
    <select
      value={value?.name ?? ""}
      onChange={(e) => {
        const selected = records.find((v) => v.name === e.target.value);
        onChange(selected ?? null);
      }}
      className="w-full rounded-xl border border-border bg-white px-3 py-2.5 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring dark:bg-slate-800"
    >
      <option value="" disabled>
        选择最佳样例（按相似度排序）...
      </option>
      {records.map((v) => {
        const sim = "similarity" in v ? (v as BestExampleRecord).similarity : null;
        const simLabel = sim != null ? ` [${(sim * 100).toFixed(1)}%]` : "";
        return (
          <option key={v.name} value={v.name}>
            {v.name}{simLabel}
          </option>
        );
      })}
    </select>
  );
}
