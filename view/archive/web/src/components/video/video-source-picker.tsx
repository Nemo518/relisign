import { cn } from "@/lib/utils";

interface Props {
  value: "sample" | "upload" | "camera";
  onChange: (v: "sample" | "upload" | "camera") => void;
}

const options = [
  { value: "sample" as const, label: "样例视频" },
  { value: "upload" as const, label: "本地上传" },
  { value: "camera" as const, label: "摄像头采集" },
];

export function VideoSourcePicker({ value, onChange }: Props) {
  return (
    <div className="flex gap-2">
      {options.map((opt) => (
        <button
          key={opt.value}
          onClick={() => onChange(opt.value)}
          className={cn(
            "flex-1 rounded-xl px-4 py-2 text-sm font-medium transition-all",
            value === opt.value
              ? "bg-gradient-to-r from-cyan-400 to-teal-400 text-slate-900 shadow-md"
              : "border border-border bg-white text-foreground hover:bg-muted dark:bg-slate-800",
          )}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}
