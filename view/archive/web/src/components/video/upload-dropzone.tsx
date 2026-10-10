import { useCallback, useState } from "react";
import { cn } from "@/lib/utils";
import { Upload } from "lucide-react";

interface Props {
  value: File | null;
  onChange: (f: File | null) => void;
}

export function UploadDropzone({ value, onChange }: Props) {
  const [dragging, setDragging] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith("video/")) {
        onChange(file);
      }
    },
    [onChange],
  );

  return (
    <div>
      <input
        type="file"
        accept="video/mp4,video/avi,video/mov,video/x-matroska"
        onChange={(e) => onChange(e.target.files?.[0] ?? null)}
        className="w-full text-sm text-foreground file:mr-3 file:rounded-xl file:border-0 file:bg-cyan-400 file:px-3 file:py-1.5 file:text-sm file:font-medium file:text-slate-900"
      />
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        className={cn(
          "mt-2 rounded-2xl border-2 border-dashed p-6 text-center text-sm text-muted-foreground transition-colors",
          dragging
            ? "border-cyan-400 bg-cyan-50 dark:bg-cyan-950"
            : "border-border",
        )}
      >
        <Upload className="mx-auto mb-2 h-6 w-6 opacity-40" />
        拖放视频文件到此处（MP4/AVI/MOV/MKV）
      </div>
      {value && (
        <p className="mt-2 text-sm text-foreground">
          已选：{value.name}（{(value.size / 1024 / 1024).toFixed(1)} MB）
        </p>
      )}
    </div>
  );
}
