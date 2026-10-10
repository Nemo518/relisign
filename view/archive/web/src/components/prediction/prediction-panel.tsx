interface Props {
  reference: string;
  prediction: string;
}

export function PredictionPanel({ reference, prediction }: Props) {
  return (
    <div className="space-y-3">
      <div className="rounded-2xl border border-border bg-gradient-to-b from-white to-blue-50/50 p-4 dark:from-slate-800 dark:to-slate-800/50">
        <div className="mb-1 text-xs text-muted-foreground">参考文本</div>
        <div className="text-base font-semibold text-foreground">
          {reference || "暂无"}
        </div>
      </div>
      <div className="rounded-2xl border border-border bg-gradient-to-b from-white to-blue-50/50 p-4 dark:from-slate-800 dark:to-slate-800/50">
        <div className="mb-1 text-xs text-muted-foreground">预测文本</div>
        <div className="text-base font-semibold text-foreground">
          {prediction || "暂无"}
        </div>
      </div>
    </div>
  );
}
