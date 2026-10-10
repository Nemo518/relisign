import { cn } from "@/lib/utils";

interface Token {
  type: "good" | "bad" | "gap";
  label: string;
}

function computeAlignment(
  reference: string,
  prediction: string,
): { good: Token[]; issues: Token[] } {
  const good: Token[] = [];
  const issues: Token[] = [];

  if (!reference && !prediction) return { good, issues };

  // Simple character-level diff: use LCS to align
  const refChars = Array.from(reference);
  const predChars = Array.from(prediction);

  // Compute LCS table
  const m = refChars.length;
  const n = predChars.length;
  const dp: number[][] = Array.from({ length: m + 1 }, () =>
    Array(n + 1).fill(0),
  );

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (refChars[i - 1] === predChars[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1] + 1;
      } else {
        dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
      }
    }
  }

  // Backtrack
  const ops: { type: string; ref?: string; pred?: string }[] = [];
  let i = m;
  let j = n;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && refChars[i - 1] === predChars[j - 1]) {
      ops.unshift({ type: "match", ref: refChars[i - 1], pred: predChars[j - 1] });
      i--;
      j--;
    } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
      ops.unshift({ type: "insert", pred: predChars[j - 1] });
      j--;
    } else if (i > 0 && (j === 0 || dp[i][j - 1] < dp[i - 1][j])) {
      ops.unshift({ type: "delete", ref: refChars[i - 1] });
      i--;
    } else {
      if (i > 0 && j > 0) {
        ops.unshift({ type: "replace", ref: refChars[i - 1], pred: predChars[j - 1] });
        i--;
        j--;
      } else if (i > 0) {
        ops.unshift({ type: "delete", ref: refChars[i - 1] });
        i--;
      } else {
        ops.unshift({ type: "insert", pred: predChars[j - 1] });
        j--;
      }
    }
  }

  for (const op of ops) {
    if (op.type === "match") {
      good.push({ type: "good", label: op.pred ?? "" });
    } else if (op.type === "insert") {
      issues.push({ type: "bad", label: `+ ${op.pred ?? ""}` });
    } else if (op.type === "delete") {
      issues.push({ type: "gap", label: `缺 ${op.ref ?? ""}` });
    } else {
      issues.push({
        type: "bad",
        label: `${op.ref ?? ""} → ${op.pred ?? ""}`,
      });
    }
  }

  return { good: good.slice(0, 18), issues: issues.slice(0, 18) };
}

interface Props {
  reference: string;
  prediction: string;
}

export function AlignmentTokens({ reference, prediction }: Props) {
  const { good, issues } = computeAlignment(reference, prediction);

  return (
    <div className="grid grid-cols-2 gap-4">
      <div className="rounded-2xl border border-border bg-white p-4 dark:bg-slate-800">
        <div className="mb-2 text-xs text-muted-foreground">表现稳定片段</div>
        <div className="flex flex-wrap gap-1">
          {good.length > 0 ? (
            good.map((t, i) => (
              <span
                key={i}
                className={cn(
                  "inline-block rounded-full px-2 py-0.5 text-xs font-semibold",
                  "bg-teal-100 text-teal-700 dark:bg-teal-900 dark:text-teal-300",
                )}
              >
                {t.label || "空"}
              </span>
            ))
          ) : (
            <span className="text-xs text-muted-foreground">暂无</span>
          )}
        </div>
      </div>
      <div className="rounded-2xl border border-border bg-white p-4 dark:bg-slate-800">
        <div className="mb-2 text-xs text-muted-foreground">需要关注片段</div>
        <div className="flex flex-wrap gap-1">
          {issues.length > 0 ? (
            issues.map((t, i) => (
              <span
                key={i}
                className={cn(
                  "inline-block rounded-full px-2 py-0.5 text-xs font-semibold",
                  t.type === "bad"
                    ? "bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300"
                    : "bg-slate-200 text-slate-600 dark:bg-slate-700 dark:text-slate-400",
                )}
              >
                {t.label}
              </span>
            ))
          ) : (
            <span
              className={cn(
                "inline-block rounded-full px-2 py-0.5 text-xs font-semibold",
                "bg-teal-100 text-teal-700",
              )}
            >
              无明显问题
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
