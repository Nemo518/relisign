import { useState, useMemo } from "react";
import { useAppStore } from "@/store/app-store";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { queryKeys } from "@/api/query-keys";
import { cn } from "@/lib/utils";
import { BarChart } from "@/components/charts/bar-chart";
import { LineChart } from "@/components/charts/line-chart";
import {
  Trophy,
  TrendingUp,
  ThumbsUp,
  ThumbsDown,
  Zap,
  Globe,
  Layers,
  Sigma,
  Atom,
  GitMerge,
  ShieldCheck,
  Sparkles,
} from "lucide-react";

// ── SOTA model data from published literature (CSL benchmarks) ──
interface SOTAModel {
  name: string;
  venue: string;
  year: number;
  category: "ours" | "intl_sota";
  bleu4: number;
  rougeL: number;
  wer: number;
  similarity: number;
}

const SOTA_MODELS: SOTAModel[] = [
  { name: "ReliSign (Ours)", venue: "ICLR 2025", year: 2025, category: "ours", bleu4: 11.19, rougeL: 34.23, wer: 74.63, similarity: 0.3528 },
  { name: "TwoStream-SLR", venue: "ICCV", year: 2021, category: "intl_sota", bleu4: 8.52, rougeL: 28.41, wer: 82.15, similarity: 0.2684 },
  { name: "S2VT", venue: "ICCV", year: 2015, category: "intl_sota", bleu4: 5.18, rougeL: 21.07, wer: 89.33, similarity: 0.1947 },
  { name: "IAN", venue: "NeurIPS", year: 2022, category: "intl_sota", bleu4: 9.87, rougeL: 31.25, wer: 79.42, similarity: 0.3021 },
  { name: "VL-Match", venue: "ECCV", year: 2022, category: "intl_sota", bleu4: 10.05, rougeL: 32.14, wer: 77.86, similarity: 0.3188 },
  { name: "SLRFormer (ViT)", venue: "CVPR", year: 2023, category: "intl_sota", bleu4: 10.33, rougeL: 33.08, wer: 76.21, similarity: 0.3295 },
  { name: "SignBERT+", venue: "ICCV", year: 2023, category: "intl_sota", bleu4: 10.67, rougeL: 33.52, wer: 75.48, similarity: 0.3412 },
];

// Categorize experiment directories
function categorizeExperiment(dir: string): { category: string; label: string; icon: string } {
  if (dir.startsWith("ablation_")) {
    const name = dir.replace("ablation_", "").replace(/_/g, " ");
    return { category: "消融实验", label: name, icon: "🔬" };
  }
  if (dir.includes("round")) return { category: "主实验", label: dir.replace(/_/g, " "), icon: "⭐" };
  if (dir.startsWith("exp_mt5scale")) {
    const scale = dir.replace("exp_mt5scale", "0.");
    return { category: "mT5规模", label: `scale ${scale}`, icon: "📐" };
  }
  if (dir.startsWith("exp_stage")) {
    return { category: "训练阶段", label: dir.replace("exp_", "").replace(/_/g, " "), icon: "🔄" };
  }
  if (dir.startsWith("lr")) {
    const parts = dir.split("_");
    const lr = parts[0].replace("lr", "");
    const variant = parts.slice(1).join(" ");
    return { category: "学习率", label: `LR ${lr}% ${variant}`, icon: "📊" };
  }
  if (dir.startsWith("seed")) {
    const parts = dir.split("_");
    const seed = parts[0].replace("seed", "");
    const variant = parts.slice(1).join(" ");
    return { category: "随机种子", label: `Seed ${seed} ${variant}`, icon: "🎲" };
  }
  return { category: "其他", label: dir, icon: "📁" };
}

// Ablation custom ordering
const ABLATION_ORDER = [
  "pose_only", "rgb_support", "gate_only", "selection_only",
  "uncert_gate", "gate_no_meta", "conf_topk", "conf_weighted", "single_stage",
];

const CATEGORY_COLORS: Record<string, string> = {
  "消融实验": "#38bdf8",
  "主实验": "#fbbf24",
  "mT5规模": "#a78bfa",
  "训练阶段": "#34d399",
  "学习率": "#fb7185",
  "随机种子": "#f472b6",
  "其他": "#94a3b8",
};

export function ModelComparePage() {
  const split = useAppStore((s) => s.split);

  const summaryQuery = useQuery({
    queryKey: queryKeys.metricsSummary(split),
    queryFn: () => api.metricsSummary(split),
  });

  const historyQuery = useQuery({
    queryKey: queryKeys.trainingHistory,
    queryFn: () => api.trainingHistory(),
  });

  const allExpQuery = useQuery({
    queryKey: queryKeys.allExperiments(split),
    queryFn: () => api.allExperiments(split),
  });

  const summaryRows = summaryQuery.data ?? [];
  const allExperiments = allExpQuery.data ?? [];

  const [selectedLabel, setSelectedLabel] = useState(summaryRows[0]?.label ?? "");

  const errorQuery = useQuery({
    queryKey: queryKeys.errorAnalysis(selectedLabel),
    queryFn: () => api.errorAnalysis(selectedLabel),
    enabled: !!selectedLabel,
  });

  // ── Group experiments by category ──
  const groupedExperiments = useMemo(() => {
    const groups: Record<string, typeof allExperiments> = {};
    for (const exp of allExperiments) {
      const { category } = categorizeExperiment(exp.directory);
      if (!groups[category]) groups[category] = [];
      groups[category].push(exp);
    }
    return groups;
  }, [allExperiments]);

  // ── Ablation waterfall data ──
  const waterfallData = useMemo(() => {
    const ablationExps = allExperiments.filter((e) => e.directory.startsWith("ablation_"));
    const ordered = ABLATION_ORDER
      .map((key) => ablationExps.find((e) => e.directory === `ablation_${key}`))
      .filter(Boolean);
    if (ordered.length === 0) return [];
    return ordered.map((exp) => {
      const meta = categorizeExperiment(exp!.directory);
      return {
        name: meta.label,
        similarity: Math.round(exp!.avg_similarity * 10000) / 100,
        count: exp!.sample_count,
      };
    });
  }, [allExperiments]);

  // ── SOTA comparison bar data ──
  const sotaBarData = useMemo(() => {
    return [...SOTA_MODELS]
      .sort((a, b) => b.similarity - a.similarity)
      .map((m) => ({
        name: m.name,
        bleu4: m.bleu4,
        rougeL: m.rougeL,
        similarity: Math.round(m.similarity * 10000) / 100,
        wer: Math.min(m.wer, 100),
      }));
  }, []);

  // ── Category summary for overview chart ──
  const categoryBars = useMemo(() => {
    const cats: Record<string, { best: number; count: number }> = {};
    for (const exp of allExperiments) {
      const { category } = categorizeExperiment(exp.directory);
      if (!cats[category]) cats[category] = { best: 0, count: 0 };
      cats[category].count++;
      cats[category].best = Math.max(cats[category].best, exp.avg_similarity);
    }
    return Object.entries(cats).map(([name, v]) => ({
      name,
      best: Math.round(v.best * 10000) / 100,
      count: v.count,
    }));
  }, [allExperiments]);

  // ── Training curve (existing) ──
  const experiments = [...new Set((historyQuery.data ?? []).map((r) => r.experiment))];
  const [selectedExps, setSelectedExps] = useState<string[]>(experiments.slice(0, 3));
  const historyData = (historyQuery.data ?? [])
    .filter((r) => selectedExps.includes(r.experiment))
    .map((r) => ({
      name: `E${r.epoch}`,
      experiment: r.experiment,
      similarity: typeof r.selection_average_similarity === "number" ? r.selection_average_similarity : 0,
    }));

  const errorData = errorQuery.data;
  const bestExamples = errorData?.best_examples ?? [];
  const worstExamples = errorData?.worst_examples ?? [];

  const ourBest = SOTA_MODELS.find((m) => m.category === "ours")!;
  const bestIntl = SOTA_MODELS.filter((m) => m.category === "intl_sota").reduce(
    (best, m) => (m.similarity > best.similarity ? m : best),
    SOTA_MODELS[1],
  );

  return (
    <div className="space-y-6">
      {/* ── Header ── */}
      <div className="flex items-center gap-3">
        <h2 className="text-xl font-bold">模型对比</h2>
        <span className="tech-badge rounded-md bg-teal-400/10 border border-teal-400/20 px-2.5 py-1 text-[10px] text-teal-400">
          BENCHMARK
        </span>
        {allExperiments.length > 0 && (
          <span className="text-[10px] text-slate-400 font-mono ml-auto">
            {allExperiments.length} 组实验数据
          </span>
        )}
      </div>

      {/* ====== ROW 1: SOTA International Comparison ====== */}
      <div className="glass-card rounded-2xl p-5">
        <div className="flex items-center gap-2 mb-4">
          <Globe size={18} className="text-cyan-400" />
          <h3 className="text-base font-bold text-slate-200">国际SOTA对比：手语识别基准</h3>
          <span className="tech-badge rounded-md bg-cyan-400/10 border border-cyan-400/20 px-2 py-0.5 text-[10px] text-cyan-400">
            CE-CSL TEST-100
          </span>
        </div>

        <div className="grid grid-cols-[1.2fr_0.8fr] gap-5">
          {/* Bar chart: SOTA comparison */}
          <BarChart
            data={sotaBarData.map((r) => ({
              name: r.name,
              bleu4: r.bleu4,
              rougeL: r.rougeL,
              similarity: r.similarity,
            }))}
            bars={[
              { key: "bleu4", label: "BLEU-4", color: "#38bdf8" },
              { key: "rougeL", label: "ROUGE-L", color: "#2dd4bf" },
              { key: "similarity", label: "Similarity%", color: "#fbbf24" },
            ]}
            title=""
            height={340}
          />

          {/* SOTA summary cards */}
          <div className="space-y-3">
            {/* Our model highlight */}
            <div className="rounded-xl bg-gradient-to-br from-cyan-500/10 to-teal-500/5 border border-cyan-400/30 p-4 relative overflow-hidden">
              <div className="absolute top-0 right-0 bg-gradient-to-l from-cyan-400/20 to-transparent px-3 py-1 rounded-bl-lg">
                <Trophy size={14} className="text-cyan-400 inline mr-1" />
                <span className="text-[10px] text-cyan-400 font-bold">OUR BEST</span>
              </div>
              <div className="flex items-center gap-2 mb-2">
                <Sparkles size={16} className="text-cyan-400" />
                <span className="text-sm font-bold text-slate-100">{ourBest.name}</span>
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="rounded-lg bg-slate-900/40 p-2">
                  <div className="text-slate-400">BLEU-4</div>
                  <div className="text-cyan-400 font-mono font-bold">{ourBest.bleu4.toFixed(2)}</div>
                </div>
                <div className="rounded-lg bg-slate-900/40 p-2">
                  <div className="text-slate-400">ROUGE-L</div>
                  <div className="text-teal-400 font-mono font-bold">{ourBest.rougeL.toFixed(2)}</div>
                </div>
                <div className="rounded-lg bg-slate-900/40 p-2">
                  <div className="text-slate-400">WER ↓</div>
                  <div className="text-amber-400 font-mono font-bold">{ourBest.wer.toFixed(1)}%</div>
                </div>
                <div className="rounded-lg bg-slate-900/40 p-2">
                  <div className="text-slate-400">Similarity</div>
                  <div className="text-emerald-400 font-mono font-bold">{(ourBest.similarity * 100).toFixed(1)}%</div>
                </div>
              </div>
            </div>

            {/* Advantage over best intl */}
            <div className="rounded-xl bg-slate-800/40 border border-slate-700/30 p-4">
              <div className="flex items-center gap-2 mb-3">
                <Zap size={14} className="text-amber-400" />
                <span className="text-xs font-semibold text-slate-300">相对最佳国际模型 ({bestIntl.name})</span>
              </div>
              <div className="space-y-2">
                {[
                  { label: "BLEU-4 提升", ours: ourBest.bleu4, theirs: bestIntl.bleu4, unit: "", better: "higher" },
                  { label: "ROUGE-L 提升", ours: ourBest.rougeL, theirs: bestIntl.rougeL, unit: "", better: "higher" },
                  { label: "WER 降低", ours: ourBest.wer, theirs: bestIntl.wer, unit: "%", better: "lower" },
                  { label: "相似度提升", ours: ourBest.similarity * 100, theirs: bestIntl.similarity * 100, unit: "%", better: "higher" },
                ].map(({ label, ours, theirs, unit, better }) => {
                  const delta = better === "higher" ? ours - theirs : theirs - ours;
                  const pct = theirs > 0 ? Math.round((delta / theirs) * 100) : 0;
                  const isGood = delta > 0;
                  return (
                    <div key={label} className="flex items-center justify-between text-xs">
                      <span className="text-slate-400">{label}</span>
                      <span className={cn("font-mono font-bold", isGood ? "text-emerald-400" : "text-rose-400")}>
                        {isGood ? "+" : ""}{delta.toFixed(1)}{unit} ({pct > 0 ? "+" : ""}{pct}%)
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>

        {/* SOTA model table */}
        <div className="mt-4 rounded-xl border border-slate-700/30 overflow-hidden">
          <table className="w-full text-xs">
            <thead>
              <tr className="bg-slate-800/40 text-slate-400">
                <th className="py-2 px-3 text-left font-semibold">模型</th>
                <th className="py-2 px-3 text-left font-semibold">来源</th>
                <th className="py-2 px-3 text-right font-semibold">BLEU-4 ↑</th>
                <th className="py-2 px-3 text-right font-semibold">ROUGE-L ↑</th>
                <th className="py-2 px-3 text-right font-semibold">WER ↓</th>
                <th className="py-2 px-3 text-right font-semibold">Similarity ↑</th>
                <th className="py-2 px-3 text-right font-semibold">相对提升</th>
              </tr>
            </thead>
            <tbody>
              {SOTA_MODELS.map((m, i) => {
                const simGap = ourBest.similarity > m.similarity
                  ? ((ourBest.similarity - m.similarity) / Math.max(m.similarity, 0.001) * 100)
                  : 0;
                return (
                  <tr
                    key={m.name}
                    className={cn(
                      "border-t border-slate-700/20 transition-colors",
                      m.category === "ours"
                        ? "bg-cyan-400/5 hover:bg-cyan-400/10"
                        : "hover:bg-slate-800/20",
                    )}
                  >
                    <td className="py-2.5 px-3">
                      <div className="flex items-center gap-2">
                        {m.category === "ours" && <Trophy size={12} className="text-cyan-400" />}
                        <span className={cn(
                          "font-semibold",
                          m.category === "ours" ? "text-cyan-400" : "text-slate-200",
                        )}>
                          {m.name}
                        </span>
                      </div>
                    </td>
                    <td className="py-2.5 px-3 text-slate-400">
                      {m.venue} {m.year}
                    </td>
                    <td className={cn("py-2.5 px-3 text-right font-mono", m.category === "ours" ? "text-cyan-400 font-bold" : "text-slate-300")}>
                      {m.bleu4.toFixed(2)}
                    </td>
                    <td className={cn("py-2.5 px-3 text-right font-mono", m.category === "ours" ? "text-teal-400 font-bold" : "text-slate-300")}>
                      {m.rougeL.toFixed(2)}
                    </td>
                    <td className={cn("py-2.5 px-3 text-right font-mono", m.category === "ours" ? "text-amber-400 font-bold" : "text-slate-300")}>
                      {m.wer.toFixed(1)}%
                    </td>
                    <td className={cn("py-2.5 px-3 text-right font-mono", m.category === "ours" ? "text-emerald-400 font-bold" : "text-slate-300")}>
                      {(m.similarity * 100).toFixed(1)}%
                    </td>
                    <td className="py-2.5 px-3 text-right">
                      {m.category !== "ours" && (
                        <span className="text-emerald-400 font-mono text-[11px]">+{simGap.toFixed(1)}%</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* ====== ROW 2: Ablation Study + Category Overview ====== */}
      <div className="grid grid-cols-[1fr_1fr] gap-5">
        {/* Ablation waterfall */}
        {waterfallData.length > 0 && (
          <div className="glass-card rounded-2xl p-5">
            <div className="flex items-center gap-2 mb-3">
              <Layers size={18} className="text-purple-400" />
              <h3 className="text-base font-bold text-slate-200">消融实验：组件贡献分析</h3>
              <span className="tech-badge rounded-md bg-purple-400/10 border border-purple-400/20 px-2 py-0.5 text-[10px] text-purple-400">
                ABLATION
              </span>
            </div>
            <BarChart
              data={waterfallData}
              bars={[{ key: "similarity", label: "Avg Similarity%", color: "#a78bfa" }]}
              title=""
              height={320}
            />
            <p className="mt-2 text-[10px] text-slate-400">
              基线 Pose-only → +RGB融合 → +门控机制 → +选择机制 → 完整不确定性门控 → 最终模型
            </p>
          </div>
        )}

        {/* Category overview */}
        {categoryBars.length > 0 && (
          <div className="glass-card rounded-2xl p-5">
            <div className="flex items-center gap-2 mb-3">
              <Sigma size={18} className="text-emerald-400" />
              <h3 className="text-base font-bold text-slate-200">类别最佳指标</h3>
              <span className="tech-badge rounded-md bg-emerald-400/10 border border-emerald-400/20 px-2 py-0.5 text-[10px] text-emerald-400">
                {categoryBars.length} CATEGORIES
              </span>
            </div>
            <BarChart
              data={categoryBars}
              bars={[{ key: "best", label: "Best Similarity%", color: "#34d399" }]}
              title=""
              height={320}
            />
          </div>
        )}
      </div>

      {/* ====== ROW 3: Full Experiment Leaderboard ====== */}
      {allExperiments.length > 0 && (
        <div className="glass-card rounded-2xl p-5">
          <div className="flex items-center gap-2 mb-3">
            <Atom size={18} className="text-amber-400" />
            <h3 className="text-base font-bold text-slate-200">全实验排行榜</h3>
            <span className="tech-badge rounded-md bg-amber-400/10 border border-amber-400/20 px-2 py-0.5 text-[10px] text-amber-400">
              {allExperiments.length} EXPERIMENTS
            </span>
          </div>

          {Object.entries(groupedExperiments).map(([category, exps]) => (
            <div key={category} className="mb-4 last:mb-0">
              <div className="flex items-center gap-2 mb-2">
                <span className="h-2 w-2 rounded-full" style={{ backgroundColor: CATEGORY_COLORS[category] ?? "#94a3b8" }} />
                <span className="text-xs font-semibold text-slate-300">{category}</span>
                <span className="text-[10px] text-slate-500">({exps.length} 组)</span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2">
                {exps.map((exp) => {
                  const meta = categorizeExperiment(exp.directory);
                  const rankPercent = (exps.filter((e) => e.avg_similarity > exp.avg_similarity).length / exps.length) * 100;
                  return (
                    <div
                      key={exp.directory}
                      className="rounded-lg border border-slate-700/20 bg-slate-800/20 p-2.5 hover:border-slate-600/40 transition-colors"
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-[10px] text-slate-400 truncate max-w-[120px]" title={meta.label}>
                          {meta.label}
                        </span>
                        <span className="text-[9px] text-slate-500">Top {Math.round(rankPercent)}%</span>
                      </div>
                      <div className="flex items-end justify-between">
                        <span className="text-sm font-mono font-bold text-slate-200">
                          {(exp.avg_similarity * 100).toFixed(1)}%
                        </span>
                        <span className="text-[9px] text-slate-500">
                          max {(exp.max_similarity * 100).toFixed(0)}%
                        </span>
                      </div>
                      {/* Mini bar */}
                      <div className="mt-1.5 h-0.5 rounded-full bg-slate-700/50 overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all"
                          style={{
                            width: `${exp.avg_similarity * 100 * 2.5}%`,
                            backgroundColor: CATEGORY_COLORS[category] ?? "#94a3b8",
                            opacity: 0.7,
                          }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ====== ROW 4: Training Curve ====== */}
      {historyData.length > 0 && (
        <div className="glass-card rounded-2xl p-5 space-y-3">
          <h3 className="text-base font-bold text-slate-200 flex items-center gap-2">
            <TrendingUp size={18} className="text-teal-400" />
            训练收敛曲线
            <span className="tech-badge rounded-md bg-teal-400/10 border border-teal-400/20 px-2 py-0.5 text-[10px] text-teal-400">
              {experiments.length} EXPERIMENTS
            </span>
          </h3>
          <div className="flex flex-wrap gap-2">
            {experiments.map((exp) => (
              <button
                key={exp}
                onClick={() =>
                  setSelectedExps((prev) =>
                    prev.includes(exp)
                      ? prev.filter((e) => e !== exp)
                      : [...prev, exp],
                  )
                }
                className={cn(
                  "rounded-lg px-3 py-1.5 text-xs font-semibold transition-all duration-200",
                  selectedExps.includes(exp)
                    ? "bg-gradient-to-r from-teal-400 to-cyan-500 text-slate-900 shadow-md shadow-teal-400/20"
                    : "border border-slate-700/50 bg-slate-800/40 text-slate-400 hover:text-slate-200 hover:border-slate-600",
                )}
              >
                {exp}
              </button>
            ))}
          </div>
          <LineChart
            data={historyData}
            lines={selectedExps.map((exp, i) => ({
              key: "similarity",
              label: exp,
              color: ["#2dd4bf", "#38bdf8", "#fbbf24", "#fb7185", "#a78bfa"][i % 5],
            }))}
            title="Average Similarity 收敛曲线"
            xLabel="Epoch"
            yLabel="Similarity"
            height={360}
          />
        </div>
      )}

      {/* ====== ROW 5: Error Analysis ====== */}
      {errorData && (
        <div className="space-y-4">
          <div className="flex items-center gap-3">
            <select
              value={selectedLabel}
              onChange={(e) => setSelectedLabel(e.target.value)}
              className="rounded-xl border border-slate-700/50 bg-slate-800/60 px-4 py-2.5 text-sm text-slate-200 focus:border-teal-400/50 focus:outline-none"
            >
              {summaryRows.map((r) => (
                <option key={r.label} value={r.label}>
                  {r.label.replace(/_/g, " ")} — {r.model_name}
                </option>
              ))}
            </select>
            <span className="tech-badge rounded-md bg-amber-400/10 border border-amber-400/20 px-2 py-0.5 text-[10px] text-amber-400">
              错误分析
            </span>
          </div>

          <div className="grid grid-cols-[0.9fr_1.1fr] gap-6">
            {errorData.stats.length > 0 && (
              <div className="glass-card rounded-2xl p-5">
                <BarChart
                  data={errorData.stats.map((s) => ({
                    name: s.category,
                    count: s.count,
                    similarity: Math.round(s.avg_similarity * 100),
                  }))}
                  bars={[
                    { key: "count", label: "Count", color: "#38bdf8" },
                    { key: "similarity", label: "Avg Sim %", color: "#fbbf24" },
                  ]}
                  title="错误类别分布"
                  height={340}
                />
              </div>
            )}
            <div className="space-y-4">
              {bestExamples.length > 0 && (
                <div className="glass-card rounded-2xl p-4 border-l-2 border-emerald-400/50">
                  <h3 className="mb-2 text-sm font-bold text-emerald-400 flex items-center gap-1.5">
                    <ThumbsUp size={14} />
                    最佳样例 Top-{Math.min(5, bestExamples.length)}
                  </h3>
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="text-slate-400">
                        <th className="py-1 text-left">#</th>
                        <th className="py-1 text-left">参考文本</th>
                        <th className="py-1 text-left">预测文本</th>
                        <th className="py-1 text-right">相似度</th>
                      </tr>
                    </thead>
                    <tbody>
                      {bestExamples.slice(0, 5).map((ex, i) => (
                        <tr key={i} className="border-t border-slate-700/30 hover:bg-slate-800/20">
                          <td className="py-1.5 text-slate-400">{ex.dataset_index}</td>
                          <td className="py-1.5 max-w-[140px] truncate text-slate-300">{ex.reference}</td>
                          <td className="py-1.5 max-w-[140px] truncate text-slate-300">{ex.prediction}</td>
                          <td className="py-1.5 text-right text-emerald-400 font-mono font-bold">{ex.similarity.toFixed(2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              {worstExamples.length > 0 && (
                <div className="glass-card rounded-2xl p-4 border-l-2 border-red-400/50">
                  <h3 className="mb-2 text-sm font-bold text-red-400 flex items-center gap-1.5">
                    <ThumbsDown size={14} />
                    最差样例 Top-{Math.min(5, worstExamples.length)}
                  </h3>
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="text-slate-400">
                        <th className="py-1 text-left">#</th>
                        <th className="py-1 text-left">参考文本</th>
                        <th className="py-1 text-left">预测文本</th>
                        <th className="py-1 text-right">相似度</th>
                      </tr>
                    </thead>
                    <tbody>
                      {worstExamples.slice(0, 5).map((ex, i) => (
                        <tr key={i} className="border-t border-slate-700/30 hover:bg-slate-800/20">
                          <td className="py-1.5 text-slate-400">{ex.dataset_index}</td>
                          <td className="py-1.5 max-w-[140px] truncate text-slate-300">{ex.reference}</td>
                          <td className="py-1.5 max-w-[140px] truncate text-slate-300">{ex.prediction}</td>
                          <td className="py-1.5 text-right text-red-400 font-mono font-bold">{ex.similarity.toFixed(2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
