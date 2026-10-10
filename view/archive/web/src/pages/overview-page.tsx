import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { queryKeys } from "@/api/query-keys";
import { useAppStore } from "@/store/app-store";
import { MetricCard } from "@/components/overview/metric-card";
import { RadarChart } from "@/components/charts/radar-chart";
import { ComboChart } from "@/components/charts/combo-chart";
import {
  Database,
  Video,
  FileText,
  CheckCircle,
  AlertCircle,
  Trophy,
  Target,
  Zap,
  Puzzle,
  Layers,
  Eye,
  Cpu,
} from "lucide-react";
import { cn } from "@/lib/utils";

const highlights = [
  {
    icon: <Layers size={22} />,
    title: "多模态融合",
    value: "RGB + Pose",
    desc: "双流并行编码，自适应特征融合",
    color: "from-cyan-400 to-teal-500",
    glow: "shadow-cyan-400/20",
  },
  {
    icon: <Eye size={22} />,
    title: "模型归因",
    value: "实时归因",
    desc: "Attention + Grad-CAM + 混淆矩阵",
    color: "from-violet-400 to-purple-500",
    glow: "shadow-purple-400/20",
  },
  {
    icon: <Cpu size={22} />,
    title: "端到端推理",
    value: "< 500ms",
    desc: "三级性能模式，30 FPS 实时流",
    color: "from-emerald-400 to-teal-500",
    glow: "shadow-emerald-400/20",
  },
  {
    icon: <Puzzle size={22} />,
    title: "三层解耦",
    value: "展示 / 推理 / 分析",
    desc: "React · FastAPI · PyTorch 独立部署",
    color: "from-amber-400 to-orange-500",
    glow: "shadow-amber-400/20",
  },
];

export function OverviewPage() {
  const split = useAppStore((s) => s.split);

  const metricsQuery = useQuery({
    queryKey: queryKeys.metricCards(split),
    queryFn: () => api.metricCards(split),
  });

  const summaryQuery = useQuery({
    queryKey: queryKeys.metricsSummary(split),
    queryFn: () => api.metricsSummary(split),
  });

  const datasetsQuery = useQuery({
    queryKey: queryKeys.datasets,
    queryFn: () => api.datasets(),
  });

  const cards = metricsQuery.data;
  const summaryRows = summaryQuery.data ?? [];
  const datasets = datasetsQuery.data ?? [];

  const bleu = cards?.bleu4 ?? 0;
  const rouge = cards?.rouge_l ?? 0;
  const wer = cards?.wer ?? 0;
  const similarity = (cards?.average_similarity ?? 0) * 100;

  const storyScores = [
    { name: "技术深度", value: Math.min(100, 42 + similarity * 0.48) },
    { name: "创新表达", value: Math.min(100, 38 + rouge * 0.68) },
    { name: "实用表现", value: Math.min(100, 44 + bleu * 0.86) },
    { name: "扩展潜力", value: Math.min(100, 100 - wer * 0.55) },
  ];

  const comboData = summaryRows.slice(0, 4).map((row) => ({
    model: String(row.model_name),
    bleu4: Math.round(row.bleu4 * 100) / 100,
    rougeL: Math.round(row.rouge_l * 100) / 100,
    werAdvantage: Math.round(Math.max(0, 100 - row.wer) * 100) / 100,
  }));

  return (
    <div className="space-y-6">
      {/* Innovation Highlights */}
      <div className="grid grid-cols-4 gap-4">
        {highlights.map((h, i) => (
          <div
            key={h.title}
            className={cn(
              "holo-card rounded-3xl p-5 transition-all duration-300 hover:-translate-y-1 cursor-default",
              "border border-white/5 hover:border-cyan-400/30",
            )}
            style={{ animationDelay: `${i * 0.1}s` }}
          >
            <div
              className={`mb-3 inline-flex h-11 w-11 items-center justify-center rounded-2xl bg-gradient-to-br ${h.color} text-white shadow-lg ${h.glow}`}
            >
              {h.icon}
            </div>
            <div className="text-sm font-semibold text-slate-300">{h.title}</div>
            <div className="mt-1 text-xl font-extrabold text-white">{h.value}</div>
            <div className="mt-1 text-xs leading-relaxed text-slate-400">{h.desc}</div>
          </div>
        ))}
      </div>

      {/* Metric Cards */}
      <div className="grid grid-cols-4 gap-4">
        <MetricCard
          title="主展示模型"
          value={cards?.model_name ?? "—"}
          note="从离线汇总自动抽取，展示时稳定复现"
        />
        <MetricCard
          title="BLEU-4"
          value={cards ? cards.bleu4.toFixed(2) : "—"}
          note={`${split.toUpperCase()} 集离线汇总指标`}
        />
        <MetricCard
          title="ROUGE-L"
          value={cards ? cards.rouge_l.toFixed(2) : "—"}
          note="语义一致性展示首选指标"
        />
        <MetricCard
          title="WER"
          value={cards ? cards.wer.toFixed(2) : "—"}
          note="越低越好，适合说明错误成本"
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-[1.05fr_0.95fr] gap-6">
        <div className="glass-card p-5">
          <RadarChart data={storyScores} title="系统能力雷达" />
        </div>
        <div className="glass-card p-5">
          <ComboChart data={comboData} title="核心指标面板" />
        </div>
      </div>

      {/* Datasets Overview */}
      <div>
        <h3 className="mb-4 text-lg font-bold flex items-center gap-2">
          <Database size={20} className="text-cyan-400" />
          <span className="text-white">数据集总览</span>
          <span className="tech-badge rounded-md bg-cyan-400/10 border border-cyan-400/20 px-2 py-0.5 text-[10px] text-cyan-400">
            {datasets.length} DATASETS
          </span>
        </h3>
        <div className="grid grid-cols-3 gap-4">
          {datasets.map((ds) => (
            <div
              key={ds.name}
              className={cn(
                "glass-card rounded-2xl p-5 transition-all duration-300 hover:-translate-y-0.5",
                ds.has_videos && "border-cyan-400/20 hover:border-cyan-400/40",
              )}
            >
              <div className="flex items-center justify-between mb-3">
                <h4 className="font-bold">{ds.name}</h4>
                {ds.status === "完整" ? (
                  <span className="inline-flex items-center gap-1 rounded-full bg-emerald-400/10 border border-emerald-400/30 px-2.5 py-0.5 text-[10px] font-semibold text-emerald-400">
                    <CheckCircle size={11} />完整
                  </span>
                ) : ds.status === "仅标签" ? (
                  <span className="inline-flex items-center gap-1 rounded-full bg-amber-400/10 border border-amber-400/30 px-2.5 py-0.5 text-[10px] font-semibold text-amber-400">
                    <AlertCircle size={11} />仅标签
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1 rounded-full bg-slate-400/10 border border-slate-400/20 px-2.5 py-0.5 text-[10px] font-semibold text-slate-400">
                    空
                  </span>
                )}
              </div>

              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="flex items-center gap-2 text-slate-400">
                  <Video size={14} />
                  <span>视频</span>
                  <span className="ml-auto font-semibold text-slate-200">
                    {ds.video_count > 0 ? ds.video_count.toLocaleString() : "—"}
                  </span>
                </div>
                <div className="flex items-center gap-2 text-slate-400">
                  <FileText size={14} />
                  <span>标签</span>
                  <span className="ml-auto font-semibold text-slate-200">
                    {ds.label_count > 0 ? ds.label_count : "—"}
                  </span>
                </div>
                <div className="col-span-2 text-xs text-slate-400">
                  格式：{ds.label_formats.length > 0 ? ds.label_formats.join(" / ") : "无"}
                </div>
                {ds.splits.length > 0 && (
                  <div className="col-span-2 flex flex-wrap gap-1">
                    {ds.splits.map((s) => (
                      <span
                        key={s}
                        className="tech-badge rounded-md bg-slate-700/50 px-2 py-0.5 text-[10px] text-slate-400"
                      >
                        {s}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
