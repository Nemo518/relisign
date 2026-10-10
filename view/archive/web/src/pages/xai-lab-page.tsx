import { useState, useMemo, useEffect } from "react";
import { useAppStore } from "@/store/app-store";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { queryKeys } from "@/api/query-keys";
import { cn } from "@/lib/utils";
import type { BestExampleRecord, XAIResponse, ConfusionArtifacts } from "@/types/api";
import { DEFAULT_CHECKPOINT } from "@/lib/constants";

import { SampleSelector } from "@/components/video/sample-selector";
import { MetricCard } from "@/components/overview/metric-card";
import { LineChart } from "@/components/charts/line-chart";
import { BarChart } from "@/components/charts/bar-chart";
import { GaugeChart } from "@/components/charts/gauge-chart";
import { AlignmentTokens } from "@/components/prediction/alignment-tokens";
import { AttentionMatrixHeatmap } from "@/components/xai/attention-matrix-heatmap";
import { KeyframeGallery } from "@/components/xai/keyframe-gallery";
import { RgbCrossAttentionGrid } from "@/components/xai/rgb-cross-attention-grid";
import { ConfusionHeatmap } from "@/components/xai/confusion-heatmap";
import { Switch } from "@/components/ui/switch";
import { Button } from "@/components/ui/button";
import {
  Brain,
  Activity,
  Search,
  Eye,
  Target,
  Sparkles,
  Loader2,
  Zap,
  Cpu,
} from "lucide-react";

// ── Demo data generators (fallback when backend unavailable) ──

function generateDemoTimeline(
  refText: string,
  predText: string,
) {
  const maxLen = Math.max(refText.length, predText.length, 20);
  const frames: {
    name: string;
    xai_score: number;
    encoder_attention: number;
    gradcam_combined: number;
    body: number;
    left: number;
    right: number;
    face_all: number;
  }[] = [];

  for (let i = 0; i < maxLen; i++) {
    const phase = i / maxLen;
    const attn = 0.2 + 0.5 * Math.sin(phase * Math.PI) + Math.random() * 0.12;
    const gradcam = 0.15 + 0.5 * Math.sin(phase * Math.PI + 0.3) + Math.random() * 0.12;
    frames.push({
      name: `F${i + 1}`,
      xai_score: Math.round((0.5 * attn + 0.5 * gradcam) * 1000) / 1000,
      encoder_attention: Math.round(attn * 1000) / 1000,
      gradcam_combined: Math.round(gradcam * 1000) / 1000,
      body: Math.round((0.1 + 0.3 * Math.cos(phase * Math.PI * 0.7) + Math.random() * 0.1) * 1000) / 1000,
      left: Math.round((0.2 + 0.45 * Math.sin(phase * Math.PI + 1.2) + Math.random() * 0.12) * 1000) / 1000,
      right: Math.round((0.25 + 0.5 * Math.sin(phase * Math.PI + 0.6) + Math.random() * 0.1) * 1000) / 1000,
      face_all: Math.round((0.05 + 0.15 * Math.sin(phase * Math.PI * 0.5) + Math.random() * 0.06) * 1000) / 1000,
    });
  }
  return frames;
}

function generateDemoParts() {
  return [
    { name: "右手", mean: 0.42, peak: 0.78 },
    { name: "左手", mean: 0.35, peak: 0.65 },
    { name: "身体", mean: 0.18, peak: 0.35 },
    { name: "面部", mean: 0.12, peak: 0.22 },
  ];
}

function generateDemoAttentionMatrix(size: number = 24) {
  const rows: { source_frame: number; target_frame: number; attention: number }[] = [];
  for (let src = 0; src < size; src++) {
    for (let tgt = 0; tgt < size; tgt++) {
      const dist = Math.abs(src - tgt);
      const attn = Math.exp(-dist / 4) * (0.6 + 0.4 * Math.random());
      rows.push({ source_frame: src, target_frame: tgt, attention: Math.round(attn * 1000) / 1000 });
    }
  }
  return rows;
}

function generateDemoTopFrames(refText: string, predText: string) {
  const maxLen = Math.max(refText.length, predText.length, 20);
  const all = Array.from({ length: maxLen }, (_, i) => ({
    frame: i,
    xai_score: 0.3 + 0.55 * Math.sin((i / maxLen) * Math.PI) + Math.random() * 0.15,
    encoder_attention: 0.2 + 0.4 * Math.sin((i / maxLen) * Math.PI) + Math.random() * 0.1,
    gradcam_combined: 0.2 + 0.5 * Math.sin((i / maxLen) * Math.PI + 0.3) + Math.random() * 0.1,
  }));
  return all.sort((a, b) => b.xai_score - a.xai_score).slice(0, 6);
}

export function XaiLabPage() {
  const split = useAppStore((s) => s.split);
  const [selectedSample, setSelectedSample] = useState<BestExampleRecord | null>(null);
  const [realXai, setRealXai] = useState(false);

  const isBestExample = selectedSample != null && "similarity" in selectedSample;
  const bestExample = selectedSample as BestExampleRecord | null;

  // Real XAI mutation — only when toggled on and sample selected
  const [xaiResult, setXaiResult] = useState<XAIResponse | null>(null);
  const [xaiRunning, setXaiRunning] = useState(false);
  const [xaiError, setXaiError] = useState<string | null>(null);

  // Auto-load best examples and pre-select first
  const bestExamplesQuery = useQuery({
    queryKey: queryKeys.bestExamples(split),
    queryFn: () => api.bestExamples(split),
  });

  // Auto-select first sample when data loads
  useEffect(() => {
    if (!selectedSample && bestExamplesQuery.data?.length) {
      setSelectedSample(bestExamplesQuery.data[0] as BestExampleRecord);
    }
  }, [bestExamplesQuery.data, selectedSample]);

  // Metrics summary for confusion model selector
  const summaryQuery = useQuery({
    queryKey: queryKeys.metricsSummary(split),
    queryFn: () => api.metricsSummary(split),
  });
  const labels = summaryQuery.data?.map((r) => r.label) ?? [];
  const [selectedLabel, setSelectedLabel] = useState("");

  // Auto-select first label when metrics load
  useEffect(() => {
    if (!selectedLabel && labels.length > 0) {
      setSelectedLabel(labels[0]);
    }
  }, [labels, selectedLabel]);

  const confusionQuery = useQuery({
    queryKey: queryKeys.confusionArtifacts(selectedLabel),
    queryFn: () => api.confusionArtifacts(selectedLabel),
    enabled: !!selectedLabel,
  });

  const predictionText = xaiResult?.prediction ?? bestExample?.prediction ?? null;
  const referenceText = selectedSample?.reference ?? "";
  const similarity = bestExample?.similarity ?? null;
  const xaiLoss = xaiResult?.loss ?? null;

  const hasRealData = xaiResult != null;
  const isDemo = !hasRealData;

  // Timeline data: real or demo
  const timelineData = useMemo(() => {
    if (xaiResult?.xai_timeline) {
      return xaiResult.xai_timeline.map((r) => ({
        name: `F${r.frame}`,
        xai_score: r.xai_score,
        encoder_attention: r.encoder_attention,
        gradcam_combined: r.gradcam_combined,
        body: (r as any).body ?? 0,
        left: (r as any).left ?? 0,
        right: (r as any).right ?? 0,
        face_all: (r as any).face_all ?? 0,
      }));
    }
    return generateDemoTimeline(referenceText, predictionText ?? "");
  }, [xaiResult, referenceText, predictionText]);

  // Part summary: real or demo
  const partData = useMemo(() => {
    if (xaiResult?.part_summary) {
      return xaiResult.part_summary.map((r) => ({
        name: r.part,
        mean: r.mean_importance,
        peak: r.peak_importance,
      }));
    }
    return generateDemoParts();
  }, [xaiResult]);

  // Attention matrix: real or demo
  const attentionMatrix = useMemo(() => {
    if (xaiResult?.attention_matrix) return xaiResult.attention_matrix;
    return generateDemoAttentionMatrix(24);
  }, [xaiResult]);

  // Top frames: real or demo
  const topFrames = useMemo(() => {
    if (xaiResult?.top_frames) {
      return xaiResult.top_frames.map((r) => ({
        frame: r.frame,
        xai_score: r.xai_score,
        encoder_attention: r.encoder_attention,
        gradcam_combined: r.gradcam_combined,
      }));
    }
    return generateDemoTopFrames(referenceText, predictionText ?? "");
  }, [xaiResult, referenceText, predictionText]);

  const runRealXai = async () => {
    if (!selectedSample) return;
    setXaiRunning(true);
    setXaiError(null);
    try {
      const result = await api.xai({
        video_path: selectedSample.relative_path,
        checkpoint_path: DEFAULT_CHECKPOINT,
        rgb_support: false,
        reference_text: referenceText || undefined,
      });
      setXaiResult(result);
    } catch (err: any) {
      setXaiError(err?.message ?? "XAI analysis failed");
    } finally {
      setXaiRunning(false);
    }
  };

  return (
    <div className="space-y-5">
      {/* ====== HEADER ====== */}
      <div className="flex items-center gap-3">
        <h2 className="text-xl font-bold">模型归因分析台</h2>
        <span className="tech-badge rounded-md bg-purple-400/10 border border-purple-400/20 px-2.5 py-1 text-[10px] text-purple-400">
          ATTRIBUTION LAB
        </span>
        <div className="ml-auto flex items-center gap-2">
          {isDemo && (
            <span className="tech-badge rounded-md bg-cyan-400/10 border border-cyan-400/20 px-2.5 py-1 text-[10px] text-cyan-400">
              演示模式
            </span>
          )}
          {hasRealData && (
            <span className="tech-badge rounded-md bg-emerald-400/10 border border-emerald-400/20 px-2.5 py-1 text-[10px] text-emerald-400">
              真实归因
            </span>
          )}
          {xaiError && (
            <span className="tech-badge rounded-md bg-red-400/10 border border-red-400/20 px-2.5 py-1 text-[10px] text-red-400">
              分析失败
            </span>
          )}
        </div>
      </div>

      {/* ====== ROW 1: Sample + Controls + Summary ====== */}
      <div className="grid grid-cols-[0.85fr_1.15fr] gap-5">
        {/* Left: Sample selector + Run button */}
        <div className="space-y-4">
          <div className="glass-card rounded-2xl p-4 space-y-4">
            <div className="flex items-center gap-2 text-sm font-semibold text-slate-300">
              <Search size={16} className="text-purple-400" />
              <span>样例选择</span>
            </div>

            <SampleSelector
              value={selectedSample}
              onChange={(v) => {
                setSelectedSample(v as BestExampleRecord);
                setXaiResult(null);
                setXaiError(null);
              }}
            />

            {selectedSample && (
              <div className="rounded-xl bg-slate-800/40 border border-slate-700/30 p-3 text-sm space-y-1">
                <div className="flex items-center gap-2">
                  <span className="text-slate-400 text-xs">参考：</span>
                  <span className="text-slate-200 text-xs">{referenceText || "暂无"}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-slate-400 text-xs">Gloss：</span>
                  <span className="text-slate-400 text-xs">{selectedSample.gloss || "暂无"}</span>
                </div>
                {isBestExample && (
                  <div className="flex items-center gap-2">
                    <span className="text-slate-400 text-xs">相似度：</span>
                    <span className="text-emerald-400 font-mono font-bold text-xs">
                      {bestExample!.similarity.toFixed(3)}
                    </span>
                  </div>
                )}
              </div>
            )}

            <Button
              variant="primary"
              className="w-full"
              onClick={runRealXai}
              disabled={!selectedSample || xaiRunning}
            >
              {xaiRunning ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  归因分析中...
                </>
              ) : (
                <>
                  <Zap size={16} />
                  运行真实归因分析
                </>
              )}
            </Button>

            {xaiError && (
              <div className="rounded-xl bg-red-400/5 border border-red-400/20 p-2.5">
                <p className="text-[10px] text-red-400">{xaiError}</p>
                <p className="mt-1 text-[9px] text-slate-400">
                  后端不可用时，页面默认使用演示数据展示归因分析能力
                </p>
              </div>
            )}
          </div>

          {/* Tech stack */}
          <div className="holo-card rounded-2xl p-4 border-purple-400/10">
            <div className="flex items-center gap-2 text-xs text-purple-400 mb-2">
              <Sparkles size={14} />
              <span className="font-semibold">归因分析技术栈</span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-[10px]">
              <div className="rounded-lg bg-slate-800/40 border border-slate-700/20 p-2">
                <div className="text-purple-300 font-semibold">Self-Attention</div>
                <div className="text-slate-400 mt-0.5">编码器层自注意力权重，捕获帧间时序依赖</div>
              </div>
              <div className="rounded-lg bg-slate-800/40 border border-slate-700/20 p-2">
                <div className="text-purple-300 font-semibold">Grad-CAM</div>
                <div className="text-slate-400 mt-0.5">梯度加权类激活映射，定位关键身体部位</div>
              </div>
              <div className="rounded-lg bg-slate-800/40 border border-slate-700/20 p-2">
                <div className="text-purple-300 font-semibold">Cross-Attn</div>
                <div className="text-slate-400 mt-0.5">姿态→RGB跨模态注意力，可变形空间采样</div>
              </div>
              <div className="rounded-lg bg-slate-800/40 border border-slate-700/20 p-2">
                <div className="text-purple-300 font-semibold">Confusion</div>
                <div className="text-slate-400 mt-0.5">字符级混淆矩阵，错误模式可追溯定位</div>
              </div>
            </div>
          </div>
        </div>

        {/* Right: Summary + Alignment */}
        <div className="space-y-4">
          {selectedSample ? (
            <>
              {/* Summary Card */}
              <div className="glass-card rounded-2xl p-5 border-l-2 border-purple-400/50">
                <div className="flex items-center gap-2 mb-3">
                  <Activity size={16} className="text-purple-400" />
                  <span className="text-sm font-bold text-slate-200">归因分析摘要</span>
                  <span className="tech-badge ml-auto rounded-md bg-purple-400/10 border border-purple-400/20 px-2 py-0.5 text-[10px] text-purple-400">
                    {hasRealData ? "LIVE XAI" : "DEMO XAI"}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <span className="text-slate-400">参考文本：</span>
                    <span className="text-slate-200 font-medium">{referenceText || "—"}</span>
                  </div>
                  <div>
                    <span className="text-slate-400">预测文本：</span>
                    <span className="text-slate-200 font-medium">{predictionText || "—"}</span>
                  </div>
                  <div>
                    <span className="text-slate-400">文本相似度：</span>
                    <span className="text-emerald-300 font-mono font-bold">
                      {similarity != null ? similarity.toFixed(3) : "—"}
                    </span>
                  </div>
                  {xaiLoss != null && (
                    <div>
                      <span className="text-slate-400">XAI Loss：</span>
                      <span className="text-amber-300 font-mono font-bold">{xaiLoss.toFixed(4)}</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Metric cards */}
              <div className="grid grid-cols-3 gap-3">
                <MetricCard
                  title="关键帧峰值"
                  value={timelineData.length > 0
                    ? Math.max(...timelineData.map((d) => d.xai_score)).toFixed(3)
                    : "—"}
                  note="Attention + Grad-CAM 融合最强帧"
                />
                <MetricCard
                  title="主导部位"
                  value={partData[0]?.name ?? "N/A"}
                  note={`峰值贡献 ${(partData[0]?.peak ?? 0).toFixed(2)}`}
                />
                <MetricCard
                  title="文本差异位"
                  value={String(
                    Array.from(referenceText).filter(
                      (c, i) => c !== (Array.from(predictionText ?? "")[i] ?? ""),
                    ).length,
                  )}
                  note="参考 vs 预测差异位置数"
                />
              </div>

              {/* Alignment Tokens */}
              <AlignmentTokens
                reference={referenceText}
                prediction={predictionText ?? ""}
              />

              {/* Gauge */}
              <GaugeChart value={similarity ?? 0.6} title="归因可信度" />
            </>
          ) : (
            <div className="glass-card rounded-3xl p-10 flex items-center justify-center">
              <div className="text-center">
                <div className="mb-4 inline-flex h-16 w-16 items-center justify-center rounded-2xl bg-purple-400/10 border border-purple-400/20">
                  <Brain size={32} className="text-purple-400/60" />
                </div>
                <p className="text-slate-400 font-mono text-sm">选择一个样例查看归因分析</p>
                <p className="mt-1 text-xs text-slate-400">
                  Self-Attention + Grad-CAM + Cross-Attention + 混淆矩阵
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ====== ROW 2: XAI Timeline + Keyframe Gallery ====== */}
      {selectedSample && (
        <>
          <div className="grid grid-cols-[1.1fr_0.9fr] gap-5">
            <div className="glass-card rounded-2xl p-4">
              <LineChart
                data={timelineData}
                lines={[
                  { key: "xai_score", label: "归因分数", color: "#a78bfa" },
                  { key: "encoder_attention", label: "Encoder Attn", color: "#38bdf8" },
                  { key: "gradcam_combined", label: "Grad-CAM", color: "#fbbf24" },
                ]}
                title="时序归因：Attention × Grad-CAM 叠合分析"
                height={320}
              />
            </div>
            <KeyframeGallery frames={topFrames} />
          </div>

          {/* ====== ROW 3: Self-Attention Matrix + Part Bar Chart ====== */}
          <div className="grid grid-cols-[1fr_0.7fr] gap-5">
            <AttentionMatrixHeatmap data={attentionMatrix} />
            <div className="glass-card rounded-2xl p-4">
              <BarChart
                data={partData}
                bars={[
                  { key: "mean", label: "均值贡献", color: "#a78bfa" },
                  { key: "peak", label: "峰值贡献", color: "#c084fc" },
                ]}
                title="身体部位贡献排序"
                height={280}
              />
            </div>
          </div>

          {/* ====== ROW 4: Per-Part Grad-CAM + RGB Cross-Attention ====== */}
          <div className="grid grid-cols-[1fr_1fr] gap-5">
            <div className="glass-card rounded-2xl p-4">
              <LineChart
                data={timelineData}
                lines={[
                  { key: "right", label: "右手", color: "#f472b6" },
                  { key: "left", label: "左手", color: "#c084fc" },
                  { key: "body", label: "身体", color: "#22d3ee" },
                  { key: "face_all", label: "面部", color: "#fbbf24" },
                ]}
                title="Grad-CAM 身体部位归因分解"
                height={300}
              />
            </div>
            {xaiResult?.rgb_cross && xaiResult.rgb_cross.length > 0 ? (
              <RgbCrossAttentionGrid data={xaiResult.rgb_cross} />
            ) : (
              <div className="glass-card rounded-2xl p-4">
                <div className="flex items-center gap-2 mb-3">
                  <Cpu size={14} className="text-emerald-400" />
                  <span className="text-sm font-bold">RGB 跨模态注意力</span>
                </div>
                <div className="flex items-center justify-center py-12 rounded-xl bg-slate-800/40 border border-slate-700/30">
                  <p className="text-xs text-slate-400 text-center">
                    RGB-Support 未启用时无跨模态注意力数据
                    <br />
                    <span className="text-[10px] text-slate-500">
                      启用 RGB-Support 后可视化姿态→RGB 可变形注意力热力图
                    </span>
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* ====== ROW 5: Confusion Matrix (full width) ====== */}
          <div className="grid grid-cols-1 gap-5">
            {(confusionQuery.data || selectedSample) && (
              <div className="glass-card rounded-2xl p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Target size={16} className="text-red-400" />
                    <h3 className="text-sm font-bold text-slate-200">字符级混淆矩阵</h3>
                    <span className="tech-badge rounded-md bg-red-400/10 border border-red-400/20 px-2 py-0.5 text-[10px] text-red-400">
                      CONFUSION
                    </span>
                  </div>
                  {labels.length > 0 && (
                    <select
                      value={selectedLabel}
                      onChange={(e) => setSelectedLabel(e.target.value)}
                      className="rounded-xl border border-slate-700/50 bg-slate-800/60 px-3 py-1.5 text-[11px] text-slate-200 focus:border-purple-400/50 focus:outline-none"
                    >
                      {labels.map((l) => (
                        <option key={l} value={l}>{l.replace(/_/g, " ")}</option>
                      ))}
                    </select>
                  )}
                </div>
                {confusionQuery.isLoading ? (
                  <div className="flex items-center justify-center py-12">
                    <Loader2 size={20} className="animate-spin text-purple-400" />
                  </div>
                ) : confusionQuery.data ? (
                  <div className="grid grid-cols-[1fr_1fr_0.7fr] gap-4">
                    {/* Pair table */}
                    <div>
                      <h4 className="text-[11px] font-semibold text-slate-400 mb-2">混淆字符对 Top-15</h4>
                      <div className="overflow-x-auto max-h-64 rounded-lg border border-slate-700/30">
                        <table className="w-full text-sm">
                          <thead>
                            <tr className="border-b border-slate-700/50 text-slate-400 sticky top-0 bg-slate-900/90">
                              <th className="py-1.5 text-left text-[10px] font-semibold px-2">参考</th>
                              <th className="py-1.5 text-left text-[10px] font-semibold px-2">预测</th>
                              <th className="py-1.5 text-right text-[10px] font-semibold px-2">次数</th>
                              <th className="py-1.5 text-left text-[10px] font-semibold px-2">类型</th>
                            </tr>
                          </thead>
                          <tbody>
                            {confusionQuery.data.pair_df.slice(0, 15).map((r, i) => (
                              <tr key={i} className="border-b border-slate-700/20 hover:bg-slate-800/30 transition-colors">
                                <td className="py-1 text-slate-200 font-mono text-[11px] px-2">{r.reference_char}</td>
                                <td className="py-1 text-red-300 font-mono text-[11px] px-2">{r.prediction_char}</td>
                                <td className="py-1 text-right text-slate-300 font-mono text-[11px] px-2">{r.count}</td>
                                <td className="py-1 px-2">
                                  <span className={cn(
                                    "rounded px-1 py-0.5 text-[8px] font-semibold",
                                    r.operation === "replace" ? "bg-red-400/10 text-red-400" :
                                    r.operation === "delete" ? "bg-amber-400/10 text-amber-400" :
                                    r.operation === "insert" ? "bg-blue-400/10 text-blue-400" :
                                    "bg-slate-400/10 text-slate-400",
                                  )}>
                                    {r.operation === "replace" ? "替换" :
                                     r.operation === "delete" ? "删除" :
                                     r.operation === "insert" ? "插入" : r.operation}
                                  </span>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>

                    {/* Matrix heatmap */}
                    <div>
                      <h4 className="text-[11px] font-semibold text-slate-400 mb-2">混淆矩阵热力图</h4>
                      <ConfusionHeatmap data={confusionQuery.data.matrix_df.slice(0, 60)} />
                    </div>

                    {/* Operation summary */}
                    <div>
                      <h4 className="text-[11px] font-semibold text-slate-400 mb-2">错误类型分布</h4>
                      <div className="space-y-2">
                        {confusionQuery.data.summary_df.map((r, i) => (
                          <div key={i} className="rounded-lg bg-slate-800/40 border border-slate-700/20 p-2.5">
                            <div className="flex items-center justify-between mb-1">
                              <span className={cn(
                                "text-[11px] font-semibold",
                                r.operation === "replace" ? "text-red-400" :
                                r.operation === "delete" ? "text-amber-400" :
                                r.operation === "insert" ? "text-blue-400" : "text-slate-400",
                              )}>
                                {r.operation === "replace" ? "替换 Replace" :
                                 r.operation === "delete" ? "删除 Delete" :
                                 r.operation === "insert" ? "插入 Insert" : r.operation}
                              </span>
                              <span className="text-[11px] font-mono font-bold text-slate-200">{r.count}</span>
                            </div>
                            <div className="h-1.5 rounded-full bg-slate-700/50 overflow-hidden">
                              <div
                                className={cn(
                                  "h-full rounded-full transition-all",
                                  r.operation === "replace" ? "bg-gradient-to-r from-red-500 to-rose-400" :
                                  r.operation === "delete" ? "bg-gradient-to-r from-amber-500 to-yellow-400" :
                                  "bg-gradient-to-r from-blue-500 to-cyan-400",
                                )}
                                style={{
                                  width: `${Math.min(100, (r.count / Math.max(...confusionQuery.data.summary_df.map(s => s.count), 1)) * 100)}%`,
                                }}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="flex items-center justify-center py-12 text-xs text-slate-400">
                    选择模型标签查看全局混淆矩阵
                  </div>
                )}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
