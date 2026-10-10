import type {
  PredictRequest,
  PredictResponse,
  XAIRequest,
  XAIResponse,
  VideoRecord,
  BestExampleRecord,
  VideoMetadata,
  MetricsSummaryRow,
  MetricCards,
  TrainingHistoryRow,
  ErrorAnalysisResult,
  ConfusionArtifacts,
  CachedPrediction,
  ModelRadarRow,
  LatencyBreakdownRow,
  FrameExtractionRequest,
  FrameExtractionRow,
  DatasetRecord,
  PoseSequence,
} from "@/types/api";

const BASE_URL =
  (import.meta as any).env?.VITE_API_BASE ?? "http://127.0.0.1:8000";

async function apiFetch<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  const url = `${BASE_URL}${path}`;
  const res = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`API ${res.status}: ${text}`);
  }

  if (
    options?.headers &&
    (options.headers as Record<string, string>)["Accept"] === "application/pdf"
  ) {
    return res.blob() as any;
  }

  return res.json();
}

export const api = {
  health: () => apiFetch<{ status: string }>("/health"),

  predict: (req: PredictRequest) =>
    apiFetch<PredictResponse>("/predict", {
      method: "POST",
      body: JSON.stringify(req),
    }),

  uploadVideo: async (file: File): Promise<{ uploaded_path: string; filename: string; size_bytes: number }> => {
    const formData = new FormData();
    formData.append("file", file);
    const url = `${BASE_URL}/predict/upload-video`;
    const res = await fetch(url, { method: "POST", body: formData });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`Upload failed ${res.status}: ${text}`);
    }
    return res.json();
  },

  xai: (req: XAIRequest) =>
    apiFetch<XAIResponse>("/xai", {
      method: "POST",
      body: JSON.stringify(req),
    }),

  // Data endpoints
  sampleVideos: () =>
    apiFetch<VideoRecord[]>("/data/sample-videos"),

  bestExamples: (split: string, limit?: number) =>
    apiFetch<BestExampleRecord[]>(
      `/data/best-examples?split=${split}&limit=${limit ?? 24}`,
    ),

  metricCards: (split: string) =>
    apiFetch<MetricCards>(`/data/metric-cards?split=${split}`),

  metricsSummary: (split: string) =>
    apiFetch<MetricsSummaryRow[]>(`/data/metrics-summary?split=${split}`),

  trainingHistory: () =>
    apiFetch<TrainingHistoryRow[]>("/data/training-history"),

  errorAnalysis: (label: string) =>
    apiFetch<ErrorAnalysisResult>(
      `/data/error-analysis?label=${encodeURIComponent(label)}`,
    ),

  confusionArtifacts: (label: string) =>
    apiFetch<ConfusionArtifacts>(
      `/data/confusion-artifacts?label=${encodeURIComponent(label)}`,
    ),

  videoMetadata: (videoPath: string) =>
    apiFetch<VideoMetadata>(
      `/data/video-metadata?path=${encodeURIComponent(videoPath)}`,
    ),

  cachedPrediction: (sampleId: string) =>
    apiFetch<CachedPrediction | null>(
      `/data/cached-prediction?sample_id=${encodeURIComponent(sampleId)}`,
    ),

  modelRadar: (split: string) =>
    apiFetch<ModelRadarRow[]>(`/data/model-radar?split=${split}`),

  latencyBreakdown: (latencyS: number) =>
    apiFetch<LatencyBreakdownRow[]>(
      `/data/latency-breakdown?latency_s=${latencyS}`,
    ),

  extractFrames: (req: FrameExtractionRequest) =>
    apiFetch<FrameExtractionRow[]>("/data/extract-frames", {
      method: "POST",
      body: JSON.stringify(req),
    }),

  errorCategoryFrame: (label: string) =>
    apiFetch<{ category: string; count: number; avg_similarity: number }[]>(
      `/data/error-category-frame?label=${encodeURIComponent(label)}`,
    ),

  judgePdf: (req: Record<string, unknown>) =>
    apiFetch<Blob>("/data/judge-pdf", {
      method: "POST",
      body: JSON.stringify(req),
      headers: { Accept: "application/pdf" },
    }),

  poseSequence: (videoPath: string) =>
    apiFetch<PoseSequence>(
      `/data/pose-sequence?video_path=${encodeURIComponent(videoPath)}`,
    ),

  allExperiments: (split: string) =>
    apiFetch<ExperimentResult[]>(
      `/data/all-experiments?split=${split}`,
    ),

  datasets: () => apiFetch<DatasetRecord[]>("/data/datasets"),

  mediaUrl: (relativePath: string) =>
    `${BASE_URL}/data/media/${encodeURIComponent(relativePath)}`,
};
