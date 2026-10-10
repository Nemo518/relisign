// API Request/Response Types

export interface PredictRequest {
  video_path: string;
  checkpoint_path: string;
  rgb_support: boolean;
  lightweight: boolean;
  ultra_lightweight: boolean;
}

export interface PredictResponse {
  success: boolean;
  prediction: string;
  latency_s: number;
  effective_fps: number;
  stdout: string;
  stderr: string;
  return_code: number;
  command: string[];
  pose_sequence?: PoseSequence | null;
}

export interface XAIRequest {
  video_path: string;
  checkpoint_path: string;
  rgb_support: boolean;
  reference_text?: string;
  pose_path?: string;
  device?: string;
}

export interface XAIResponse {
  prediction: string;
  target_sentence: string;
  loss: number;
  xai_timeline: XaiTimelineRow[];
  attention_matrix: AttentionMatrixRow[];
  rgb_cross: RgbCrossRow[];
  top_frames: TopFrameRow[];
  part_summary: PartSummaryRow[];
  alignment: AlignmentRow[];
}

export interface XaiTimelineRow {
  frame: number;
  xai_score: number;
  encoder_attention: number;
  encoder_attention_norm: number;
  gradcam_combined: number;
  body: number;
  left: number;
  right: number;
  face_all: number;
}

export interface AttentionMatrixRow {
  source_frame: number;
  target_frame: number;
  attention: number;
}

export interface RgbCrossRow {
  branch: string;
  patch_x: number;
  patch_y: number;
  attention: number;
}

export interface TopFrameRow {
  frame: number;
  xai_score: number;
  encoder_attention?: number;
  gradcam_combined?: number;
}

export interface PartSummaryRow {
  part: string;
  mean_importance: number;
  peak_importance: number;
}

export interface AlignmentRow {
  position: number;
  reference_char: string;
  prediction_char: string;
  operation: "match" | "insert" | "delete" | "replace";
}

// Data API types
export interface VideoRecord {
  name: string;
  split: string;
  relative_path: string;
  sample_id: string;
  dataset_index: number | null;
  reference: string | null;
  gloss: string | null;
}

export interface BestExampleRecord extends VideoRecord {
  similarity: number;
  prediction: string;
}

export interface VideoMetadata {
  frame_count: number;
  fps: number;
  duration_s: number;
  width: number;
  height: number;
}

export interface MetricsSummaryRow {
  label: string;
  model_name: string;
  bleu4: number;
  rouge_l: number;
  wer: number;
  average_similarity: number;
  csv_path: string;
}

export interface MetricCards {
  model_name: string;
  bleu4: number;
  rouge_l: number;
  wer: number;
  average_similarity: number;
}

export interface TrainingHistoryRow {
  epoch: number;
  selection_average_similarity: number;
  experiment: string;
  [key: string]: unknown;
}

export interface ErrorExample {
  dataset_index: number;
  reference: string;
  prediction: string;
  similarity: number;
  gloss?: string;
}

export interface ErrorAnalysisResult {
  best_examples: ErrorExample[];
  worst_examples: ErrorExample[];
  stats: ErrorCategory[];
}

export interface ErrorCategory {
  category: string;
  count: number;
  avg_similarity: number;
}

export interface ConfusionPair {
  reference_char: string;
  prediction_char: string;
  count: number;
  operation: string;
}

export interface ConfusionArtifacts {
  pair_df: ConfusionPair[];
  matrix_df: ConfusionPair[];
  summary_df: { operation: string; count: number }[];
}

export interface CachedPrediction {
  model_label: string;
  model_name: string;
  prediction: string;
  similarity: number;
  reference: string;
  gloss: string;
}

export interface ModelRadarRow {
  model_name: string;
  metric: string;
  value: number;
}

export interface LatencyBreakdownRow {
  stage: string;
  start_ms: number;
  duration_ms: number;
}

export interface FrameExtractionRequest {
  video_path: string;
  frame_indices: number[];
  resize_width?: number;
}

export interface FrameExtractionRow {
  frame: number;
  image_base64: string;
}

export interface PoseSequenceFrame {
  frame: number;
  points: { x: number; y: number; confidence: number }[];
}

export interface PoseSequence {
  video_path: string;
  num_frames: number;
  num_keypoints: number;
  frame_width: number;
  frame_height: number;
  fps: number;
  frames: PoseSequenceFrame[];
}

export interface DatasetRecord {
  name: string;
  path: string;
  video_count: number;
  label_count: number;
  label_formats: string[];
  pose_count: number;
  splits: string[];
  has_videos: boolean;
  has_labels: boolean;
  has_poses: boolean;
  status: string;
}

export interface ExperimentResult {
  directory: string;
  split: string;
  sample_count: number;
  avg_similarity: number;
  max_similarity: number;
  min_similarity: number;
}
