// Types mirroring the Gist API `/v1/demo/run` SSE payloads.

export type Answerer = "openai" | "claude" | "extractive";

export interface CandidatePoint {
  timestamp_seconds: number;
  score: number;
  text: string;
}

export interface ScoredEvent {
  visual: CandidatePoint[];
  audio: CandidatePoint[];
}

export interface SelectedCandidate {
  id: string;
  modality: "visual" | "audio";
  timestamp_seconds: number;
  text: string;
  selection_rank: number;
  relevance_score: number;
  normalized_score: number;
  mmr_score: number;
  source_score_type: string;
  support_label: string | null;
  reason: string;
}

export interface CompressionMetrics {
  input_candidates: number;
  selected_candidates: number;
  visual_selected: number;
  audio_selected: number;
  raw_input_candidates: number | null;
  estimated_candidate_reduction_percent: number;
  estimated_baseline_tokens: number;
  estimated_compressed_tokens: number;
  estimated_saved_tokens: number;
  estimated_token_reduction_percent: number;
}

export interface CompressionResponse {
  video_id: string;
  query: string;
  answer: string | null;
  answer_provider: string | null;
  query_intent: string | null;
  audio_scorer_used: string | null;
  selected: SelectedCandidate[];
  metrics: CompressionMetrics;
}

export interface VideoMeta {
  duration_seconds: number;
  frame_count: number;
  audio_window_count: number;
}

export interface DoneEvent {
  answer: string;
  provider: string;
  compression: CompressionResponse;
  video: VideoMeta;
}

export interface StreamHandlers {
  onProgress?: (stage: string, message: string) => void;
  onScored?: (scored: ScoredEvent) => void;
  onDone?: (done: DoneEvent) => void;
  onError?: (message: string) => void;
}

export interface RunRequest {
  video_path?: string;
  video_url?: string;
  query: string;
  answerer: Answerer;
  visual_scorer?: string;
  audio_scorer?: string;
  adaptive_budget?: boolean;
  decompose_query?: boolean;
  sample_count?: number;
  max_frames?: number;
  output_root?: string;
}
