/**
 * Client for the two-phase library API.
 *
 * Phase one (`addVideo` + `streamIngestion`) runs the query-independent half of
 * the pipeline once and is slow. Phase two (`streamQuery`) answers questions
 * against the stored result and is fast. The UI's whole shape follows from that
 * asymmetry.
 */

const API_BASE =
  process.env.NEXT_PUBLIC_GIST_API?.replace(/\/$/, "") ?? "http://127.0.0.1:8000";

export type VideoStatus = "pending" | "ingesting" | "ready" | "failed";

export type Video = {
  id: string;
  url: string;
  youtube_id: string | null;
  title: string;
  duration_seconds: number;
  thumbnail_url: string | null;
  status: VideoStatus;
  status_detail: string | null;
  progress: number;
  frame_count: number;
  audio_window_count: number;
  error: string | null;
  created_at: string | null;
};

export type ScoredCandidate = {
  id: string;
  modality: "visual" | "audio";
  timestamp_seconds: number;
  score: number | null;
  text: string;
};

export type SelectedEvidence = {
  id: string;
  modality: "visual" | "audio";
  timestamp_seconds: number;
  text: string;
  reason: string;
  selection_rank: number;
  relevance_score: number;
  normalized_score: number;
  mmr_score: number;
  merged_from_count?: number;
};

export type Metrics = {
  input_candidates: number;
  selected_candidates: number;
  visual_selected: number;
  audio_selected: number;
  estimated_candidate_reduction_percent: number;
  estimated_token_reduction_percent: number;
  estimated_compressed_tokens: number;
  estimated_baseline_tokens: number;
  budget_preset_used: string;
  budget_stages_used?: number;
  tail_merged_groups?: number;
};

export type Clip = {
  candidate_id: string;
  modality: "visual" | "audio";
  start_seconds: number;
  end_seconds: number;
  timestamp_seconds: number;
  url: string;
  reason: string;
  text: string;
};

export type Message = {
  id: string;
  role: "user" | "assistant";
  query: string | null;
  answer: string | null;
  answer_provider: string | null;
  selected_evidence: SelectedEvidence[] | null;
  metrics: Metrics | null;
  clips: Clip[] | null;
  created_at: string | null;
};

export type VideoDetail = {
  video: Video;
  conversation_id: string | null;
  messages: Message[];
};

export function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

async function jsonOrThrow<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : detail;
    } catch {
      // Non-JSON error body; the status line is the best we have.
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export async function listVideos(): Promise<Video[]> {
  const response = await fetch(apiUrl("/v1/library/videos"), { cache: "no-store" });
  const body = await jsonOrThrow<{ videos: Video[] }>(response);
  return body.videos;
}

export async function getVideo(id: string): Promise<VideoDetail> {
  const response = await fetch(apiUrl(`/v1/library/videos/${id}`), { cache: "no-store" });
  return jsonOrThrow<VideoDetail>(response);
}

export async function addVideo(url: string): Promise<{ video: Video; started: boolean }> {
  const response = await fetch(apiUrl("/v1/library/videos"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
  return jsonOrThrow<{ video: Video; started: boolean }>(response);
}

export async function deleteVideo(id: string): Promise<void> {
  const response = await fetch(apiUrl(`/v1/library/videos/${id}`), { method: "DELETE" });
  if (!response.ok && response.status !== 404) {
    throw new Error(`could not remove video (${response.status})`);
  }
}

/** Parse an SSE byte stream into typed events. */
async function* readEvents(
  response: Response,
  signal?: AbortSignal,
): AsyncGenerator<{ event: string; data: unknown }> {
  if (!response.ok || !response.body) {
    throw new Error(`stream failed (${response.status})`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (!signal?.aborted) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE frames are separated by a blank line.
      let split = buffer.indexOf("\n\n");
      while (split !== -1) {
        const frame = buffer.slice(0, split);
        buffer = buffer.slice(split + 2);
        split = buffer.indexOf("\n\n");

        let event = "message";
        const dataLines: string[] = [];
        for (const line of frame.split("\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
        }
        if (dataLines.length === 0) continue;
        try {
          yield { event, data: JSON.parse(dataLines.join("\n")) };
        } catch {
          // A partial or malformed frame is not worth killing the stream over.
        }
      }
    }
  } finally {
    reader.cancel().catch(() => {});
  }
}

export type IngestionHandlers = {
  onProgress?: (video: Video) => void;
  onDone?: (video: Video) => void;
  onError?: (message: string) => void;
};

export async function streamIngestion(
  id: string,
  handlers: IngestionHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(apiUrl(`/v1/library/videos/${id}/events`), { signal });
  for await (const { event, data } of readEvents(response, signal)) {
    if (event === "progress") handlers.onProgress?.(data as Video);
    else if (event === "done") handlers.onDone?.(data as Video);
    else if (event === "error") handlers.onError?.((data as { message: string }).message);
  }
}

export type QueryHandlers = {
  onStage?: (stage: string, label: string) => void;
  onScored?: (candidates: ScoredCandidate[]) => void;
  onSelected?: (selected: SelectedEvidence[], metrics: Metrics) => void;
  onClips?: (clips: Clip[]) => void;
  onDone?: (payload: {
    answer: string | null;
    answer_provider: string | null;
    metrics: Metrics;
    clips: Clip[];
  }) => void;
  onError?: (message: string) => void;
};

export async function streamQuery(
  id: string,
  query: string,
  options: { answerer?: string; tailMerging?: boolean },
  handlers: QueryHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(apiUrl(`/v1/library/videos/${id}/query`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      answerer: options.answerer ?? "twelvelabs",
      tail_merging: options.tailMerging ?? false,
    }),
    signal,
  });

  if (!response.ok) {
    let detail = `request failed (${response.status})`;
    try {
      const body = await response.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      // keep the status-line message
    }
    handlers.onError?.(detail);
    return;
  }

  for await (const { event, data } of readEvents(response, signal)) {
    switch (event) {
      case "stage": {
        const payload = data as { stage: string; label: string };
        handlers.onStage?.(payload.stage, payload.label);
        break;
      }
      case "scored":
        handlers.onScored?.((data as { candidates: ScoredCandidate[] }).candidates);
        break;
      case "selected": {
        const payload = data as { selected: SelectedEvidence[]; metrics: Metrics };
        handlers.onSelected?.(payload.selected, payload.metrics);
        break;
      }
      case "clips":
        handlers.onClips?.((data as { clips: Clip[] }).clips);
        break;
      case "done":
        handlers.onDone?.(
          data as {
            answer: string | null;
            answer_provider: string | null;
            metrics: Metrics;
            clips: Clip[];
          },
        );
        break;
      case "error":
        handlers.onError?.((data as { message: string }).message);
        break;
    }
  }
}

export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return "—";
  const total = Math.round(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  }
  return `${minutes}:${String(secs).padStart(2, "0")}`;
}

export function formatTimestamp(seconds: number): string {
  return formatDuration(seconds);
}
