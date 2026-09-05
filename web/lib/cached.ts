/**
 * Offline fallback for the library demo.
 *
 * A live demo depends on a reachable API, a network, and model weights loading
 * on time. On stage, any of those failing means nothing to show. `scripts/
 * bake_cached_run.py` captures real runs into `public/cached-runs/`, and this
 * replays them through the same handlers the live stream drives.
 *
 * Everything here is a snapshot of genuine pipeline output — never fabricated.
 * A replayed run is always labelled as cached rather than passed off as live: a
 * demo that quietly lies about what it just computed is worse than one that
 * visibly falls back.
 */

import type {
  Clip,
  Message,
  Metrics,
  ScoredCandidate,
  SelectedEvidence,
  Video,
  VideoDetail,
} from "@/lib/library";

export type CachedRun = {
  video_id: string;
  query: string;
  scored: ScoredCandidate[];
  selected: SelectedEvidence[];
  metrics: Metrics | null;
  clips: Clip[];
  answer: string | null;
  answer_provider: string | null;
};

type Manifest = {
  videos: Video[];
  details: Record<string, VideoDetail>;
  runs: CachedRun[];
};

let cache: Manifest | null | undefined;

async function manifest(): Promise<Manifest | null> {
  if (cache !== undefined) return cache;
  try {
    const response = await fetch("/cached-runs/manifest.json", { cache: "no-store" });
    cache = response.ok ? ((await response.json()) as Manifest) : null;
  } catch {
    cache = null;
  }
  return cache;
}

export async function cachedVideos(): Promise<Video[]> {
  return (await manifest())?.videos ?? [];
}

export async function cachedVideoDetail(id: string): Promise<VideoDetail | null> {
  const loaded = await manifest();
  return loaded?.details?.[id] ?? null;
}

function normalize(value: string): string {
  return value.trim().toLowerCase().replace(/\s+/g, " ").replace(/[?.!]+$/, "");
}

/**
 * Find a cached run for this question.
 *
 * Falls back to any run for the same video when the wording does not match, so
 * an unrehearsed question still shows real selected evidence rather than an
 * error — the run is labelled cached, and the question shown is the one that
 * was actually answered.
 */
export async function cachedRun(videoId: string, query: string): Promise<CachedRun | null> {
  const loaded = await manifest();
  if (!loaded?.runs?.length) return null;

  const forVideo = loaded.runs.filter((run) => run.video_id === videoId);
  if (forVideo.length === 0) return null;

  const wanted = normalize(query);
  return forVideo.find((run) => normalize(run.query) === wanted) ?? forVideo[0];
}

export type ReplayHandlers = {
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
};

/**
 * Replay a captured run, paced so the collapse animation still reads.
 *
 * The delays are not decorative: the scored → selected transition IS the
 * demonstration, and firing both in the same tick would skip it entirely.
 */
export async function replayCachedRun(
  run: CachedRun,
  handlers: ReplayHandlers,
): Promise<void> {
  const metrics = run.metrics ?? ({} as Metrics);

  handlers.onStage?.("scoring", "Scoring stored evidence (cached)");
  await delay(350);
  handlers.onScored?.(run.scored);

  handlers.onStage?.("selecting", "Compressing to key evidence (cached)");
  await delay(750);
  handlers.onSelected?.(run.selected, metrics);

  if (run.clips.length > 0) {
    await delay(250);
    handlers.onClips?.(run.clips);
  }

  handlers.onStage?.("answering", "Answering from the evidence (cached)");
  await delay(450);
  handlers.onDone?.({
    answer: run.answer,
    answer_provider: run.answer_provider,
    metrics,
    clips: run.clips,
  });
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
