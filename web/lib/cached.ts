// Cached-run fallback. Flagship runs are pre-baked (see
// scripts/bake_cached_run.py) into public/cached-runs/ as the exact `scored` +
// `done` payloads the live stream emits. When the API is unreachable — a cold/
// sleeping HF Space, dead WiFi, an API hiccup — the same UI replays a known-good
// run identically, so a live demo can never break on stage.

import type { DoneEvent, ScoredEvent, StreamHandlers } from "@/lib/types";

export interface CachedRunSummary {
  slug: string;
  label: string;
  query: string;
  provider: string;
}

export interface CachedRun {
  slug: string;
  label: string;
  query: string;
  scored: ScoredEvent;
  done: DoneEvent;
}

export async function loadCachedManifest(): Promise<CachedRunSummary[]> {
  try {
    const res = await fetch("/cached-runs/manifest.json", { cache: "no-store" });
    if (!res.ok) return [];
    return (await res.json()) as CachedRunSummary[];
  } catch {
    return [];
  }
}

export async function loadCachedRun(slug: string): Promise<CachedRun | null> {
  try {
    const res = await fetch(`/cached-runs/${slug}.json`, { cache: "no-store" });
    if (!res.ok) return null;
    return (await res.json()) as CachedRun;
  } catch {
    return null;
  }
}

// Replay a cached run through the same handlers a live stream drives, pacing the
// scored → done transition so the selection-reveal animation still plays.
export async function replayCachedRun(
  run: CachedRun,
  handlers: StreamHandlers,
): Promise<void> {
  handlers.onProgress?.("cached", "replaying a pre-baked run (offline)");
  await delay(400);
  handlers.onScored?.(run.scored);
  await delay(900);
  handlers.onDone?.(run.done);
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
