"use client";

import { useRef, useState } from "react";

import { AnswerPanel } from "@/components/answer-panel";
import { ScoringTimeline } from "@/components/scoring-timeline";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { runDemo } from "@/lib/api";
import { loadCachedRun, replayCachedRun } from "@/lib/cached";
import type { Answerer, DoneEvent, ScoredEvent } from "@/lib/types";

type Phase = "idle" | "running" | "done" | "error";

const PRESETS: {
  label: string;
  path: string;
  query: string;
  cachedSlug?: string;
}[] = [
  {
    label: "Sample clip (5s)",
    path: ".gist/videos/sample_5s.mp4",
    query: "What is shown in this video?",
    cachedSlug: "sample-5s",
  },
  {
    label: "Paul Graham talk (3 min)",
    path: ".gist/videos/demo-trims/paul-graham-3min.mp4",
    query: "How do founders get startup ideas unconsciously?",
    cachedSlug: "paul-graham",
  },
];

export default function Home() {
  const [query, setQuery] = useState(PRESETS[0].query);
  const [videoUrl, setVideoUrl] = useState("");
  const [videoPath, setVideoPath] = useState(PRESETS[0].path);
  const [answerer, setAnswerer] = useState<Answerer>("extractive");

  const [phase, setPhase] = useState<Phase>("idle");
  const [log, setLog] = useState<string[]>([]);
  const [scored, setScored] = useState<ScoredEvent | null>(null);
  const [done, setDone] = useState<DoneEvent | null>(null);
  const [revealed, setRevealed] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [offline, setOffline] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const running = phase === "running";

  const handlers = {
    onProgress: (stage: string, message: string) =>
      setLog((l) => [...l, `${stage ? stage + ": " : ""}${message}`]),
    onScored: (s: ScoredEvent) => setScored(s),
    onDone: (d: DoneEvent) => {
      setDone(d);
      setPhase("done");
      setTimeout(() => setRevealed(true), 700);
    },
    onError: (message: string) => {
      setError(message);
      setPhase("error");
    },
  };

  async function start() {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setPhase("running");
    setLog([]);
    setScored(null);
    setDone(null);
    setRevealed(false);
    setError(null);
    setOffline(false);

    const reached = await runDemo(
      {
        query,
        answerer,
        video_url: videoUrl.trim() || undefined,
        video_path: videoUrl.trim() ? undefined : videoPath,
        visual_scorer: "clip_scene",
        // Explicit dispatcher (not "auto"): auto only runs transcript scoring on
        // videos >=10min, so short demo clips would otherwise fall back to
        // baseline and select zero audio, leaving the LLM with only slide frames.
        audio_scorer: "dispatcher",
        adaptive_budget: true,
        decompose_query: true,
        sample_count: 64,
        max_frames: 8,
        output_root: ".gist/demo-web",
      },
      handlers,
      controller.signal,
    );

    if (controller.signal.aborted || reached) return;

    // The live API was unreachable (sleeping Space / dead WiFi / hiccup). Fall
    // back to a pre-baked run for this preset so the demo never breaks on stage.
    const preset = PRESETS.find((p) => p.path === videoPath);
    const slug = videoUrl.trim() ? undefined : preset?.cachedSlug;
    const cached = slug ? await loadCachedRun(slug) : null;
    if (!cached) {
      setError(
        "The live API is unreachable and no cached run is available for this input. " +
          "Start the API (uvicorn) or pick a preset with a cached run.",
      );
      setPhase("error");
      return;
    }
    setOffline(true);
    await replayCachedRun(cached, handlers);
  }

  return (
    <main className="mx-auto max-w-4xl space-y-6 px-4 py-10">
      <header className="space-y-2">
        <div className="flex items-center gap-2">
          <h1 className="text-2xl font-semibold tracking-tight">Gist</h1>
          <Badge variant="outline">training-free AV compression</Badge>
        </div>
        <p className="text-muted-foreground text-sm">
          Gist scores every frame and audio window against your query and keeps only a
          small salient set <em>before</em> the encoders run. Watch it happen.
        </p>
      </header>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">1 · Ask a question about a video</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-1.5">
            <Label htmlFor="query">Query</Label>
            <Textarea
              id="query"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              rows={2}
            />
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-1.5">
              <Label>Preset video</Label>
              <div className="flex flex-wrap gap-2">
                {PRESETS.map((p) => (
                  <Button
                    key={p.path}
                    type="button"
                    size="sm"
                    variant={videoPath === p.path && !videoUrl ? "default" : "outline"}
                    onClick={() => {
                      setVideoPath(p.path);
                      setVideoUrl("");
                      setQuery(p.query);
                    }}
                  >
                    {p.label}
                  </Button>
                ))}
              </div>
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="url">…or a video URL (YouTube)</Label>
              <Input
                id="url"
                placeholder="https://youtube.com/watch?v=…"
                value={videoUrl}
                onChange={(e) => setVideoUrl(e.target.value)}
              />
            </div>
          </div>

          <div className="flex flex-wrap items-end justify-between gap-4">
            <div className="space-y-1.5">
              <Label>Answerer</Label>
              <Tabs value={answerer} onValueChange={(v) => setAnswerer(v as Answerer)}>
                <TabsList>
                  <TabsTrigger value="openai">OpenAI</TabsTrigger>
                  <TabsTrigger value="claude">Claude</TabsTrigger>
                  <TabsTrigger value="extractive">Extractive</TabsTrigger>
                </TabsList>
              </Tabs>
            </div>
            <Button onClick={start} disabled={running || !query.trim()}>
              {running ? "Running…" : "Run Gist"}
            </Button>
          </div>
        </CardContent>
      </Card>

      {(running || scored || done) && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex flex-wrap items-center gap-2 text-base">
              2 · Scoring &amp; selection
              {offline && (
                <Badge variant="outline" className="border-amber-500 text-amber-600">
                  cached run (offline)
                </Badge>
              )}
              {done && (
                <span className="text-muted-foreground ml-2 text-sm font-normal">
                  {done.compression.metrics.raw_input_candidates ??
                    done.compression.metrics.input_candidates}{" "}
                  candidates → {done.compression.metrics.selected_candidates} kept
                </span>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {scored ? (
              <ScoringTimeline
                visual={scored.visual}
                audio={scored.audio}
                selected={done?.compression.selected ?? []}
                revealed={revealed}
              />
            ) : (
              <p className="text-muted-foreground text-sm">Scoring frames and audio…</p>
            )}
            {running && (
              <pre className="bg-muted/40 text-muted-foreground max-h-28 overflow-auto rounded-md p-3 text-xs">
                {log.slice(-6).join("\n") || "starting…"}
              </pre>
            )}
          </CardContent>
        </Card>
      )}

      {done && (
        <section className="space-y-2">
          <h2 className="text-base font-semibold">3 · Answer from the compressed set</h2>
          <AnswerPanel done={done} />
        </section>
      )}

      {error && (
        <Card className="border-destructive/50">
          <CardContent className="text-destructive pt-6 text-sm">{error}</CardContent>
        </Card>
      )}
    </main>
  );
}
