"use client";

import { ArrowLeft, ArrowUp, Loader2, Square } from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";

import { CollapseField } from "@/components/collapse-field";
import { EvidencePanel } from "@/components/evidence-panel";
import type {
  Clip,
  Message,
  Metrics,
  ScoredCandidate,
  SelectedEvidence,
  VideoDetail,
} from "@/lib/library";
import { formatDuration, streamQuery } from "@/lib/library";
import { cn } from "@/lib/utils";

type Turn = {
  id: string;
  query: string;
  answer: string | null;
  provider: string | null;
  candidates: ScoredCandidate[];
  selected: SelectedEvidence[] | null;
  clips: Clip[];
  metrics: Metrics | null;
  stage: string | null;
  error: string | null;
  elapsedMs: number | null;
};

/** Rebuild past turns from persisted messages so history survives a reload. */
function turnsFromMessages(messages: Message[]): Turn[] {
  const turns: Turn[] = [];
  for (const message of messages) {
    if (message.role === "user") {
      turns.push({
        id: message.id,
        query: message.query ?? "",
        answer: null,
        provider: null,
        candidates: [],
        selected: null,
        clips: [],
        metrics: null,
        stage: null,
        error: null,
        elapsedMs: null,
      });
      continue;
    }
    const current = turns[turns.length - 1];
    if (!current) continue;
    current.answer = message.answer;
    current.provider = message.answer_provider;
    current.selected = message.selected_evidence ?? null;
    current.clips = message.clips ?? [];
    current.metrics = message.metrics ?? null;
  }
  return turns;
}

export function VideoWorkspace({ initial }: { initial: VideoDetail }) {
  const { video } = initial;
  const [turns, setTurns] = useState<Turn[]>(() => turnsFromMessages(initial.messages));
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const abort = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement>(null);
  const textarea = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns.length]);

  const patch = useCallback((id: string, changes: Partial<Turn>) => {
    setTurns((current) =>
      current.map((turn) => (turn.id === id ? { ...turn, ...changes } : turn)),
    );
  }, []);

  const ask = useCallback(
    async (question: string) => {
      const trimmed = question.trim();
      if (!trimmed || busy) return;

      const id = `turn-${Date.now()}`;
      const startedAt = performance.now();
      setTurns((current) => [
        ...current,
        {
          id,
          query: trimmed,
          answer: null,
          provider: null,
          candidates: [],
          selected: null,
          clips: [],
          metrics: null,
          stage: "Scoring stored evidence",
          error: null,
          elapsedMs: null,
        },
      ]);
      setDraft("");
      setBusy(true);

      const controller = new AbortController();
      abort.current = controller;

      try {
        await streamQuery(
          video.id,
          trimmed,
          { answerer: "twelvelabs" },
          {
            onStage: (_stage, label) => patch(id, { stage: label }),
            onScored: (candidates) => patch(id, { candidates }),
            onSelected: (selected, metrics) => patch(id, { selected, metrics }),
            onClips: (clips) => patch(id, { clips }),
            onDone: (payload) =>
              patch(id, {
                answer: payload.answer,
                provider: payload.answer_provider,
                metrics: payload.metrics,
                clips: payload.clips,
                stage: null,
                elapsedMs: performance.now() - startedAt,
              }),
            onError: (message) => patch(id, { error: message, stage: null }),
          },
          controller.signal,
        );
      } catch (err) {
        if (!controller.signal.aborted) {
          patch(id, {
            error:
              err instanceof Error && !/fetch|Failed/.test(err.message)
                ? err.message
                : "Lost connection to the Gist API.",
            stage: null,
          });
        }
      } finally {
        setBusy(false);
        abort.current = null;
      }
    },
    [busy, patch, video.id],
  );

  return (
    <div className="mx-auto flex min-h-dvh w-full max-w-3xl flex-col px-6">
      <header className="sticky top-0 z-20 -mx-6 border-b border-border bg-background/85 px-6 py-3 backdrop-blur-sm">
        <div className="flex items-center gap-3">
          <Link
            href="/"
            aria-label="Back to library"
            className="rounded p-1 text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
          >
            <ArrowLeft className="size-4" aria-hidden />
          </Link>
          <div className="min-w-0 flex-1">
            <h1 className="truncate text-sm font-medium">{video.title}</h1>
            <p className="tabular text-xs text-muted-foreground">
              {formatDuration(video.duration_seconds)} · {video.frame_count} frames ·{" "}
              {video.audio_window_count} audio windows
            </p>
          </div>
        </div>
      </header>

      <div className="flex-1 py-8">
        {turns.length === 0 ? (
          <EmptyState onPick={ask} />
        ) : (
          <ol className="space-y-12">
            {turns.map((turn) => (
              <TurnView key={turn.id} turn={turn} duration={video.duration_seconds} />
            ))}
          </ol>
        )}
        <div ref={bottom} />
      </div>

      <div className="sticky bottom-0 -mx-6 bg-gradient-to-t from-background via-background to-transparent px-6 pb-6 pt-4">
        <form
          onSubmit={(event) => {
            event.preventDefault();
            void ask(draft);
          }}
        >
          <label htmlFor="question" className="sr-only">
            Ask about this video
          </label>
          <div
            className={cn(
              "flex items-end gap-2 rounded-xl border border-border bg-card p-2 pl-4 shadow-sm transition-colors",
              "focus-within:border-ring/60 focus-within:ring-[3px] focus-within:ring-ring/15",
            )}
          >
            <textarea
              id="question"
              ref={textarea}
              rows={1}
              value={draft}
              disabled={busy}
              onChange={(event) => {
                setDraft(event.target.value);
                const node = event.target;
                node.style.height = "auto";
                node.style.height = `${Math.min(node.scrollHeight, 160)}px`;
              }}
              onKeyDown={(event) => {
                // Enter sends; Shift+Enter is a newline. Standard for this shape
                // of input and what anyone will reflexively try.
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void ask(draft);
                }
              }}
              placeholder="Ask about this video"
              className="max-h-40 min-h-[36px] flex-1 resize-none bg-transparent py-1.5 text-[15px] leading-relaxed outline-none placeholder:text-muted-foreground/70 disabled:opacity-60"
            />
            {busy ? (
              <button
                type="button"
                onClick={() => abort.current?.abort()}
                aria-label="Stop"
                className="grid size-9 shrink-0 place-items-center rounded-lg bg-secondary text-secondary-foreground transition-colors hover:bg-accent focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
              >
                <Square className="size-3.5 fill-current" aria-hidden />
              </button>
            ) : (
              <button
                type="submit"
                disabled={draft.trim().length === 0}
                aria-label="Ask"
                className="grid size-9 shrink-0 place-items-center rounded-lg bg-primary text-primary-foreground transition-opacity hover:opacity-90 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring disabled:pointer-events-none disabled:opacity-30"
              >
                <ArrowUp className="size-4" aria-hidden />
              </button>
            )}
          </div>
        </form>
      </div>
    </div>
  );
}

function EmptyState({ onPick }: { onPick: (query: string) => void }) {
  const suggestions = [
    "What is the main argument?",
    "What appears on screen at the start?",
    "What does the speaker say about the results?",
  ];
  return (
    <div className="py-10">
      <p className="text-[15px] font-medium">Ask anything about this video.</p>
      <p className="mt-1.5 max-w-md text-sm leading-relaxed text-muted-foreground">
        Gist scores every frame and audio window against your question, keeps the
        few that matter, and shows you the clip it answered from.
      </p>
      <ul className="mt-5 flex flex-wrap gap-2">
        {suggestions.map((suggestion) => (
          <li key={suggestion}>
            <button
              type="button"
              onClick={() => onPick(suggestion)}
              className="rounded-full border border-border px-3 py-1.5 text-[13px] text-muted-foreground transition-colors hover:border-ring/50 hover:text-foreground focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
            >
              {suggestion}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

function TurnView({ turn, duration }: { turn: Turn; duration: number }) {
  const phase = turn.selected ? "collapsed" : turn.candidates.length > 0 ? "scoring" : "idle";

  return (
    <li className="space-y-5">
      <h2 className="text-balance text-lg font-medium leading-snug tracking-tight">
        {turn.query}
      </h2>

      {turn.error ? (
        <p
          role="alert"
          className="rounded-md border border-destructive/30 bg-destructive/5 px-4 py-3 text-sm text-destructive"
        >
          {turn.error}
        </p>
      ) : (
        <>
          {(turn.candidates.length > 0 || turn.stage) && (
            <CollapseField
              candidates={turn.candidates}
              selected={turn.selected}
              durationSeconds={duration}
              phase={phase}
            />
          )}

          {turn.stage && (
            <p className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="size-3.5 animate-spin" aria-hidden />
              {turn.stage}
            </p>
          )}

          {turn.answer && (
            <div className="space-y-1.5">
              <p className="whitespace-pre-wrap text-[15px] leading-[1.7]">
                {turn.answer}
              </p>
              <p className="tabular text-[11px] text-muted-foreground">
                {turn.provider && <>answered by {turn.provider}</>}
                {turn.elapsedMs !== null && (
                  <> · {(turn.elapsedMs / 1000).toFixed(1)}s</>
                )}
              </p>
            </div>
          )}

          {turn.selected && turn.selected.length > 0 && (
            <EvidencePanel
              selected={turn.selected}
              clips={turn.clips}
              metrics={turn.metrics}
            />
          )}
        </>
      )}
    </li>
  );
}
