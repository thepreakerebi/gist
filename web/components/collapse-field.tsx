"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import type { ScoredCandidate, SelectedEvidence } from "@/lib/library";
import { formatTimestamp } from "@/lib/library";
import { cn } from "@/lib/utils";

/**
 * The argument, drawn.
 *
 * Every sampled frame and audio window appears as a mark on a shared time
 * axis, its height set by how well it scored against the question — visual
 * above the line, audio below, so modality never depends on colour alone.
 * Then the selection arrives and the field *collapses*: the hundreds that lost
 * drain to a flat neutral, the handful that survived rise and take the signal
 * colour.
 *
 * That transition is the whole thesis in one second of motion, so it is
 * deliberately the only place in the interface with choreography: a staggered
 * transform/opacity animation, exponential deceleration, and no bounce.
 */

type Phase = "idle" | "scoring" | "collapsed";

type Props = {
  candidates: ScoredCandidate[];
  selected: SelectedEvidence[] | null;
  durationSeconds: number;
  phase: Phase;
  className?: string;
};

const HEIGHT = 132;
const AXIS = HEIGHT / 2;
const MAX_ARM = AXIS - 14;

export function CollapseField({
  candidates,
  selected,
  durationSeconds,
  phase,
  className,
}: Props) {
  const selectedIds = useMemo(
    () => new Set((selected ?? []).map((item) => item.id)),
    [selected],
  );
  const [hovered, setHovered] = useState<string | null>(null);

  // Scores are unnormalized cosine similarities clustered in a narrow band, so
  // a raw 0..1 mapping would render every mark at nearly the same height and
  // hide exactly the contrast this chart exists to show. Rescale to the
  // observed range instead.
  const bounds = useMemo(() => {
    const scores = candidates
      .map((candidate) => candidate.score)
      .filter((score): score is number => typeof score === "number");
    if (scores.length === 0) return { min: 0, max: 1 };
    const min = Math.min(...scores);
    const max = Math.max(...scores);
    return max - min < 1e-6 ? { min, max: min + 1e-6 } : { min, max };
  }, [candidates]);

  const marks = useMemo(() => {
    const span = Math.max(durationSeconds, 1);
    return candidates.map((candidate, index) => {
      const normalized =
        typeof candidate.score === "number"
          ? (candidate.score - bounds.min) / (bounds.max - bounds.min)
          : 0.12;
      return {
        key: `${candidate.modality}-${candidate.id}-${index}`,
        id: candidate.id,
        modality: candidate.modality,
        left: Math.min(100, Math.max(0, (candidate.timestamp_seconds / span) * 100)),
        arm: 6 + normalized * (MAX_ARM - 6),
        timestamp: candidate.timestamp_seconds,
        text: candidate.text,
        survived: selectedIds.has(candidate.id),
      };
    });
  }, [candidates, bounds, durationSeconds, selectedIds]);

  const survivors = marks.filter((mark) => mark.survived).length;

  return (
    <figure className={cn("space-y-2", className)}>
      <figcaption className="flex items-baseline justify-between gap-4 text-xs">
        <span className="font-medium text-foreground">
          {phase === "collapsed" ? "Evidence kept" : "Scoring every candidate"}
        </span>
        <span className="tabular text-muted-foreground">
          {phase === "collapsed" && candidates.length > 0 ? (
            <>
              <span className="text-signal font-semibold">{survivors}</span>
              {" of "}
              {candidates.length}
            </>
          ) : (
            `${candidates.length} candidates`
          )}
        </span>
      </figcaption>

      <div
        className="relative w-full overflow-hidden rounded-md border border-border bg-card"
        style={{ height: HEIGHT }}
        onMouseLeave={() => setHovered(null)}
      >
        {/* Time axis. Visual candidates sit above it, audio below. */}
        <div
          className="absolute inset-x-0 border-t border-border/80"
          style={{ top: AXIS }}
          aria-hidden
        />

        {marks.map((mark, index) => {
          const isUp = mark.modality === "visual";
          const collapsed = phase === "collapsed";
          const dimmed = collapsed && !mark.survived;
          // Dropped candidates keep a scaled-down silhouette rather than
          // flattening to a line. Flattening made the collapse read as
          // "everything vanished"; retaining the terrain makes it read as
          // "these few stood out of that", which is the actual claim.
          const arm = dimmed ? Math.max(3, mark.arm * 0.28) : mark.arm;

          return (
            <button
              key={mark.key}
              type="button"
              onMouseEnter={() => setHovered(mark.key)}
              onFocus={() => setHovered(mark.key)}
              onBlur={() => setHovered(null)}
              aria-label={`${mark.modality} candidate at ${formatTimestamp(mark.timestamp)}${
                mark.survived ? ", selected as evidence" : ""
              }`}
              className={cn(
                "absolute -translate-x-1/2 rounded-full transition-all duration-700 ease-out-expo",
                "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring",
                // Survivors are wider as well as coloured, so the distinction
                // survives greyscale printing and colour-blind viewers.
                mark.survived ? "w-[5px] bg-signal" : "w-[3px]",
                !mark.survived && (collapsed ? "bg-dropped" : "bg-foreground/35"),
                hovered === mark.key && !mark.survived && "w-[5px] bg-foreground",
              )}
              style={{
                left: `${mark.left}%`,
                height: arm,
                top: isUp ? AXIS - arm : AXIS,
                opacity: dimmed ? 0.75 : 1,
                // Stagger by position so the collapse reads as a wave across
                // the timeline rather than everything snapping at once.
                transitionDelay: collapsed ? `${Math.min(index * 6, 320)}ms` : "0ms",
              }}
            />
          );
        })}

        {/* Survivors get a dot at the tip: after the collapse they should be
            findable without reading heights. */}
        {phase === "collapsed" &&
          marks
            .filter((mark) => mark.survived)
            .map((mark) => (
              <span
                key={`dot-${mark.key}`}
                aria-hidden
                className="absolute size-1.5 -translate-x-1/2 rounded-full bg-signal ring-2 ring-card"
                style={{
                  left: `${mark.left}%`,
                  top:
                    mark.modality === "visual"
                      ? AXIS - mark.arm - 3
                      : AXIS + mark.arm - 3,
                }}
              />
            ))}

        {candidates.length === 0 && (
          <p className="absolute inset-0 grid place-items-center text-xs text-muted-foreground">
            Ask a question to score this video&rsquo;s evidence
          </p>
        )}

        <HoverLabel marks={marks} hovered={hovered} />
      </div>

      <div className="flex items-center justify-between text-[11px] text-muted-foreground">
        <span className="tabular">0:00</span>
        <Legend />
        <span className="tabular">{formatTimestamp(durationSeconds)}</span>
      </div>
    </figure>
  );
}

function Legend() {
  return (
    <span className="flex items-center gap-3">
      <span className="flex items-center gap-1.5">
        <span className="h-2.5 w-[3px] rounded-full bg-foreground/35" aria-hidden />
        above: frames
      </span>
      <span className="flex items-center gap-1.5">
        <span className="h-2.5 w-[3px] rounded-full bg-foreground/35" aria-hidden />
        below: audio
      </span>
      <span className="flex items-center gap-1.5">
        <span className="h-2.5 w-[3px] rounded-full bg-signal" aria-hidden />
        kept
      </span>
    </span>
  );
}

function HoverLabel({
  marks,
  hovered,
}: {
  marks: { key: string; left: number; timestamp: number; modality: string; text: string }[];
  hovered: string | null;
}) {
  const mark = marks.find((item) => item.key === hovered);
  const ref = useRef<HTMLDivElement>(null);
  const [flip, setFlip] = useState(false);

  useEffect(() => {
    if (!mark || !ref.current) return;
    // Keep the label inside the plot when hovering near the right edge.
    setFlip(mark.left > 70);
  }, [mark]);

  if (!mark) return null;

  return (
    <div
      ref={ref}
      className={cn(
        "pointer-events-none absolute top-1.5 z-10 max-w-[16rem] rounded border border-border bg-popover px-2 py-1 text-[11px] shadow-sm",
        flip ? "-translate-x-full" : "translate-x-1",
      )}
      style={{ left: `${mark.left}%` }}
    >
      <span className="tabular font-medium">{formatTimestamp(mark.timestamp)}</span>
      <span className="text-muted-foreground"> · {mark.modality}</span>
      {mark.text && (
        <p className="mt-0.5 line-clamp-2 text-muted-foreground">{mark.text}</p>
      )}
    </div>
  );
}
