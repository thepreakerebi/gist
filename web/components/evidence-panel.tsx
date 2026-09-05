"use client";

import { useState } from "react";

import type { Clip, Metrics, SelectedEvidence } from "@/lib/library";
import { apiUrl, formatTimestamp } from "@/lib/library";
import { cn } from "@/lib/utils";

/**
 * What Gist kept, and the video to prove it.
 *
 * The panel exists to make an answer auditable: each retained piece of evidence
 * shows its timestamp, why the selector kept it, and the actual clip cut from
 * the source at that moment. A panel member should be able to check any claim
 * without leaving the page.
 */
export function EvidencePanel({
  selected,
  clips,
  metrics,
}: {
  selected: SelectedEvidence[];
  clips: Clip[];
  metrics: Metrics | null;
}) {
  if (selected.length === 0) return null;

  const clipFor = (id: string) => clips.find((clip) => clip.candidate_id === id);

  return (
    <section className="space-y-3">
      {metrics && <Savings metrics={metrics} />}

      <ol className="space-y-2">
        {selected.map((item) => (
          <EvidenceItem key={item.id} item={item} clip={clipFor(item.id)} />
        ))}
      </ol>
    </section>
  );
}

function Savings({ metrics }: { metrics: Metrics }) {
  const kept = metrics.selected_candidates;
  const total = metrics.input_candidates;
  return (
    <p className="text-xs leading-relaxed text-muted-foreground">
      Kept <span className="tabular font-medium text-foreground">{kept}</span> of{" "}
      <span className="tabular">{total}</span> candidates
      {metrics.estimated_token_reduction_percent > 0 && (
        <>
          {" · "}
          <span className="tabular font-medium text-foreground">
            {metrics.estimated_token_reduction_percent.toFixed(1)}%
          </span>{" "}
          fewer tokens sent onward
        </>
      )}
      {(metrics.budget_stages_used ?? 1) > 1 && (
        <> · budget escalated to stage {metrics.budget_stages_used}</>
      )}
    </p>
  );
}

/**
 * A frame with no speech and no on-screen text still needs a readable label.
 * Left blank the row looks broken, when in fact the frame itself is the
 * evidence — so say that, and let the clip below carry the content.
 */
function describeEvidence(item: SelectedEvidence): string {
  return item.modality === "audio"
    ? "Audio with no transcribed speech"
    : "Frame selected on visual match — no on-screen text";
}

function EvidenceItem({ item, clip }: { item: SelectedEvidence; clip?: Clip }) {
  const [open, setOpen] = useState(false);

  return (
    <li className="overflow-hidden rounded-md border border-border bg-card">
      <button
        type="button"
        onClick={() => setOpen((value) => !open)}
        aria-expanded={open}
        className={cn(
          "flex w-full items-start gap-3 px-3 py-2.5 text-left transition-colors hover:bg-accent/60",
          "focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-ring",
        )}
      >
        <span
          className="mt-[3px] h-3 w-[3px] shrink-0 rounded-full bg-signal"
          aria-hidden
        />
        <span className="min-w-0 flex-1">
          <span className="flex items-baseline gap-2">
            <span className="tabular text-xs font-medium">
              {formatTimestamp(item.timestamp_seconds)}
            </span>
            <span className="text-[11px] uppercase tracking-wide text-muted-foreground">
              {item.modality}
            </span>
            {(item.merged_from_count ?? 0) > 0 && (
              <span className="rounded-sm bg-secondary px-1.5 py-px text-[10px] text-secondary-foreground">
                merged ×{(item.merged_from_count ?? 0) + 1}
              </span>
            )}
          </span>
          <span className="mt-1 line-clamp-2 block text-[13px] leading-relaxed text-muted-foreground">
            {item.text || describeEvidence(item)}
          </span>
        </span>
      </button>

      {/* grid-template-rows rather than height: animating height forces layout
          on every frame, and 0fr→1fr transitions cleanly without a fixed size. */}
      <div
        className="grid transition-[grid-template-rows] duration-500 ease-out-expo"
        style={{ gridTemplateRows: open ? "1fr" : "0fr" }}
      >
        <div className="overflow-hidden">
          <div className="space-y-2.5 border-t border-border px-3 py-3">
            {clip ? (
              <>
                <video
                  key={clip.url}
                  // Cached clips are static files served by Next under
                  // /cached-runs/; only live clips go through the API host.
                  src={clip.url.startsWith("/cached-runs/") ? clip.url : apiUrl(clip.url)}
                  controls
                  preload="metadata"
                  playsInline
                  className="w-full rounded bg-black"
                />
                <p className="tabular text-[11px] text-muted-foreground">
                  {formatTimestamp(clip.start_seconds)} –{" "}
                  {formatTimestamp(clip.end_seconds)} of the source
                </p>
              </>
            ) : (
              <p className="text-[13px] text-muted-foreground">
                No clip for this candidate.
              </p>
            )}
            <p className="text-[13px] leading-relaxed">
              <span className="text-muted-foreground">Why it was kept: </span>
              {item.reason}
            </p>
          </div>
        </div>
      </div>
    </li>
  );
}
