"use client";

import { cn } from "@/lib/utils";
import type { CandidatePoint, SelectedCandidate } from "@/lib/types";

interface TrackProps {
  label: string;
  hint: string;
  points: CandidatePoint[];
  selectedKeys: Set<string>;
  modality: "visual" | "audio";
  revealed: boolean;
}

function keyFor(modality: string, t: number): string {
  return `${modality}@${t.toFixed(2)}`;
}

function normalizedHeights(points: CandidatePoint[]): number[] {
  const scores = points.map((p) => p.score);
  const min = Math.min(...scores);
  const max = Math.max(...scores);
  const span = max - min;
  return points.map((p) => {
    if (span <= 1e-9) return 0.45; // flat scores -> uniform mid bars
    return 0.15 + 0.85 * ((p.score - min) / span);
  });
}

function Track({ label, hint, points, selectedKeys, modality, revealed }: TrackProps) {
  if (points.length === 0) return null;
  const heights = normalizedHeights(points);

  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline justify-between">
        <span className="text-sm font-medium">{label}</span>
        <span className="text-muted-foreground text-xs">{hint}</span>
      </div>
      <div className="bg-muted/40 flex h-24 items-end gap-[2px] rounded-md p-2">
        {points.map((p, i) => {
          const selected = selectedKeys.has(keyFor(modality, p.timestamp_seconds));
          const dim = revealed && !selected;
          return (
            <div
              key={`${modality}-${i}`}
              title={`t=${p.timestamp_seconds.toFixed(1)}s  score=${p.score.toFixed(3)}${
                p.text ? `\n${p.text}` : ""
              }`}
              className={cn(
                "flex-1 rounded-sm transition-all duration-500",
                modality === "visual" ? "bg-sky-500" : "bg-violet-500",
                dim && "opacity-15",
                revealed && selected && "ring-2 ring-offset-1 ring-offset-background",
                revealed && selected && modality === "visual" && "ring-sky-400",
                revealed && selected && modality === "audio" && "ring-violet-400",
              )}
              style={{ height: `${Math.round(heights[i] * 100)}%` }}
            />
          );
        })}
      </div>
    </div>
  );
}

interface ScoringTimelineProps {
  visual: CandidatePoint[];
  audio: CandidatePoint[];
  selected: SelectedCandidate[];
  revealed: boolean;
}

export function ScoringTimeline({ visual, audio, selected, revealed }: ScoringTimelineProps) {
  const selectedKeys = new Set(
    selected.map((s) => keyFor(s.modality, s.timestamp_seconds)),
  );

  return (
    <div className="space-y-4">
      <Track
        label="Video frames"
        hint="CLIP relevance to your query"
        points={visual}
        selectedKeys={selectedKeys}
        modality="visual"
        revealed={revealed}
      />
      <Track
        label="Audio windows"
        hint="CLAP / Whisper dispatcher"
        points={audio}
        selectedKeys={selectedKeys}
        modality="audio"
        revealed={revealed}
      />
    </div>
  );
}
