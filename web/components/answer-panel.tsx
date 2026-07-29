"use client";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { DoneEvent } from "@/lib/types";

function StatBlock({
  value,
  label,
  accent,
}: {
  value: string;
  label: string;
  accent?: boolean;
}) {
  return (
    <div className="flex flex-col">
      <span
        className={
          accent
            ? "text-3xl font-semibold tabular-nums text-emerald-500"
            : "text-3xl font-semibold tabular-nums"
        }
      >
        {value}
      </span>
      <span className="text-muted-foreground text-xs">{label}</span>
    </div>
  );
}

export function AnswerPanel({ done }: { done: DoneEvent }) {
  const m = done.compression.metrics;
  const total = m.raw_input_candidates ?? m.input_candidates;
  const kept = m.selected_candidates;

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-base">
            Answer
            <Badge variant="secondary">{done.provider}</Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {done.answer ? (
            <p className="text-sm leading-relaxed">{done.answer}</p>
          ) : (
            <p className="text-muted-foreground text-sm italic leading-relaxed">
              No answer text for this run. The extractive answerer only summarizes
              evidence when a matching modality is present — re-bake with{" "}
              <code>--answerer openai</code> or <code>claude</code> for a full answer.
            </p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">What Gist saved</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <StatBlock value={`${total} → ${kept}`} label="candidates encoded" />
          <StatBlock
            value={`${m.estimated_candidate_reduction_percent.toFixed(0)}%`}
            label="fewer candidates"
            accent
          />
          <StatBlock
            value={`${m.estimated_token_reduction_percent.toFixed(0)}%`}
            label="fewer downstream tokens"
            accent
          />
          <StatBlock
            value={`${done.video.duration_seconds.toFixed(0)}s`}
            label="video length"
          />
        </CardContent>
      </Card>

      <p className="text-muted-foreground text-xs leading-relaxed">
        The live answer is produced by a hosted multimodal LLM ({done.provider}) for
        demo reliability. The capstone paper&apos;s measured encoder-FLOP savings come
        from Qwen2.5-Omni-7B run offline, not from this hosted API. Token figures shown
        here are estimated from the compressed evidence.
      </p>
    </div>
  );
}
