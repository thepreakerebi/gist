"use client";

import { ArrowRight, Loader2 } from "lucide-react";
import { useState } from "react";

import type { Video } from "@/lib/library";
import { addVideo } from "@/lib/library";
import { cn } from "@/lib/utils";

/**
 * The single entry point to the library.
 *
 * Kept as one unadorned field rather than a card: this is the primary action on
 * the page and wrapping it in a container would make it compete with the
 * library list rather than lead it.
 */
export function AddVideo({ onAdded }: { onAdded: (video: Video) => void }) {
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    const trimmed = url.trim();
    if (!trimmed || busy) return;

    setBusy(true);
    setError(null);
    setNote(null);
    try {
      const { video, started } = await addVideo(trimmed);
      onAdded(video);
      setUrl("");
      if (!started) setNote("Already in your library.");
    } catch (err) {
      setError(
        err instanceof Error && !/fetch|Failed/.test(err.message)
          ? err.message
          : "Can't reach the Gist API. Start it with: uvicorn gist.api.app:app --port 8000",
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} noValidate>
      <label htmlFor="video-url" className="sr-only">
        YouTube video URL
      </label>
      <div
        className={cn(
          "flex items-center gap-2 rounded-lg border bg-card pl-4 pr-2 transition-colors",
          "focus-within:border-ring/60 focus-within:ring-[3px] focus-within:ring-ring/15",
          error ? "border-destructive/50" : "border-border",
        )}
      >
        <input
          id="video-url"
          type="url"
          inputMode="url"
          value={url}
          onChange={(event) => setUrl(event.target.value)}
          placeholder="Paste a YouTube link"
          disabled={busy}
          className="h-12 min-w-0 flex-1 bg-transparent text-[15px] outline-none placeholder:text-muted-foreground/70 disabled:opacity-60"
        />
        <button
          type="submit"
          disabled={busy || url.trim().length === 0}
          className={cn(
            "inline-flex h-9 items-center gap-1.5 rounded-md px-3 text-sm font-medium transition-all",
            "bg-primary text-primary-foreground hover:opacity-90",
            "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring",
            "disabled:pointer-events-none disabled:opacity-40",
          )}
        >
          {busy ? (
            <Loader2 className="size-4 animate-spin" aria-hidden />
          ) : (
            <ArrowRight className="size-4" aria-hidden />
          )}
          <span>{busy ? "Adding" : "Add"}</span>
        </button>
      </div>

      {error && (
        <p role="alert" className="mt-2 text-sm text-destructive">
          {error}
        </p>
      )}
      {note && !error && <p className="mt-2 text-sm text-muted-foreground">{note}</p>}
    </form>
  );
}
