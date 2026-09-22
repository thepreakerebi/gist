"use client";

import { AlertCircle, Play, RotateCcw, Trash2 } from "lucide-react";
import Link from "next/link";
import { useState } from "react";

import { VideoPlayerModal } from "@/components/video-player-modal";
import type { Video } from "@/lib/library";
import { deleteVideo, formatDuration } from "@/lib/library";
import { cn } from "@/lib/utils";

/**
 * One row of the library.
 *
 * A list rather than a grid of identical cards: rows scan faster, keep the
 * thumbnail at a consistent size, and leave the page's visual weight with the
 * heading and the add field instead of scattering it across tiles.
 *
 * The row has two distinct targets — the thumbnail plays the source video, the
 * text opens the chat. They are siblings rather than nested, because an
 * interactive element inside a link is invalid HTML and behaves unpredictably
 * with keyboard and middle-click.
 */
export function VideoRow({
  video,
  onRemoved,
  onRetry,
}: {
  video: Video;
  onRemoved?: (id: string) => void;
  onRetry?: (video: Video) => Promise<void>;
}) {
  const [removing, setRemoving] = useState(false);
  const [retrying, setRetrying] = useState(false);
  const [playing, setPlaying] = useState(false);
  const ready = video.status === "ready";
  const failed = video.status === "failed";

  // A failed row is the one case where the actions must be visible without
  // hovering: the row is a dead end until the user acts on it, and an affordance
  // that only appears on hover is one a person may never find. Ingestion most
  // often fails because yt-dlp has gone stale against YouTube, which a retry
  // fixes outright once it is updated.
  const busy = removing || retrying;

  async function retry() {
    if (busy || !onRetry) return;
    setRetrying(true);
    try {
      await onRetry(video);
    } catch {
      setRetrying(false);
    }
  }

  async function remove() {
    if (busy) return;
    setRemoving(true);
    try {
      await deleteVideo(video.id);
      onRemoved?.(video.id);
    } catch {
      setRemoving(false);
    }
  }

  return (
    <li
      className={cn(
        "group/row flex items-center gap-4 py-3.5 transition-opacity",
        busy && "pointer-events-none opacity-40",
      )}
    >
      {ready ? (
        <button
          type="button"
          onClick={() => setPlaying(true)}
          aria-label={`Play ${video.title}`}
          className="group/play relative shrink-0 overflow-hidden rounded focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
        >
          <Thumbnail video={video} />
          <span
            className="absolute inset-0 grid place-items-center bg-black/0 transition-colors group-hover/play:bg-black/45"
            aria-hidden
          >
            <Play className="size-6 fill-white text-white opacity-0 transition-opacity group-hover/play:opacity-100" />
          </span>
        </button>
      ) : (
        <div className="shrink-0">
          <Thumbnail video={video} />
        </div>
      )}

      <div className="min-w-0 flex-1">
        {ready ? (
          <Link
            href={`/v/${video.id}`}
            className="rounded focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
          >
            <p className="truncate text-[15px] font-medium leading-snug hover:underline underline-offset-4">
              {video.title}
            </p>
          </Link>
        ) : (
          <p className="truncate text-[15px] font-medium leading-snug text-muted-foreground">
            {video.title}
          </p>
        )}

        {ready ? (
          <p className="mt-1 flex items-center gap-2 text-xs text-muted-foreground">
            <span className="tabular">{formatDuration(video.duration_seconds)}</span>
            <Dot />
            <span className="tabular">{video.frame_count} frames</span>
            <Dot />
            <span className="tabular">{video.audio_window_count} audio windows</span>
          </p>
        ) : failed ? (
          <p className="mt-1 flex items-start gap-1.5 text-xs text-destructive">
            <AlertCircle className="mt-px size-3.5 shrink-0" aria-hidden />
            <span className="line-clamp-2">{video.error ?? "Ingestion failed"}</span>
          </p>
        ) : (
          <IngestionProgress video={video} />
        )}
      </div>

      {(onRetry || onRemoved) && (
        <div className="flex shrink-0 items-center gap-0.5">
          {failed && onRetry && (
            <button
              type="button"
              onClick={retry}
              disabled={busy}
              aria-label={`Retry ${video.title}`}
              title="Try again"
              className={cn(
                "rounded p-1.5 text-muted-foreground transition-colors hover:text-foreground",
                "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring",
              )}
            >
              <RotateCcw className={cn("size-3.5", retrying && "animate-spin")} aria-hidden />
            </button>
          )}
          {onRemoved && (
            <button
              type="button"
              onClick={remove}
              disabled={busy}
              aria-label={`Remove ${video.title}`}
              title="Remove"
              className={cn(
                "rounded p-1.5 transition-colors hover:!text-destructive",
                "focus-visible:text-muted-foreground focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring",
                failed
                  ? "text-muted-foreground"
                  : "text-transparent group-hover/row:text-muted-foreground",
              )}
            >
              <Trash2 className="size-3.5" aria-hidden />
            </button>
          )}
        </div>
      )}

      {ready && (
        <VideoPlayerModal
          videoId={video.id}
          title={video.title}
          open={playing}
          onClose={() => setPlaying(false)}
        />
      )}
    </li>
  );
}

function Dot() {
  return (
    <span className="text-border" aria-hidden>
      ·
    </span>
  );
}

function Thumbnail({ video }: { video: Video }) {
  const ready = video.status === "ready";
  return (
    <div
      className={cn(
        "relative aspect-video w-24 overflow-hidden rounded bg-muted sm:w-28",
        !ready && "opacity-60",
      )}
    >
      {video.thumbnail_url ? (
        // Remote YouTube thumbnails: a plain img avoids configuring a remote
        // image host for what is a fixed, small, non-critical asset.
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={video.thumbnail_url}
          alt=""
          loading="lazy"
          className="size-full object-cover"
        />
      ) : (
        <div className="size-full bg-gradient-to-br from-muted to-secondary" />
      )}
    </div>
  );
}

function IngestionProgress({ video }: { video: Video }) {
  const percent = Math.round((video.progress ?? 0) * 100);
  return (
    <div className="mt-1.5 max-w-xs">
      <div className="flex items-baseline justify-between gap-3">
        <span className="truncate text-xs text-muted-foreground">
          {video.status_detail ?? "Queued"}
        </span>
        <span className="tabular text-xs text-muted-foreground">{percent}%</span>
      </div>
      <div
        className="mt-1.5 h-0.5 w-full overflow-hidden rounded-full bg-border"
        role="progressbar"
        aria-valuenow={percent}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={`Processing ${video.title}`}
      >
        <div
          className="h-full rounded-full bg-signal transition-[width] duration-700 ease-out-expo"
          style={{ width: `${Math.max(percent, 2)}%` }}
        />
      </div>
    </div>
  );
}
