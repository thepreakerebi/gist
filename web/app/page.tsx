"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";

import { AddVideo } from "@/components/add-video";
import { VideoRow } from "@/components/video-row";
import type { Video } from "@/lib/library";
import { listVideos, streamIngestion } from "@/lib/library";

export default function LibraryPage() {
  const [videos, setVideos] = useState<Video[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [offline, setOffline] = useState(false);
  const watching = useRef(new Set<string>());

  const upsert = useCallback((video: Video) => {
    setVideos((current) => {
      const index = current.findIndex((item) => item.id === video.id);
      if (index === -1) return [video, ...current];
      const next = [...current];
      next[index] = video;
      return next;
    });
  }, []);

  // Follow any video that is still ingesting, including ones a previous session
  // started: progress lives in the database, so a reload picks the stream back up.
  const watch = useCallback(
    (id: string) => {
      if (watching.current.has(id)) return;
      watching.current.add(id);
      streamIngestion(id, {
        onProgress: upsert,
        onDone: (video) => {
          upsert(video);
          watching.current.delete(id);
        },
        onError: () => watching.current.delete(id),
      }).catch(() => watching.current.delete(id));
    },
    [upsert],
  );

  useEffect(() => {
    let cancelled = false;
    listVideos()
      .then(({ data, cached }) => {
        if (cancelled) return;
        setVideos(data);
        setOffline(cached);
        // Nothing is still ingesting in a snapshot, and there is no API to
        // stream progress from, so skip the watchers entirely when cached.
        if (cached) return;
        data
          .filter((item) => item.status === "ingesting" || item.status === "pending")
          .forEach((item) => watch(item.id));
      })
      .catch((err: Error) => !cancelled && setError(err.message))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [watch]);

  const ready = videos.filter((video) => video.status === "ready");
  const working = videos.filter((video) => video.status !== "ready");

  return (
    <main className="mx-auto w-full max-w-3xl px-6 pb-24 pt-16 sm:pt-24">
      <header className="mb-12">
        <Link href="/" className="text-sm font-semibold tracking-tight">
          Gist
        </Link>
        <h1 className="mt-8 max-w-xl text-balance text-3xl font-semibold leading-[1.15] tracking-tight sm:text-4xl">
          Ask questions about hours of video.
        </h1>
        <p className="mt-3 max-w-lg text-[15px] leading-relaxed text-muted-foreground">
          Add a video once. Gist watches and listens to all of it, then keeps only
          the few seconds that answer each question you ask.
        </p>
      </header>

      <AddVideo
        onAdded={(video) => {
          upsert(video);
          watch(video.id);
        }}
      />

      <section className="mt-14" aria-labelledby="library-heading">
        <h2
          id="library-heading"
          className="mb-1 text-xs font-medium uppercase tracking-[0.08em] text-muted-foreground"
        >
          Library
        </h2>

        {offline && (
          <p className="mt-4 rounded-md border border-border bg-secondary/60 px-4 py-2.5 text-sm text-muted-foreground">
            <span className="font-medium text-foreground">Offline.</span>{" "}
            Showing a pre-recorded snapshot of real runs — the API isn&rsquo;t
            reachable.
          </p>
        )}

        {error && (
          <p className="mt-6 rounded-md border border-destructive/30 bg-destructive/5 px-4 py-3 text-sm text-destructive">
            {error.includes("fetch") || error.includes("Failed")
              ? "Can't reach the Gist API. Start it with: uvicorn gist.api.app:app --port 8000"
              : error}
          </p>
        )}

        {loading && !error && (
          <div className="mt-4 space-y-px" aria-busy>
            {[0, 1, 2].map((index) => (
              <div
                key={index}
                className="h-[74px] animate-pulse rounded-md bg-muted/60"
                style={{ animationDelay: `${index * 90}ms` }}
              />
            ))}
          </div>
        )}

        {!loading && !error && videos.length === 0 && (
          <div className="mt-4 rounded-lg border border-dashed border-border px-6 py-12 text-center">
            <p className="text-sm font-medium">Nothing here yet</p>
            <p className="mx-auto mt-1.5 max-w-sm text-sm leading-relaxed text-muted-foreground">
              Paste a YouTube link above. A one-hour video takes a few minutes to
              process; after that every question is answered in about a second.
            </p>
          </div>
        )}

        {working.length > 0 && (
          <ul className="mt-4 divide-y divide-border/70">
            {working.map((video) => (
              <VideoRow key={video.id} video={video} />
            ))}
          </ul>
        )}

        {ready.length > 0 && (
          <ul className="mt-4 divide-y divide-border/70">
            {ready.map((video) => (
              <VideoRow
                key={video.id}
                video={video}
                onRemoved={(id) =>
                  setVideos((current) => current.filter((item) => item.id !== id))
                }
              />
            ))}
          </ul>
        )}
      </section>
    </main>
  );
}
