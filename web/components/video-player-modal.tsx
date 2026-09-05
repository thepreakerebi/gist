"use client";

import { X } from "lucide-react";
import { useEffect, useRef } from "react";

import { apiUrl } from "@/lib/library";

/**
 * Full-source playback for a library entry.
 *
 * Built on the native <dialog> rather than a div-with-a-backdrop: showModal()
 * brings focus trapping, Escape-to-close, inertness of the page behind, and
 * the top layer with no JavaScript of our own. Reimplementing those by hand is
 * where custom modals usually get accessibility wrong.
 */
export function VideoPlayerModal({
  videoId,
  title,
  open,
  onClose,
}: {
  videoId: string;
  title: string;
  open: boolean;
  onClose: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const video = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const node = dialog.current;
    if (!node) return;

    if (open && !node.open) {
      node.showModal();
    } else if (!open && node.open) {
      node.close();
    }
  }, [open]);

  // Escape and the backdrop both fire the dialog's own close event, so pausing
  // and rewinding here covers every exit path rather than just the button.
  useEffect(() => {
    const node = dialog.current;
    if (!node) return;

    const handleClose = () => {
      const player = video.current;
      if (player) {
        player.pause();
        player.currentTime = 0;
      }
      onClose();
    };
    node.addEventListener("close", handleClose);
    return () => node.removeEventListener("close", handleClose);
  }, [onClose]);

  return (
    <dialog
      ref={dialog}
      aria-label={`Play ${title}`}
      onClick={(event) => {
        // A click landing on the dialog element itself is the backdrop; clicks
        // on the content bubble from a child and must not close it.
        if (event.target === dialog.current) dialog.current?.close();
      }}
      className="m-auto w-[min(56rem,92vw)] rounded-xl border border-border bg-card p-0 text-foreground shadow-2xl backdrop:bg-black/60 backdrop:backdrop-blur-sm"
    >
      <div className="flex items-center gap-3 border-b border-border px-4 py-3">
        <h2 className="min-w-0 flex-1 truncate text-sm font-medium">{title}</h2>
        <button
          type="button"
          onClick={() => dialog.current?.close()}
          aria-label="Close"
          className="rounded p-1 text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
        >
          <X className="size-4" aria-hidden />
        </button>
      </div>

      {/* Mounted only while open so the browser does not fetch the source for
          every row in the library on page load. */}
      {open && (
        <video
          ref={video}
          src={apiUrl(`/v1/library/videos/${videoId}/source`)}
          controls
          autoPlay
          playsInline
          preload="metadata"
          className="aspect-video w-full bg-black"
        />
      )}
    </dialog>
  );
}
