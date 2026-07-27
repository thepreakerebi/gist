// Streaming client for the Gist demo API. The endpoint POSTs a JSON body and
// replies with Server-Sent Events, so native EventSource (GET-only) can't be
// used; we read the fetch body stream and parse SSE frames by hand.

import type {
  DoneEvent,
  RunRequest,
  ScoredEvent,
} from "@/lib/types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") ?? "http://localhost:8000";

export interface StreamHandlers {
  onProgress?: (stage: string, message: string) => void;
  onScored?: (scored: ScoredEvent) => void;
  onDone?: (done: DoneEvent) => void;
  onError?: (message: string) => void;
}

export async function runDemo(
  request: RunRequest,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/v1/demo/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
      signal,
    });
  } catch (err) {
    handlers.onError?.(
      `Could not reach the Gist API at ${API_BASE}. ${(err as Error).message}`,
    );
    return;
  }

  if (!response.ok || !response.body) {
    handlers.onError?.(`API returned ${response.status}`);
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let sep: number;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      dispatch(frame, handlers);
    }
  }
}

function dispatch(frame: string, handlers: StreamHandlers): void {
  let event = "message";
  const dataLines: string[] = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (dataLines.length === 0) return;

  let data: unknown;
  try {
    data = JSON.parse(dataLines.join("\n"));
  } catch {
    return;
  }

  switch (event) {
    case "progress": {
      const d = data as { stage?: string; message?: string };
      handlers.onProgress?.(d.stage ?? "", d.message ?? "");
      break;
    }
    case "scored":
      handlers.onScored?.(data as ScoredEvent);
      break;
    case "done":
      handlers.onDone?.(data as DoneEvent);
      break;
    case "error":
      handlers.onError?.((data as { message?: string }).message ?? "unknown error");
      break;
  }
}
