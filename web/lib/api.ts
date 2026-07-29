// Streaming client for the Gist demo API. The endpoint POSTs a JSON body and
// replies with Server-Sent Events, so native EventSource (GET-only) can't be
// used; we read the fetch body stream and parse SSE frames by hand.

import type {
  DoneEvent,
  RunRequest,
  ScoredEvent,
  StreamHandlers,
} from "@/lib/types";

export type { StreamHandlers } from "@/lib/types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") ?? "http://localhost:8000";

// Resolves to `true` if the live API was reached (whether it streamed a result
// or returned an HTTP error, which is surfaced via onError), or `false` only if
// the API was truly unreachable (network error). The caller uses `false` to
// trigger the cached-run fallback. Stream-level errors go through onError too.
export async function runDemo(
  request: RunRequest,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<boolean> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/v1/demo/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
      signal,
    });
  } catch {
    return false; // network-level failure: fall back to a cached run
  }

  // The API responded but with an error status (e.g. 422 bad request, 500).
  // This is a real server error, not an unreachable API — surface it directly
  // instead of the misleading cached-fallback path.
  if (!response.ok || !response.body) {
    handlers.onError?.(await describeHttpError(response));
    return true;
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
  return true;
}

// Turn a non-OK response into a readable message. FastAPI validation errors come
// back as {detail: [{loc, msg}, ...]}; other errors as {detail: "..."} or text.
async function describeHttpError(response: Response): Promise<string> {
  let detail = "";
  try {
    const body = await response.json();
    if (typeof body?.detail === "string") {
      detail = body.detail;
    } else if (Array.isArray(body?.detail)) {
      detail = body.detail
        .map((e: { loc?: unknown[]; msg?: string }) =>
          [Array.isArray(e.loc) ? e.loc.join(".") : "", e.msg].filter(Boolean).join(": "),
        )
        .join("; ");
    }
  } catch {
    // non-JSON body; fall through to the status line
  }
  return `API returned ${response.status}${detail ? `: ${detail}` : ""}`;
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
