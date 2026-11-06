import { cookies } from "next/headers";

import { VideoWorkspace } from "@/components/video-workspace";
import type { VideoDetail } from "@/lib/library";
import { apiUrl } from "@/lib/library";

// How long the server waits for the API before handing the page to the client.
// Long enough for a healthy API on the same network, short enough that a dead
// one costs the viewer little before the client-side fallback takes over.
const SERVER_FETCH_TIMEOUT_MS = 1500;

/**
 * Fetch the video on the server when the API answers, so the first HTML already
 * holds the page. Rendered only in the browser, nothing readable painted until
 * the JavaScript had loaded and its own request had returned: 3.2 s to the
 * largest paint on a throttled phone (results/web-performance/).
 *
 * Any failure returns null and the client loads the page as before. That path
 * stays because the offline fallback reads a static manifest under
 * /cached-runs/, which a server-side fetch cannot reach; making the server
 * fetch mandatory would 404 the page exactly when the API is down, the one
 * moment the fallback exists for.
 */
async function initialDetail(id: string): Promise<VideoDetail | null> {
  // The API only answers for a signed-in code, so pass the visitor's session
  // cookie on. It reaches this server only when the frontend and the API share
  // a cookie domain (GIST_COOKIE_DOMAIN in deployment, plain localhost in
  // development); otherwise the API says 401 and the client loads the page.
  const session = (await cookies()).get("gist_session");
  try {
    const response = await fetch(apiUrl(`/v1/library/videos/${id}`), {
      cache: "no-store",
      headers: session ? { Cookie: `gist_session=${session.value}` } : undefined,
      signal: AbortSignal.timeout(SERVER_FETCH_TIMEOUT_MS),
    });
    return response.ok ? ((await response.json()) as VideoDetail) : null;
  } catch {
    return null;
  }
}

export default async function VideoPage({
  params,
}: {
  // Next 16: route params arrive as a promise.
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  return <VideoWorkspace videoId={id} initialDetail={await initialDetail(id)} />;
}
