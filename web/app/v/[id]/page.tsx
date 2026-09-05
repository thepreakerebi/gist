import { VideoWorkspace } from "@/components/video-workspace";

export default async function VideoPage({
  params,
}: {
  // Next 16: route params arrive as a promise.
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;

  // Loading happens in the client, not here. The offline fallback reads a
  // static manifest under /cached-runs/, and a relative fetch does not resolve
  // server-side — rendering this on the server 404'd the page precisely when
  // the API was down, which is the one moment the fallback exists for.
  return <VideoWorkspace videoId={id} />;
}
