import { notFound } from "next/navigation";

import { VideoWorkspace } from "@/components/video-workspace";
import { getVideo } from "@/lib/library";

export default async function VideoPage({
  params,
}: {
  // Next 16: route params arrive as a promise.
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;

  try {
    const detail = await getVideo(id);
    return <VideoWorkspace initial={detail} />;
  } catch {
    notFound();
  }
}
