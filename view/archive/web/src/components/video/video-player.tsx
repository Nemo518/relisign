import { forwardRef, useMemo, useRef } from "react";
import { useDemoStore } from "@/store/demo-store";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { queryKeys } from "@/api/query-keys";
import { cn } from "@/lib/utils";

interface Props {
  className?: string;
}

export const VideoPlayer = forwardRef<HTMLVideoElement, Props>(
  function VideoPlayer({ className }, ref) {
    // Select only what this component actually needs
    const videoSource = useDemoStore((s) => s.videoSource);
    const selectedSample = useDemoStore((s) => s.selectedSample);
    const cameraClipBlob = useDemoStore((s) => s.cameraClipBlob);

    const videoQuery = useQuery({
      queryKey: queryKeys.videoMetadata(selectedSample?.relative_path ?? ""),
      queryFn: () => api.videoMetadata(selectedSample!.relative_path),
      enabled: videoSource === "sample" && !!selectedSample?.relative_path,
    });

    // Stable blob URL — only recreate when blob changes
    const blobUrlRef = useRef<string | null>(null);
    const lastBlobRef = useRef<Blob | null>(null);

    const src = useMemo(() => {
      if (videoSource === "camera" && cameraClipBlob) {
        if (cameraClipBlob !== lastBlobRef.current) {
          if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current);
          blobUrlRef.current = URL.createObjectURL(cameraClipBlob);
          lastBlobRef.current = cameraClipBlob;
        }
        return blobUrlRef.current;
      }
      if (videoSource === "sample" && selectedSample) {
        return api.mediaUrl(selectedSample.relative_path);
      }
      return undefined;
    }, [videoSource, selectedSample, cameraClipBlob]);

    const meta = videoQuery.data;

    if (!src) return null;

    return (
      <div className={cn("space-y-2", className)}>
        <video
          ref={ref}
          src={src}
          controls
          preload="auto"
          className="w-full rounded-2xl shadow-lg"
          style={{ maxHeight: 360 }}
        />
        {meta && (
          <div className="text-xs text-muted-foreground">
            {meta.width}x{meta.height} · {meta.fps.toFixed(1)} FPS ·{" "}
            {meta.frame_count} 帧 · {meta.duration_s.toFixed(1)}s
          </div>
        )}
      </div>
    );
  },
);
