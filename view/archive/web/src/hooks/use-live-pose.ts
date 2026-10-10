import { useEffect, useRef, useState, useCallback } from "react";
import {
  PoseLandmarker,
  FilesetResolver,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision";

let poseLandmarker: PoseLandmarker | null = null;
let initPromise: Promise<PoseLandmarker> | null = null;

async function ensureLandmarker(): Promise<PoseLandmarker> {
  if (poseLandmarker) return poseLandmarker;
  if (!initPromise) {
    initPromise = (async () => {
      // Try local WASM first (more reliable, no CDN dependency)
      let vision: any;
      try {
        vision = await FilesetResolver.forVisionTasks("/mediapipe-wasm");
      } catch {
        console.warn("Local WASM failed, trying CDN...");
        vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm",
        );
      }
      return PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: "/pose_landmarker_lite.task",
          delegate: "CPU",
        },
        runningMode: "VIDEO",
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
    })();
  }
  const lm = await initPromise;
  poseLandmarker = lm;
  return lm;
}

export interface UseLivePoseOptions {
  video: HTMLVideoElement | null;
  enabled: boolean;
}

export function useLivePose({ video, enabled }: UseLivePoseOptions) {
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [error, setError] = useState<string | null>(null);

  // Use a ref for landmarks so consumers can read without triggering re-renders
  const landmarksRef = useRef<NormalizedLandmark[] | null>(null);
  // Minimal state update for React bindings — throttled
  const [landmarks, setLandmarks] = useState<NormalizedLandmark[] | null>(null);

  const rafRef = useRef<number>(0);
  const lastTimeRef = useRef(0);
  const lastStateUpdateRef = useRef(0);
  const lmRef = useRef<PoseLandmarker | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  videoRef.current = video;
  const statusRef = useRef(status);
  statusRef.current = status;

  // Initialize MediaPipe
  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    setStatus("loading");
    setError(null);
    ensureLandmarker()
      .then((lm) => {
        if (cancelled) return;
        lmRef.current = lm;
        setStatus("ready");
      })
      .catch((err) => {
        if (cancelled) return;
        const msg = err?.message ?? String(err ?? "未知错误");
        console.error("MediaPipe init error:", err);
        setError(msg.includes("NetworkError") || msg.includes("Failed to fetch")
          ? "姿态模型网络加载失败，请检查网络连接（CDN: jsdelivr.net）"
          : `姿态模型加载失败: ${msg}`);
        setStatus("error");
      });
    return () => {
      cancelled = true;
      lmRef.current = null;
    };
  }, [enabled]);

  // Process frames — throttled to ~15 fps for lower CPU
  useEffect(() => {
    if (!enabled || status !== "ready") return;

    // Stabilize video reference — don't restart loop if video object identity changes
    const v = videoRef.current;
    if (!v) return;

    let running = true;
    const DETECT_INTERVAL = 66; // ~15 fps, reduce CPU load
    const STATE_UPDATE_INTERVAL = 100; // Only update React state every 100ms

    const processFrame = () => {
      if (!running) return;

      if (v.readyState < 2 || v.videoWidth === 0) {
        rafRef.current = requestAnimationFrame(processFrame);
        return;
      }

      const now = performance.now();
      if (now - lastTimeRef.current < DETECT_INTERVAL) {
        rafRef.current = requestAnimationFrame(processFrame);
        return;
      }
      lastTimeRef.current = now;

      const lm = lmRef.current;
      if (!lm || statusRef.current !== "ready") {
        rafRef.current = requestAnimationFrame(processFrame);
        return;
      }

      try {
        const result = lm.detectForVideo(v, now);
        if (result.landmarks?.length) {
          const newLandmarks = result.landmarks[0];
          landmarksRef.current = newLandmarks;

          // Throttle React state updates to avoid excessive re-renders
          if (now - lastStateUpdateRef.current >= STATE_UPDATE_INTERVAL) {
            lastStateUpdateRef.current = now;
            setLandmarks(newLandmarks);
          }
        }
      } catch {
        // skip bad frames
      }

      rafRef.current = requestAnimationFrame(processFrame);
    };

    // Reset timing refs when loop restarts
    lastTimeRef.current = 0;
    lastStateUpdateRef.current = 0;
    rafRef.current = requestAnimationFrame(processFrame);

    return () => {
      running = false;
      cancelAnimationFrame(rafRef.current);
    };
  }, [enabled, status, video]);

  const connections = PoseLandmarker.POSE_CONNECTIONS;

  // Expose ref for high-frequency reading (canvas overlay)
  return { status, error, landmarks, landmarksRef, connections };
}
