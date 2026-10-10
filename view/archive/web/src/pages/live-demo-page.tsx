import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import { useDemoStore } from "@/store/demo-store";
import { usePrediction } from "@/hooks/use-prediction";
import { useLivePose } from "@/hooks/use-live-pose";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { queryKeys } from "@/api/query-keys";
import { DEFAULT_CHECKPOINT } from "@/lib/constants";
import { cn } from "@/lib/utils";
import type { PoseSequence } from "@/types/api";

import { SampleSelector } from "@/components/video/sample-selector";
import { VideoPlayer } from "@/components/video/video-player";
import { PoseSkeletonViewer } from "@/components/pose/pose-skeleton-viewer";
import { LivePoseOverlay } from "@/components/pose/live-pose-overlay";

import { Button } from "@/components/ui/button";
import { Loader2, Camera, Video, AlertCircle, CheckCircle2, Play, RefreshCw, StopCircle, Eye, Activity } from "lucide-react";

export function LiveDemoPage() {
  const {
    videoSource,
    selectedSample,
    captureFps,
    captureSeconds,
    cameraIndex,
    lightweight,
    ultraLightweight,
    rgbSupport,
    checkpointPath,
    setVideoSource,
    setSelectedSample,
  } = useDemoStore();

  // ── Camera state ──
  const [camStatus, setCamStatus] = useState<"idle" | "starting" | "ready" | "error">("idle");
  const [camError, setCamError] = useState<string | null>(null);
  const [camDevices, setCamDevices] = useState<MediaDeviceInfo[]>([]);
  const [recording, setRecording] = useState(false);
  const [recProgress, setRecProgress] = useState(0);

  const streamRef = useRef<MediaStream | null>(null);
  const cameraVideoRef = useRef<HTMLVideoElement | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  // Live pose detection via MediaPipe
  const [cameraVideoEl, setCameraVideoEl] = useState<HTMLVideoElement | null>(null);
  const [overlaySize, setOverlaySize] = useState({ w: 640, h: 480 });
  const { landmarksRef, connections, status: poseStatus, error: poseError } = useLivePose({
    video: cameraVideoEl,
    enabled: camStatus === "ready",
  });

  // ── Main state ──
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stage, setStage] = useState<"idle" | "recording" | "uploading" | "inferring" | "done">("idle");
  const [predictionText, setPredictionText] = useState<string | null>(null);
  const [cameraPose, setCameraPose] = useState<PoseSequence | null>(null);

  // ── Skeleton auto-play ──
  const [skeletonFrame, setSkeletonFrame] = useState(0);
  const [skeletonPlaying, setSkeletonPlaying] = useState(false);
  const skeletonTimerRef = useRef<number>(0);

  useEffect(() => {
    if (!skeletonPlaying || !cameraPose?.frames?.length) return;
    const fps = cameraPose.fps || 25;
    const advance = () => {
      setSkeletonFrame((prev) => {
        const next = prev + 1;
        return next < cameraPose.frames.length ? next : 0;
      });
      skeletonTimerRef.current = window.setTimeout(advance, 1000 / fps);
    };
    skeletonTimerRef.current = window.setTimeout(advance, 1000 / fps);
    return () => clearTimeout(skeletonTimerRef.current);
  }, [skeletonPlaying, cameraPose]);

  const predictMutation = usePrediction();

  // ── Attach stream to video element AFTER React renders it ──
  useEffect(() => {
    const stream = streamRef.current;
    const video = cameraVideoRef.current;
    if (stream && video && camStatus === "ready") {
      video.srcObject = stream;
      video.play().catch((e) => console.warn("Camera video play failed:", e));
    }
  }, [camStatus, cameraVideoEl]);

  // ── Cleanup on unmount ──
  useEffect(() => {
    return () => {
      clearTimeout(skeletonTimerRef.current);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  // ── Stop camera stream ──
  const stopCamera = useCallback(() => {
    if (recorderRef.current && recorderRef.current.state === "recording") {
      recorderRef.current.stop();
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setCamStatus("idle");
    setRecording(false);
  }, []);

  // ── Start camera (triggered by user click) ──
  const startCamera = useCallback(async () => {
    stopCamera();
    setCamStatus("starting");
    setCamError(null);

    try {
      const allDevices = await navigator.mediaDevices.enumerateDevices();
      const videoDevs = allDevices.filter((d) => d.kind === "videoinput");
      setCamDevices(videoDevs);

      const deviceId = videoDevs[cameraIndex]?.deviceId ?? videoDevs[0]?.deviceId;

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          deviceId: deviceId ? { exact: deviceId } : undefined,
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: captureFps },
        },
        audio: false,
      });

      streamRef.current = stream;
      // Signal "ready" so React renders the <video> element,
      // then a useEffect will attach the stream to the DOM node.
      setCamStatus("ready");
    } catch (err: any) {
      const msg =
        err?.name === "NotAllowedError"
          ? "摄像头权限被拒绝，请在浏览器设置中允许访问摄像头，然后点击重试"
          : err?.name === "NotFoundError"
            ? "未检测到摄像头设备"
            : err?.name === "NotReadableError"
              ? "摄像头被其他应用占用，请关闭其他程序后重试"
              : `启动失败: ${err?.message ?? "未知错误"}`;
      setCamError(msg);
      setCamStatus("error");
    }
  }, [cameraIndex, captureFps, stopCamera]);

  // ── Auto-switch camera device when index changes ──
  const prevCameraIndex = useRef(cameraIndex);
  useEffect(() => {
    if (prevCameraIndex.current !== cameraIndex && camStatus === "ready") {
      prevCameraIndex.current = cameraIndex;
      startCamera();
    } else {
      prevCameraIndex.current = cameraIndex;
    }
  }, [cameraIndex]);

  // ── Record from camera stream ──
  const startRecording = useCallback((durationSeconds: number): Promise<Blob | null> => {
    return new Promise((resolve) => {
      const stream = streamRef.current;
      if (!stream?.active) {
        resolve(null);
        return;
      }

      chunksRef.current = [];
      const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
        ? "video/webm;codecs=vp9"
        : "video/webm";

      const recorder = new MediaRecorder(stream, { mimeType });
      recorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      setRecording(true);
      setRecProgress(0);
      const startTime = performance.now();

      const interval = setInterval(() => {
        setRecProgress(Math.min(100, Math.round(((performance.now() - startTime) / (durationSeconds * 1000)) * 100)));
      }, 100);

      recorder.start(250);

      setTimeout(() => {
        clearInterval(interval);
        recorder.onstop = () => {
          const blob = new Blob(chunksRef.current, { type: mimeType });
          setRecording(false);
          setRecProgress(100);
          resolve(blob.size > 0 ? blob : null);
        };
        if (recorder.state === "recording") {
          recorder.stop();
        } else {
          setRecording(false);
          resolve(null);
        }
      }, durationSeconds * 1000);
    });
  }, []);

  // ── Full translate flow ──
  const handleTranslate = async () => {
    setRunning(true);
    setError(null);
    setStage("recording");
    setCameraPose(null);
    setSkeletonPlaying(false);

    const blob = await startRecording(captureSeconds);
    if (!blob) {
      setError("录制失败，未获取到视频数据");
      setRunning(false);
      setStage("idle");
      return;
    }

    setStage("uploading");
    try {
      const file = new File([blob], "camera.webm", { type: blob.type || "video/webm" });
      const up = await api.uploadVideo(file);

      setStage("inferring");
      const result = await predictMutation.mutateAsync({
        video_path: up.uploaded_path,
        checkpoint_path: checkpointPath || DEFAULT_CHECKPOINT,
        rgb_support: rgbSupport,
        lightweight,
        ultra_lightweight: ultraLightweight,
      });

      if (result.success) {
        setPredictionText(result.prediction);
        if (result.pose_sequence) {
          setCameraPose(result.pose_sequence);
          setSkeletonFrame(0);
          setSkeletonPlaying(true);
        }
        setStage("done");
      } else {
        setError(result.stderr || "推理失败");
        setStage("idle");
      }
    } catch (err: any) {
      setError(err?.message ?? "翻译失败");
      setStage("idle");
    } finally {
      setRunning(false);
    }
  };

  // ── Sample video data ──
  const cachedQuery = useQuery({
    queryKey: queryKeys.cachedPrediction(selectedSample?.sample_id ?? ""),
    queryFn: () => api.cachedPrediction(selectedSample!.sample_id),
    enabled: !!selectedSample?.sample_id && videoSource === "sample",
    staleTime: 60_000,
  });

  const metaQuery = useQuery({
    queryKey: queryKeys.videoMetadata(selectedSample?.relative_path ?? ""),
    queryFn: () => api.videoMetadata(selectedSample!.relative_path),
    enabled: !!selectedSample?.relative_path && videoSource === "sample",
  });

  const poseQuery = useQuery({
    queryKey: queryKeys.poseSequence(selectedSample?.relative_path ?? ""),
    queryFn: () => api.poseSequence(selectedSample!.relative_path),
    enabled: !!selectedSample?.relative_path && videoSource === "sample",
    staleTime: 5 * 60 * 1000,
  });

  const samplePrediction = useMemo(() => {
    if (videoSource !== "sample") return null;
    if (cachedQuery.data?.prediction) return cachedQuery.data.prediction;
    if (selectedSample?.reference) return selectedSample.reference;
    return null;
  }, [videoSource, cachedQuery.data, selectedSample]);

  const displayText = predictionText ?? samplePrediction;
  const meta = metaQuery.data;

  // ── Sample video → skeleton sync ──
  const sampleVideoRef = useRef<HTMLVideoElement>(null);
  const [videoFrameIndex, setVideoFrameIndex] = useState(0);
  const [videoIsPlaying, setVideoIsPlaying] = useState(false);

  useEffect(() => {
    const v = sampleVideoRef.current;
    if (!v || videoSource !== "sample") return;

    const syncFrame = () => {
      if (v.readyState >= 2 && meta?.fps) {
        setVideoFrameIndex(Math.floor(v.currentTime * meta.fps));
      }
    };
    const onPlay = () => setVideoIsPlaying(true);
    const onPause = () => setVideoIsPlaying(false);
    const onEnded = () => setVideoIsPlaying(false);

    // Use timeupdate (fires ~4-15fps) instead of rAF (60fps) — avoids flooding React
    v.addEventListener("timeupdate", syncFrame);
    v.addEventListener("play", onPlay);
    v.addEventListener("pause", onPause);
    v.addEventListener("ended", onEnded);
    v.addEventListener("seeked", syncFrame);
    v.addEventListener("loadedmetadata", syncFrame);

    return () => {
      v.removeEventListener("timeupdate", syncFrame);
      v.removeEventListener("play", onPlay);
      v.removeEventListener("pause", onPause);
      v.removeEventListener("ended", onEnded);
      v.removeEventListener("seeked", syncFrame);
      v.removeEventListener("loadedmetadata", syncFrame);
    };
  }, [videoSource, meta?.fps]);

  // ── Switch source → reset ──
  const switchSource = (src: "sample" | "camera") => {
    setVideoSource(src);
    setError(null);
    setStage("idle");
    setPredictionText(null);
    setCameraPose(null);
    setSkeletonPlaying(false);
    if (src === "sample") stopCamera();
  };

  const profile = ultraLightweight ? "超轻量" : lightweight ? "轻量" : "标准";

  return (
    <div className="space-y-5">
      {/* ====== HEADER ====== */}
      <div className="flex items-center gap-3">
        <h2 className="text-xl font-bold">实时翻译演示</h2>
        <span className="tech-badge rounded-md bg-cyan-400/10 border border-cyan-400/20 px-2.5 py-1 text-[10px] text-cyan-400">
          LIVE TRANSLATION
        </span>
        {displayText && (
          <span className="tech-badge rounded-md bg-emerald-400/10 border border-emerald-400/20 px-2 py-0.5 text-[10px] text-emerald-400 font-mono">
            {predictionText ? "MODEL" : "CACHED"}
          </span>
        )}
        <span className="ml-auto tech-badge rounded-md bg-cyan-400/10 border border-cyan-400/20 px-2 py-0.5 text-[10px] text-cyan-400">
          {profile} · {captureFps}FPS
        </span>
      </div>

      {/* ====== MAIN ====== */}
      <div className="grid grid-cols-[1fr_1fr] gap-5">
        {/* ── LEFT COLUMN ── */}
        <div className="space-y-4">
          {/* Source tabs */}
          <div className="glass-card rounded-2xl p-4">
            <div className="flex gap-2 mb-3">
              {(["sample", "camera"] as const).map((src) => (
                <button
                  key={src}
                  onClick={() => switchSource(src)}
                  className={cn(
                    "flex-1 flex items-center justify-center gap-1.5 rounded-lg px-3 py-2 text-xs font-medium transition-all",
                    videoSource === src
                      ? "bg-gradient-to-r from-cyan-400 to-teal-400 text-slate-900 shadow-md"
                      : "border border-slate-700/50 text-slate-400 hover:text-slate-200",
                  )}
                >
                  {src === "sample" ? <><Video size={13} />样例视频</> : <><Camera size={13} />摄像头实时</>}
                </button>
              ))}
            </div>

            {/* ── SAMPLE MODE ── */}
            {videoSource === "sample" && (
              <>
                <SampleSelector value={selectedSample} onChange={setSelectedSample} />
                {selectedSample && (
                  <div className="mt-2 rounded-xl bg-slate-800/40 border border-slate-700/30 p-2.5">
                    <span className="text-slate-400 text-[10px]">参考：</span>
                    <span className="text-slate-200 text-[11px] ml-1">{selectedSample.reference || "暂无"}</span>
                  </div>
                )}
              </>
            )}

            {/* ── CAMERA MODE ── */}
            {videoSource === "camera" && (
              <div className="space-y-3">
                {/* Idle: show start button */}
                {camStatus === "idle" && (
                  <div className="rounded-xl bg-slate-800/40 border border-slate-700/30 p-4 text-center">
                    <Camera size={28} className="text-slate-500 mx-auto mb-2" />
                    <p className="text-[11px] text-slate-400 mb-3">点击下方按钮开启摄像头</p>
                    <Button variant="primary" size="sm" onClick={startCamera}>
                      <Play size={14} /> 打开摄像头
                    </Button>
                  </div>
                )}

                {/* Starting */}
                {camStatus === "starting" && (
                  <div className="rounded-xl bg-slate-800/40 border border-slate-700/30 p-6 text-center">
                    <Loader2 size={28} className="animate-spin text-cyan-400 mx-auto mb-2" />
                    <p className="text-[11px] text-slate-400">正在启动摄像头...</p>
                    <p className="text-[9px] text-slate-500 mt-1">请在浏览器权限弹窗中点击"允许"</p>
                  </div>
                )}

                {/* Error */}
                {camStatus === "error" && (
                  <div className="rounded-xl bg-rose-400/5 border border-rose-400/20 p-4 text-center">
                    <AlertCircle size={28} className="text-rose-400 mx-auto mb-2" />
                    <p className="text-[11px] text-rose-400 leading-relaxed mb-3">{camError}</p>
                    <button
                      onClick={startCamera}
                      className="inline-flex items-center gap-1.5 rounded-lg bg-slate-800 border border-slate-600 px-3 py-1.5 text-[11px] text-slate-300 hover:text-slate-100"
                    >
                      <RefreshCw size={12} /> 重试
                    </button>
                  </div>
                )}

                {/* Device selector */}
                {camStatus === "ready" && camDevices.length > 1 && (
                  <select
                    value={cameraIndex}
                    onChange={(e) => useDemoStore.getState().setCameraIndex(Number(e.target.value))}
                    className="w-full rounded-lg border border-slate-700/50 bg-slate-800/60 px-2 py-1.5 text-xs text-slate-200 focus:border-cyan-400/50 focus:outline-none"
                  >
                    {camDevices.map((d, i) => (
                      <option key={d.deviceId} value={i}>{d.label || `摄像头 ${i + 1}`}</option>
                    ))}
                  </select>
                )}

                {/* Camera preview — always mounted so ref is available */}
                <div className={cn(
                  "relative rounded-xl bg-slate-950 border border-cyan-400/20 overflow-hidden aspect-[4/3]",
                  camStatus !== "ready" && "hidden",
                )}>
                  <video
                    ref={(el) => {
                      cameraVideoRef.current = el;
                      if (el !== cameraVideoEl) setCameraVideoEl(el);
                      if (el && (el.videoWidth !== overlaySize.w || el.videoHeight !== overlaySize.h)) {
                        if (el.videoWidth > 0) setOverlaySize({ w: el.videoWidth, h: el.videoHeight });
                      }
                    }}
                    onLoadedMetadata={(e) => {
                      const v = e.currentTarget;
                      if (v.videoWidth > 0) setOverlaySize({ w: v.videoWidth, h: v.videoHeight });
                    }}
                    muted
                    playsInline
                    className="w-full h-full object-cover"
                  />
                  <LivePoseOverlay
                    landmarksRef={landmarksRef}
                    connections={connections}
                    width={overlaySize.w}
                    height={overlaySize.h}
                  />
                  {recording && (
                      <div className="absolute top-2 right-2 flex items-center gap-1.5 rounded-full bg-rose-500/90 px-2.5 py-1 z-10">
                        <span className="h-2 w-2 rounded-full bg-white animate-pulse" />
                        <span className="text-[10px] font-bold text-white">REC</span>
                      </div>
                    )}
                    {!recording && (
                      <div className="absolute top-2 right-2 flex items-center gap-1.5 rounded-full bg-emerald-500/80 px-2 py-0.5 z-10">
                        <span className="h-1.5 w-1.5 rounded-full bg-white" />
                        <span className="text-[9px] font-bold text-white">LIVE</span>
                      </div>
                    )}
                    {recording && (
                      <div className="absolute bottom-0 left-0 right-0 h-1 bg-slate-800 z-10">
                        <div
                          className="h-full bg-gradient-to-r from-cyan-500 to-teal-400 transition-all duration-100"
                          style={{ width: `${recProgress}%` }}
                        />
                      </div>
                    )}
                </div>

                {camStatus === "ready" && (
                  <div className="space-y-1">
                    <div className="flex items-center justify-between text-[10px] text-slate-400">
                      <span className="flex items-center gap-2">
                        摄像头就绪 · {captureFps}FPS
                        {poseStatus === "loading" && (
                          <span className="flex items-center gap-1 text-amber-400">
                            <Loader2 size={10} className="animate-spin" /> 姿态模型加载中
                          </span>
                        )}
                        {poseStatus === "ready" && (
                          <span className="flex items-center gap-1 text-emerald-400">
                            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
                            姿态检测就绪
                          </span>
                        )}
                        {poseStatus === "error" && (
                          <span className="text-rose-400">姿态模型失败</span>
                        )}
                      </span>
                      <button
                        onClick={stopCamera}
                        className="flex items-center gap-1 text-slate-500 hover:text-slate-300 transition-colors"
                      >
                        <StopCircle size={12} /> 关闭摄像头
                      </button>
                    </div>
                    {poseError && (
                      <p className="text-[10px] text-rose-400/80 leading-relaxed">{poseError}</p>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Sample video player */}
          {videoSource === "sample" && <VideoPlayer ref={sampleVideoRef} />}

          {/* Error */}
          {error && (
            <div className="glass-card rounded-xl p-3 border border-rose-400/30 bg-rose-400/5">
              <div className="flex items-start gap-2">
                <AlertCircle size={16} className="text-rose-400 mt-0.5 shrink-0" />
                <p className="text-[11px] text-rose-400">{error}</p>
              </div>
            </div>
          )}

          {/* Stage indicator */}
          {stage !== "idle" && (
            <div className="glass-card rounded-xl p-3 border border-cyan-400/20">
              <div className="flex items-center gap-2">
                {stage !== "done" && <Loader2 size={14} className="animate-spin text-cyan-400" />}
                <span className="text-[11px] text-slate-300">
                  {stage === "recording" && `录制中... ${recProgress}%`}
                  {stage === "uploading" && "上传视频到服务器..."}
                  {stage === "inferring" && "模型推理中... (首次约10-30秒)"}
                  {stage === "done" && (
                    <span className="flex items-center gap-1 text-emerald-400">
                      <CheckCircle2 size={13} /> 翻译完成！
                    </span>
                  )}
                </span>
              </div>
            </div>
          )}

          {/* Translate button */}
          {videoSource === "camera" && camStatus === "ready" && (
            <Button
              variant="primary"
              className="w-full"
              disabled={running}
              onClick={handleTranslate}
            >
              {running ? (
                <><Loader2 size={15} className="animate-spin" />翻译中...</>
              ) : (
                <><Camera size={15} />开始实时翻译</>
              )}
            </Button>
          )}

          {videoSource === "sample" && !samplePrediction && (
            <p className="text-[10px] text-amber-400/80 text-center">选择样例视频查看翻译结果</p>
          )}
        </div>

        {/* ── RIGHT COLUMN ── */}
        <div className="space-y-4">
          {/* Translation result */}
          {displayText ? (
            <div className="glass-card rounded-2xl p-4 border border-cyan-400/20 bg-gradient-to-br from-cyan-500/5 to-teal-500/5">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <CheckCircle2 size={16} className="text-emerald-400" />
                  <span className="text-sm font-bold text-slate-200">翻译结果</span>
                </div>
                <span className={cn(
                  "text-[9px] font-mono px-1.5 py-0.5 rounded",
                  predictionText ? "bg-emerald-400/10 text-emerald-400" : "bg-amber-400/10 text-amber-400",
                )}>
                  {predictionText ? "实时推理" : "缓存匹配"}
                </span>
              </div>
              <div className="rounded-xl bg-slate-900/60 border border-slate-700/30 p-4 mb-3">
                <p className="text-xl font-bold text-cyan-400 tracking-wide">{displayText}</p>
              </div>
              {selectedSample?.reference && (
                <div className="flex items-center gap-2 text-[10px]">
                  <span className="text-slate-400">参考文本：</span>
                  <span className="text-slate-300">{selectedSample.reference}</span>
                </div>
              )}
            </div>
          ) : (
            <div className="glass-card rounded-2xl p-6 flex items-center justify-center border border-slate-700/20 min-h-[120px]">
              <div className="text-center">
                <Video size={24} className="text-slate-500 mx-auto mb-2" />
                <p className="text-[11px] text-slate-400">选择样例视频或使用摄像头开始翻译</p>
                <p className="text-[9px] text-slate-500 mt-1">133关键点 → 双流编码 → mT5解码</p>
              </div>
            </div>
          )}

          {/* Pose skeleton */}
          {videoSource === "sample" && (
            <PoseSkeletonViewer
              frameIndex={videoFrameIndex}
              isPlaying={videoIsPlaying}
              poseSequence={poseQuery.data ?? null}
              isLoading={poseQuery.isLoading}
            />
          )}
          {/* Camera: prompt to translate for 133-KP full-body analysis */}
          {videoSource === "camera" && camStatus === "ready" && !cameraPose && (
            <div className="glass-card rounded-2xl p-5 border border-purple-400/20 bg-gradient-to-br from-purple-500/5 to-pink-500/5">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-gradient-to-br from-purple-500 to-pink-500 text-white shadow">
                    <Eye size={14} />
                  </div>
                  <span className="text-sm font-bold">姿态提取可视化</span>
                </div>
                <span className="rounded bg-purple-400/10 px-2 py-0.5 text-[10px] text-purple-400 font-mono">
                  RTMPose 133KP
                </span>
              </div>
              <div className="rounded-xl bg-slate-950/80 border border-slate-700/30 aspect-[4/3] flex items-center justify-center">
                <div className="text-center px-6">
                  <Activity size={32} className="text-slate-500 mx-auto mb-3" />
                  <p className="text-sm text-slate-300 font-medium mb-1">点击下方按钮录制并翻译</p>
                  <p className="text-[11px] text-slate-400 leading-relaxed">
                    系统将调用 RTMPose Wholebody 提取 <span className="text-cyan-400 font-bold">133个关键点</span>，
                    包含身体(17) + 足部(6) + 面部(68) + 双手(2×21)，
                    并生成与样例视频完全一致的全身姿态可视化
                  </p>
                  <div className="mt-3 flex items-center justify-center gap-4 text-[10px] text-slate-500">
                    <span className="flex items-center gap-1"><span className="h-1.5 w-1.5 rounded-full bg-cyan-400" />身体</span>
                    <span className="flex items-center gap-1"><span className="h-1.5 w-1.5 rounded-full bg-purple-400" />左手</span>
                    <span className="flex items-center gap-1"><span className="h-1.5 w-1.5 rounded-full bg-pink-400" />右手</span>
                    <span className="flex items-center gap-1"><span className="h-1.5 w-1.5 rounded-full bg-amber-400" />面部</span>
                  </div>
                </div>
              </div>
            </div>
          )}
          {videoSource === "camera" && cameraPose && (
            <PoseSkeletonViewer
              frameIndex={skeletonFrame}
              isPlaying={skeletonPlaying}
              poseSequence={cameraPose}
              isLoading={false}
            />
          )}
        </div>
      </div>
    </div>
  );
}
