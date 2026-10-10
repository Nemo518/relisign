import { useEffect, useRef } from "react";
import { useDemoStore } from "@/store/demo-store";
import { Switch } from "@/components/ui/switch";
import { useMediaRecorder } from "@/hooks/use-media-recorder";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";
import { Camera, Loader2, AlertCircle, RefreshCw } from "lucide-react";

interface Props {
  onCaptureComplete: (blob: Blob) => void;
}

export function CameraCapture({ onCaptureComplete }: Props) {
  const {
    cameraMode,
    cameraIndex,
    captureFps,
    captureSeconds,
    windowSeconds,
    stepSeconds,
    streamTotalSeconds,
    lightweight,
    ultraLightweight,
    setCameraMode,
    setCameraIndex,
    setCaptureFps,
    setCaptureSeconds,
    setWindowSeconds,
    setStepSeconds,
    setStreamTotalSeconds,
    setLightweight,
    setUltraLightweight,
  } = useDemoStore();

  const {
    isRecording,
    previewFrame,
    progress,
    cameraReady,
    cameraError,
    devices,
    startCamera,
    startRecording,
    stopStream,
  } = useMediaRecorder({ cameraIndex, fps: captureFps });

  const hasAutoStarted = useRef(false);

  // Auto-start camera preview when component mounts
  useEffect(() => {
    if (!hasAutoStarted.current) {
      hasAutoStarted.current = true;
      startCamera();
    }
    return () => { stopStream(); hasAutoStarted.current = false; };
  }, [cameraIndex]); // restart when camera index changes

  const profile = ultraLightweight ? "超轻量" : lightweight ? "轻量" : "标准";

  const handleCapture = async () => {
    const duration = cameraMode === "rolling" ? streamTotalSeconds : captureSeconds;
    const blob = await startRecording(duration);
    if (blob) onCaptureComplete(blob);
  };

  return (
    <div className="space-y-3">
      {/* Mode toggle */}
      <div className="flex gap-2">
        {(["single", "rolling"] as const).map((mode) => (
          <button
            key={mode}
            onClick={() => setCameraMode(mode)}
            className={cn(
              "flex-1 rounded-lg px-3 py-1.5 text-xs font-medium transition-all",
              cameraMode === mode
                ? "bg-gradient-to-r from-cyan-400 to-teal-400 text-slate-900 shadow-md"
                : "border border-slate-700/50 text-slate-400 hover:text-slate-200",
            )}
          >
            {mode === "single" ? "单段识别" : "滚动窗口"}
          </button>
        ))}
      </div>

      {/* Config row */}
      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="text-[10px] text-slate-400">摄像头</label>
          <select
            value={cameraIndex}
            onChange={(e) => setCameraIndex(Number(e.target.value))}
            className="w-full rounded-lg border border-slate-700/50 bg-slate-800/60 px-2 py-1.5 text-xs text-slate-200 focus:border-cyan-400/50 focus:outline-none"
          >
            {devices.length === 0 && <option value={0}>默认摄像头</option>}
            {devices.map((d, i) => (
              <option key={d.deviceId} value={i}>
                {d.label || `摄像头 ${i + 1}`}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="text-[10px] text-slate-400">采集帧率</label>
          <select
            value={captureFps}
            onChange={(e) => setCaptureFps(Number(e.target.value))}
            className="w-full rounded-lg border border-slate-700/50 bg-slate-800/60 px-2 py-1.5 text-xs text-slate-200 focus:border-cyan-400/50 focus:outline-none"
          >
            {[10, 15, 20, 25, 30].map((f) => (
              <option key={f} value={f}>{f} FPS</option>
            ))}
          </select>
        </div>
        <div>
          <label className="text-[10px] text-slate-400">
            {cameraMode === "rolling" ? "总时长(秒)" : "采集时长(秒)"}
          </label>
          <select
            value={cameraMode === "rolling" ? streamTotalSeconds : captureSeconds}
            onChange={(e) => {
              const v = Number(e.target.value);
              if (cameraMode === "rolling") setStreamTotalSeconds(v);
              else setCaptureSeconds(v);
            }}
            className="w-full rounded-lg border border-slate-700/50 bg-slate-800/60 px-2 py-1.5 text-xs text-slate-200 focus:border-cyan-400/50 focus:outline-none"
          >
            {[3, 5, 8, 10, 15, 20].map((s) => (
              <option key={s} value={s}>{s}s</option>
            ))}
          </select>
        </div>
        <div>
          <label className="text-[10px] text-slate-400">性能模式</label>
          <select
            value={ultraLightweight ? "ultra" : lightweight ? "light" : "std"}
            onChange={(e) => {
              if (e.target.value === "ultra") setUltraLightweight(true);
              else if (e.target.value === "light") setLightweight(true);
              else { setLightweight(false); setUltraLightweight(false); }
            }}
            className="w-full rounded-lg border border-slate-700/50 bg-slate-800/60 px-2 py-1.5 text-xs text-slate-200 focus:border-cyan-400/50 focus:outline-none"
          >
            <option value="std">标准</option>
            <option value="light">轻量</option>
            <option value="ultra">超轻量</option>
          </select>
        </div>
      </div>

      {/* Camera preview */}
      <div className="relative rounded-xl bg-slate-950 border border-slate-700/30 overflow-hidden aspect-[4/3]">
        {cameraError ? (
          <div className="flex items-center justify-center h-full p-4">
            <div className="text-center">
              <AlertCircle size={28} className="text-rose-400 mx-auto mb-2" />
              <p className="text-[11px] text-rose-400 leading-relaxed">{cameraError}</p>
              <button
                onClick={() => startCamera()}
                className="mt-3 inline-flex items-center gap-1.5 rounded-lg bg-slate-800 border border-slate-600 px-3 py-1.5 text-[11px] text-slate-300 hover:text-slate-100 transition-colors"
              >
                <RefreshCw size={12} /> 重试
              </button>
            </div>
          </div>
        ) : previewFrame ? (
          <img
            src={previewFrame}
            alt="摄像头实时预览"
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <Loader2 size={24} className="animate-spin text-slate-500 mx-auto mb-2" />
              <p className="text-[10px] text-slate-500">正在启动摄像头...</p>
            </div>
          </div>
        )}

        {/* Recording indicator */}
        {isRecording && (
          <div className="absolute top-2 right-2 flex items-center gap-1.5 rounded-full bg-rose-500/90 px-2.5 py-1">
            <span className="h-2 w-2 rounded-full bg-white animate-pulse" />
            <span className="text-[10px] font-bold text-white">REC</span>
          </div>
        )}

        {/* Camera ready indicator */}
        {cameraReady && !isRecording && (
          <div className="absolute top-2 right-2 flex items-center gap-1.5 rounded-full bg-emerald-500/80 px-2 py-0.5">
            <span className="h-1.5 w-1.5 rounded-full bg-white" />
            <span className="text-[9px] font-bold text-white">LIVE</span>
          </div>
        )}

        {/* Progress bar */}
        {isRecording && (
          <div className="absolute bottom-0 left-0 right-0 h-1 bg-slate-800">
            <div
              className="h-full bg-gradient-to-r from-cyan-500 to-teal-400 transition-all duration-100"
              style={{ width: `${progress}%` }}
            />
          </div>
        )}
      </div>

      {/* Recording info */}
      <div className="flex items-center justify-between text-[10px] text-slate-400">
        <span>
          {cameraReady ? "摄像头就绪" : "等待摄像头..."} · {captureFps}FPS · {profile}
        </span>
        {cameraMode === "rolling" && (
          <span>窗口{windowSeconds}s · 步长{stepSeconds}s</span>
        )}
      </div>

      {/* Action button */}
      <Button
        variant="primary"
        disabled={isRecording || !cameraReady}
        className="w-full"
        onClick={handleCapture}
      >
        {isRecording ? (
          <><Loader2 size={15} className="animate-spin" />{cameraMode === "rolling" ? "持续识别中..." : "采集中..."} {Math.round(progress)}%</>
        ) : (
          <><Camera size={15} />{cameraMode === "rolling" ? "开始实时翻译" : "开始采集并翻译"}</>
        )}
      </Button>
    </div>
  );
}
