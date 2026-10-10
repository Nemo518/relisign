import { useState, useRef, useCallback, useEffect } from "react";

interface UseMediaRecorderOptions {
  cameraIndex?: number;
  fps?: number;
}

export function useMediaRecorder(opts: UseMediaRecorderOptions = {}) {
  const { cameraIndex = 0, fps = 20 } = opts;
  const [isRecording, setIsRecording] = useState(false);
  const [previewFrame, setPreviewFrame] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [cameraReady, setCameraReady] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);

  const streamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const videoElRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const previewRafRef = useRef<number>(0);

  // Enumerate devices on mount
  useEffect(() => {
    navigator.mediaDevices
      .enumerateDevices()
      .then((devs) => setDevices(devs.filter((d) => d.kind === "videoinput")))
      .catch(() => {});
  }, []);

  const stopStream = useCallback(() => {
    cancelAnimationFrame(previewRafRef.current);
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setCameraReady(false);
    setPreviewFrame(null);
  }, []);

  const startCamera = useCallback(async () => {
    setCameraError(null);
    stopStream();
    try {
      const devs = await navigator.mediaDevices.enumerateDevices();
      const videoDevs = devs.filter((d) => d.kind === "videoinput");
      setDevices(videoDevs);

      const deviceId =
        videoDevs[cameraIndex]?.deviceId ?? videoDevs[0]?.deviceId;

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          deviceId: deviceId ? { exact: deviceId } : undefined,
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: fps },
        },
        audio: false,
      });
      streamRef.current = stream;

      // Create hidden video for frame capture
      const video = document.createElement("video");
      video.srcObject = stream;
      video.muted = true;
      video.playsInline = true;
      await video.play();
      videoElRef.current = video;
      canvasRef.current = document.createElement("canvas");

      // Continuous preview frame capture
      const capturePreview = () => {
        if (!canvasRef.current || !videoElRef.current) return;
        const c = canvasRef.current;
        const v = videoElRef.current;
        if (v.videoWidth === 0) { previewRafRef.current = requestAnimationFrame(capturePreview); return; }
        c.width = v.videoWidth;
        c.height = v.videoHeight;
        c.getContext("2d")?.drawImage(v, 0, 0, c.width, c.height);
        setPreviewFrame(c.toDataURL("image/jpeg", 0.75));
        previewRafRef.current = requestAnimationFrame(capturePreview);
      };
      previewRafRef.current = requestAnimationFrame(capturePreview);
      setCameraReady(true);
      return stream;
    } catch (err: any) {
      const msg = err?.name === "NotAllowedError"
        ? "摄像头权限被拒绝，请在浏览器设置中允许访问摄像头"
        : err?.name === "NotFoundError"
          ? "未检测到摄像头设备"
          : `摄像头启动失败: ${err?.message ?? "未知错误"}`;
      setCameraError(msg);
      return null;
    }
  }, [cameraIndex, fps, stopStream]);

  const startRecording = useCallback(
    async (durationSeconds: number): Promise<Blob | null> => {
      setProgress(0);
      chunksRef.current = [];

      const stream = streamRef.current?.active ? streamRef.current : await startCamera();
      if (!stream) return null;

      const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
        ? "video/webm;codecs=vp9"
        : "video/webm";

      const recorder = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      const startTime = performance.now();
      setIsRecording(true);

      const interval = setInterval(() => {
        const elapsed = performance.now() - startTime;
        setProgress(Math.min(100, Math.round((elapsed / (durationSeconds * 1000)) * 100)));
      }, 100);

      return new Promise((resolve) => {
        recorder.start(250);
        setTimeout(() => {
          clearInterval(interval);
          recorder.onstop = () => {
            const blob = new Blob(chunksRef.current, { type: mimeType });
            setIsRecording(false);
            setProgress(100);
            resolve(blob);
          };
          recorder.stop();
        }, durationSeconds * 1000);
      });
    },
    [startCamera],
  );

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cancelAnimationFrame(previewRafRef.current);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  return {
    isRecording,
    previewFrame,
    progress,
    cameraReady,
    cameraError,
    devices,
    startCamera,
    startRecording,
    stopStream,
  };
}
