import { create } from "zustand";
import type { VideoRecord } from "@/types/api";

interface DemoState {
  videoSource: "sample" | "upload" | "camera";
  cameraMode: "single" | "rolling";
  selectedSample: VideoRecord | null;
  uploadedFile: File | null;
  cameraClipBlob: Blob | null;
  realInference: boolean;
  rgbSupport: boolean;
  lightweight: boolean;
  ultraLightweight: boolean;
  checkpointPath: string;
  captureFps: number;
  captureSeconds: number;
  cameraIndex: number;
  windowSeconds: number;
  stepSeconds: number;
  streamTotalSeconds: number;
  lastPrediction: string | null;
  lastLatencyS: number | null;
  lastEffectiveFps: number | null;
  cameraPreviewFrame: string | null;

  setVideoSource: (s: "sample" | "upload" | "camera") => void;
  setCameraMode: (m: "single" | "rolling") => void;
  setSelectedSample: (s: VideoRecord | null) => void;
  setUploadedFile: (f: File | null) => void;
  setCameraClipBlob: (b: Blob | null) => void;
  setRealInference: (v: boolean) => void;
  setRgbSupport: (v: boolean) => void;
  setLightweight: (v: boolean) => void;
  setUltraLightweight: (v: boolean) => void;
  setCheckpointPath: (p: string) => void;
  setCaptureFps: (n: number) => void;
  setCaptureSeconds: (n: number) => void;
  setCameraIndex: (n: number) => void;
  setWindowSeconds: (n: number) => void;
  setStepSeconds: (n: number) => void;
  setStreamTotalSeconds: (n: number) => void;
  setPredictionResult: (
    prediction: string,
    latencyS: number,
    effectiveFps: number,
  ) => void;
  setCameraPreviewFrame: (b64: string | null) => void;
  reset: () => void;
}

const initialState = {
  videoSource: "sample" as const,
  cameraMode: "single" as const,
  selectedSample: null,
  uploadedFile: null,
  cameraClipBlob: null,
  realInference: false,
  rgbSupport: true,
  lightweight: false,
  ultraLightweight: false,
  checkpointPath: "out/stage3_finetuning/best_checkpoint.pth",
  captureFps: 20,
  captureSeconds: 5,
  cameraIndex: 0,
  windowSeconds: 3,
  stepSeconds: 2,
  streamTotalSeconds: 18,
  lastPrediction: null,
  lastLatencyS: null,
  lastEffectiveFps: null,
  cameraPreviewFrame: null,
};

export const useDemoStore = create<DemoState>((set) => ({
  ...initialState,
  setVideoSource: (videoSource) => set({ videoSource }),
  setCameraMode: (cameraMode) => set({ cameraMode }),
  setSelectedSample: (selectedSample) => set({ selectedSample }),
  setUploadedFile: (uploadedFile) => set({ uploadedFile }),
  setCameraClipBlob: (cameraClipBlob) => set({ cameraClipBlob }),
  setRealInference: (realInference) => set({ realInference }),
  setRgbSupport: (rgbSupport) => set({ rgbSupport }),
  setLightweight: (lightweight) =>
    set((s) => ({ lightweight, ultraLightweight: lightweight ? false : s.ultraLightweight })),
  setUltraLightweight: (ultraLightweight) =>
    set((s) => ({ ultraLightweight, lightweight: ultraLightweight ? true : s.lightweight })),
  setCheckpointPath: (checkpointPath) => set({ checkpointPath }),
  setCaptureFps: (captureFps) => set({ captureFps }),
  setCaptureSeconds: (captureSeconds) => set({ captureSeconds }),
  setCameraIndex: (cameraIndex) => set({ cameraIndex }),
  setWindowSeconds: (windowSeconds) => set({ windowSeconds }),
  setStepSeconds: (stepSeconds) => set({ stepSeconds }),
  setStreamTotalSeconds: (streamTotalSeconds) => set({ streamTotalSeconds }),
  setPredictionResult: (prediction, latencyS, effectiveFps) =>
    set({
      lastPrediction: prediction,
      lastLatencyS: latencyS,
      lastEffectiveFps: effectiveFps,
    }),
  setCameraPreviewFrame: (cameraPreviewFrame) => set({ cameraPreviewFrame }),
  reset: () => set(initialState),
}));
