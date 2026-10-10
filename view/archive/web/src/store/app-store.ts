import { create } from "zustand";
import type { JudgePersona, StageLevel, SectionKey } from "@/types/presentation";

interface AppState {
  stage: StageLevel;
  judge: JudgePersona;
  split: "test" | "dev";
  apiBase: string;
  darkMode: boolean;
  setStage: (stage: StageLevel) => void;
  setJudge: (judge: JudgePersona) => void;
  setSplit: (split: "test" | "dev") => void;
  setApiBase: (url: string) => void;
  toggleDarkMode: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  stage: "Level 3 技术深度",
  judge: "综合评委",
  split: "test",
  apiBase: "http://127.0.0.1:8000",
  darkMode: true,
  setStage: (stage) => set({ stage }),
  setJudge: (judge) => set({ judge }),
  setSplit: (split) => set({ split }),
  setApiBase: (apiBase) => set({ apiBase }),
  toggleDarkMode: () => set((s) => ({ darkMode: !s.darkMode })),
}));

interface PresentationState {
  activeSection: SectionKey;
  autoScript: boolean;
  autoSwitch: boolean;
  intervalS: number;
  currentScriptIndex: number;
  setSection: (section: SectionKey) => void;
  toggleAutoScript: () => void;
  toggleAutoSwitch: () => void;
  setInterval: (s: number) => void;
  advanceScript: () => void;
}

export const usePresentationStore = create<PresentationState>((set) => ({
  activeSection: "live",
  autoScript: true,
  autoSwitch: false,
  intervalS: 35,
  currentScriptIndex: 0,
  setSection: (section) => set({ activeSection: section, currentScriptIndex: 0 }),
  toggleAutoScript: () => set((s) => ({ autoScript: !s.autoScript })),
  toggleAutoSwitch: () => set((s) => ({ autoSwitch: !s.autoSwitch })),
  setInterval: (s) => set({ intervalS: s }),
  advanceScript: () =>
    set((s) => ({ currentScriptIndex: s.currentScriptIndex + 1 })),
}));
