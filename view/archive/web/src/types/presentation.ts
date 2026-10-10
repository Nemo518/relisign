export type JudgePersona = "综合评委" | "技术评委" | "应用评委";

export type StageLevel =
  | "Level 1 快速体验"
  | "Level 2 核心功能"
  | "Level 3 技术深度"
  | "Level 4 创新扩展";

export interface Section {
  key: string;
  label: string;
  seconds: number;
  accent: string;
}

export type SectionKey = "live" | "xai" | "compare" | "arch" | "scene";
