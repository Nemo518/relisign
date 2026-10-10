import { useAppStore } from "@/store/app-store";
import { STAGE_OPTIONS, JUDGE_OPTIONS, SPLIT_OPTIONS } from "@/lib/constants";
import { cn } from "@/lib/utils";
import { Moon, Sun, Cpu } from "lucide-react";

const selectClass = cn(
  "w-full rounded-xl border border-slate-700/50 bg-slate-800/60 px-3 py-2 text-sm text-slate-200",
  "focus:outline-none focus:ring-2 focus:ring-cyan-400/30 focus:border-cyan-400/30",
  "transition-all duration-200",
);

export function SidebarControls() {
  const {
    stage,
    judge,
    split,
    apiBase,
    darkMode,
    setStage,
    setJudge,
    setSplit,
    setApiBase,
    toggleDarkMode,
  } = useAppStore();

  return (
    <aside className="relative z-20 w-64 flex-shrink-0 border-r border-slate-700/30 bg-slate-900/80 backdrop-blur-xl p-5 flex flex-col gap-5">
      {/* Logo */}
      <div className="flex items-center gap-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-cyan-400 to-teal-500 text-white shadow-lg shadow-cyan-500/20">
          <Cpu size={18} />
        </div>
        <div>
          <h2 className="text-lg font-extrabold text-white tracking-tight">
            ReliSign
          </h2>
          <p className="text-[10px] text-slate-300 font-mono tracking-wider">v2.0 DISPLAY ENGINE</p>
        </div>
      </div>

      <div className="space-y-4">
        <div>
          <label className="block text-[10px] font-semibold text-slate-300 uppercase tracking-wider mb-1.5">
            展示层级
          </label>
          <select
            value={stage}
            onChange={(e) => setStage(e.target.value as any)}
            className={selectClass}
          >
            {STAGE_OPTIONS.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-[10px] font-semibold text-slate-300 uppercase tracking-wider mb-1.5">
            评委视角
          </label>
          <select
            value={judge}
            onChange={(e) => setJudge(e.target.value as any)}
            className={selectClass}
          >
            {JUDGE_OPTIONS.map((j) => (
              <option key={j} value={j}>
                {j}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-[10px] font-semibold text-slate-300 uppercase tracking-wider mb-1.5">
            主数据集
          </label>
          <select
            value={split}
            onChange={(e) => setSplit(e.target.value as "test" | "dev")}
            className={selectClass}
          >
            {SPLIT_OPTIONS.map((s) => (
              <option key={s} value={s}>
                {s.toUpperCase()}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-[10px] font-semibold text-slate-300 uppercase tracking-wider mb-1.5">
            后端 API
          </label>
          <input
            type="text"
            value={apiBase}
            onChange={(e) => setApiBase(e.target.value)}
            className={cn(selectClass, "font-mono text-xs")}
          />
        </div>
      </div>

      {/* Theme toggle */}
      <button
        onClick={toggleDarkMode}
        className="mt-auto flex items-center gap-2 rounded-xl border border-slate-700/50 bg-slate-800/60 px-3 py-2.5 text-sm text-slate-300 hover:border-cyan-400/30 hover:text-cyan-300 transition-all duration-200"
      >
        {darkMode ? <Sun size={16} /> : <Moon size={16} />}
        {darkMode ? "浅色模式" : "深色模式"}
      </button>

      {/* Version info */}
      <div className="text-center">
        <div className="tech-badge text-[10px] text-slate-400 flex items-center justify-center gap-1.5">
          <div className="pulse-dot" />
          SYSTEM ONLINE
        </div>
      </div>
    </aside>
  );
}
