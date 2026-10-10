import { useEffect, useRef, useState } from "react";
import { useParams, useSearchParams, useNavigate } from "react-router-dom";
import { useAppStore, usePresentationStore } from "@/store/app-store";
import {
  SECTION_ORDER,
  SECTION_LOOKUP,
  DECK_SCRIPTS,
  QUOTE_LINES,
} from "@/lib/constants";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { cn } from "@/lib/utils";
import { Download, ChevronLeft, ChevronRight } from "lucide-react";

export function PresentationPage() {
  const { section } = useParams<{ section: string }>();
  const navigate = useNavigate();
  const judge = useAppStore((s) => s.judge);
  const stage = useAppStore((s) => s.stage);
  const split = useAppStore((s) => s.split);

  const {
    activeSection,
    autoScript,
    autoSwitch,
    intervalS,
    currentScriptIndex,
    setSection,
    toggleAutoScript,
    toggleAutoSwitch,
    setInterval: setIntervalS,
    advanceScript,
  } = usePresentationStore();

  const [scriptLine, setScriptLine] = useState("");
  const [fading, setFading] = useState(false);
  const scriptTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const switchTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const currentSection = section ?? activeSection;
  const sectionInfo = SECTION_LOOKUP[currentSection];
  const currentIndex = SECTION_ORDER.findIndex((s) => s.key === currentSection);

  const scriptLinesForSection =
    DECK_SCRIPTS[judge]?.[currentSection] ?? QUOTE_LINES;

  // Sync active section via store
  useEffect(() => {
    if (section) setSection(section as any);
  }, [section, setSection]);

  // Script carousel
  useEffect(() => {
    const showLine = () => {
      const line =
        scriptLinesForSection[
          currentScriptIndex % scriptLinesForSection.length
        ] ?? "";
      setFading(true);
      setTimeout(() => {
        setScriptLine(line);
        setFading(false);
      }, 150);
      advanceScript();
    };

    showLine();

    if (autoScript && scriptLinesForSection.length > 1) {
      scriptTimerRef.current = setInterval(showLine, 3600);
    }

    return () => {
      if (scriptTimerRef.current) clearInterval(scriptTimerRef.current);
    };
  }, [currentSection, autoScript, judge]);

  // Auto switch sections
  useEffect(() => {
    if (autoSwitch) {
      switchTimerRef.current = setInterval(() => {
        const nextIdx = (currentIndex + 1) % SECTION_ORDER.length;
        const nextKey = SECTION_ORDER[nextIdx].key;
        navigate(`/present/${nextKey}`);
      }, intervalS * 1000);
    }
    return () => {
      if (switchTimerRef.current) clearInterval(switchTimerRef.current);
    };
  }, [autoSwitch, intervalS, currentIndex, navigate]);

  const goSection = (key: string) => {
    navigate(`/present/${key}`);
  };

  const goNext = () => {
    const nextIdx = (currentIndex + 1) % SECTION_ORDER.length;
    navigate(`/present/${SECTION_ORDER[nextIdx].key}`);
  };

  const goPrev = () => {
    const prevIdx =
      (currentIndex - 1 + SECTION_ORDER.length) % SECTION_ORDER.length;
    navigate(`/present/${SECTION_ORDER[prevIdx].key}`);
  };

  return (
    <div className="space-y-5">
      {/* Top bar: navigation pills + PDF */}
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          {SECTION_ORDER.map((s) => (
            <button
              key={s.key}
              onClick={() => goSection(s.key)}
              className={cn(
                "rounded-xl px-4 py-2 text-sm font-semibold transition-all duration-200",
                s.key === currentSection
                  ? "bg-gradient-to-r from-cyan-400 to-teal-400 text-slate-900 shadow-md"
                  : "border border-border bg-white text-foreground hover:bg-muted dark:bg-slate-800",
              )}
            >
              {s.label}
            </button>
          ))}
        </div>
        <Button variant="outline" size="sm">
          <Download size={14} />
          导出评委演示 PDF
        </Button>
      </div>

      {/* Deck Console */}
      <div className="hero-gradient rounded-3xl p-6 text-white shadow-xl">
        <div className="flex justify-between gap-4">
          <div className="flex-1">
            <div className="text-lg font-extrabold mb-2">展示节奏</div>
            <div className="text-base leading-relaxed text-white/90">
              当前页签：{sectionInfo?.label} · 评委视角：{judge} · 展示层级：{stage}
            </div>
            <div
              className={cn(
                "mt-4 rounded-2xl border border-white/15 bg-white/10 p-4 text-base leading-relaxed min-h-[72px] transition-opacity duration-150",
                fading ? "opacity-0" : "opacity-100",
              )}
            >
              {scriptLine || "准备中..."}
            </div>
          </div>

          {/* Timeline chart */}
          <div className="w-[340px]">
            <div className="space-y-1.5">
              {SECTION_ORDER.map((s) => (
                <div key={s.key} className="flex items-center gap-2">
                  <div className="w-20 text-xs text-white/80 text-right">
                    {s.label}
                  </div>
                  <div className="flex-1 h-6 rounded-full bg-white/10 overflow-hidden">
                    <div
                      className={cn(
                        "h-full rounded-full transition-all duration-500",
                        s.key === currentSection
                          ? "bg-yellow-400"
                          : "bg-white/20",
                      )}
                      style={{
                        width: `${(s.seconds / 60) * 100}%`,
                      }}
                    />
                  </div>
                  <div className="w-10 text-xs text-white/70">{s.seconds}s</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="mt-4 flex items-center gap-6 text-sm text-white/80">
          <span>自动轮播讲解词：{autoScript ? "开启" : "关闭"}</span>
          <span>自动切换页签：{autoSwitch ? "开启" : "关闭"}</span>
          <span>间隔：{intervalS} 秒</span>
        </div>
      </div>

      {/* Controls */}
      <div className="glass-card rounded-3xl p-5">
        <div className="grid grid-cols-3 gap-6">
          <div>
            <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider">评委视角</label>
            <select
              value={judge}
              onChange={(e) => useAppStore.getState().setJudge(e.target.value as any)}
              className="mt-1 w-full rounded-xl border border-slate-700/50 bg-slate-800/60 px-3 py-2.5 text-sm text-slate-200 focus:border-cyan-400/50 focus:outline-none"
            >
              <option value="综合评委">综合评委</option>
              <option value="技术评委">技术评委</option>
              <option value="应用评委">应用评委</option>
            </select>
          </div>
          <div>
            <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider">展示层级</label>
            <select
              value={stage}
              onChange={(e) => useAppStore.getState().setStage(e.target.value as any)}
              className="mt-1 w-full rounded-xl border border-slate-700/50 bg-slate-800/60 px-3 py-2.5 text-sm text-slate-200 focus:border-cyan-400/50 focus:outline-none"
            >
              <option value="Level 1 快速体验">Level 1 快速体验</option>
              <option value="Level 2 核心功能">Level 2 核心功能</option>
              <option value="Level 3 技术深度">Level 3 技术深度</option>
              <option value="Level 4 创新扩展">Level 4 创新扩展</option>
            </select>
          </div>
          <div className="space-y-3">
            <Switch checked={autoScript} onCheckedChange={toggleAutoScript} label="自动轮播讲解词" />
            <Switch checked={autoSwitch} onCheckedChange={toggleAutoSwitch} label="自动切换页签" />
          </div>
        </div>
        <div className="mt-4">
          <Slider value={intervalS} min={10} max={90} step={5} label="切换间隔（秒）" onChange={setIntervalS} />
        </div>
      </div>

      {/* Navigation */}
      <div className="flex justify-between">
        <Button variant="outline" onClick={goPrev}>
          <ChevronLeft size={16} />
          {SECTION_ORDER[(currentIndex - 1 + SECTION_ORDER.length) % SECTION_ORDER.length]?.label}
        </Button>
        <Button variant="outline" onClick={goNext}>
          {SECTION_ORDER[(currentIndex + 1) % SECTION_ORDER.length]?.label}
          <ChevronRight size={16} />
        </Button>
      </div>
    </div>
  );
}
