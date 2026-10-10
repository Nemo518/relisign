import { useNavigate, useLocation } from "react-router-dom";
import { SECTION_ORDER } from "@/lib/constants";
import { cn } from "@/lib/utils";

export function TabNavigation() {
  const navigate = useNavigate();
  const location = useLocation();

  const currentSection =
    SECTION_ORDER.find(
      (s) =>
        location.pathname === `/${s.key}` ||
        location.pathname.startsWith(`/${s.key}/`),
    )?.key ?? null;

  return (
    <div className="flex flex-wrap gap-2 py-4">
      {SECTION_ORDER.map((section) => (
        <button
          key={section.key}
          onClick={() => navigate(`/${section.key}`)}
          className={cn(
            "rounded-xl px-5 py-2.5 text-sm font-semibold transition-all duration-300",
            section.key === currentSection
              ? "bg-gradient-to-r from-cyan-400 to-teal-500 text-slate-900 shadow-lg shadow-cyan-500/25 scale-105"
              : "border border-slate-700/50 bg-slate-800/60 text-slate-400 hover:text-slate-200 hover:border-slate-600/50 backdrop-blur-sm",
          )}
        >
          {section.label}
          <span className={cn(
            "ml-1.5 text-[10px] font-mono",
            section.key === currentSection ? "text-slate-700" : "text-slate-400",
          )}>
            {section.seconds}s
          </span>
        </button>
      ))}
    </div>
  );
}
