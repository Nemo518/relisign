import { cn } from "@/lib/utils";
import { type Section } from "@/types/presentation";

interface TabsProps {
  sections: Section[];
  activeKey: string;
  onSelect: (key: string) => void;
}

function Tabs({ sections, activeKey, onSelect }: TabsProps) {
  return (
    <div className="flex flex-wrap gap-2">
      {sections.map((section) => (
        <button
          key={section.key}
          onClick={() => onSelect(section.key)}
          className={cn(
            "rounded-xl px-5 py-2.5 text-sm font-semibold transition-all duration-200",
            section.key === activeKey
              ? "bg-gradient-to-r from-cyan-400 to-teal-400 text-slate-900 shadow-md"
              : "border border-border bg-white text-foreground hover:bg-muted shadow-sm",
          )}
        >
          {section.label}
        </button>
      ))}
    </div>
  );
}

export { Tabs };
