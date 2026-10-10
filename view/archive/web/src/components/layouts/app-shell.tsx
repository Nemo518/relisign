import { Outlet } from "react-router-dom";
import { SidebarControls } from "./sidebar-controls";
import { HeroBanner } from "@/components/hero/hero-banner";
import { TabNavigation } from "./tab-navigation";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/store/app-store";
import { useEffect, useMemo } from "react";

function StarfieldBg() {
  const stars = useMemo(() => {
    const items: { x: number; y: number; size: number; dur: number; delay: number }[] = [];
    for (let i = 0; i < 100; i++) {
      items.push({
        x: Math.random() * 100,
        y: Math.random() * 100,
        size: Math.random() * 2 + 1,
        dur: Math.random() * 3 + 2,
        delay: Math.random() * 3,
      });
    }
    return items;
  }, []);

  return (
    <div className="starfield">
      {stars.map((s, i) => (
        <div
          key={i}
          className="star"
          style={{
            left: `${s.x}%`,
            top: `${s.y}%`,
            width: s.size,
            height: s.size,
            "--dur": `${s.dur}s`,
            "--delay": `${s.delay}s`,
          } as React.CSSProperties}
        />
      ))}
    </div>
  );
}

export function AppShell() {
  const darkMode = useAppStore((s) => s.darkMode);

  useEffect(() => {
    document.documentElement.classList.toggle("light", !darkMode);
  }, [darkMode]);

  return (
    <div className="relative flex min-h-screen bg-background">
      <StarfieldBg />
      <div className="grid-overlay" />
      <SidebarControls />
      <main className="relative z-10 flex-1 overflow-auto">
        <div className="mx-auto max-w-7xl px-6 py-4 pb-10">
          <HeroBanner />
          <TabNavigation />
          <div className={cn("animate-fade-in")}>
            <Outlet />
          </div>
        </div>
      </main>
    </div>
  );
}
