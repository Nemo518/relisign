import { useEffect, useRef } from "react";
import type { NormalizedLandmark } from "@mediapipe/tasks-vision";

export interface Connection {
  start: number;
  end: number;
}

interface Props {
  landmarksRef: React.RefObject<NormalizedLandmark[] | null>;
  connections: Connection[];
  width: number;
  height: number;
}

// MediaPipe 33-landmark body-part grouping
function getGroup(idx: number): "face" | "body" | "left_hand" | "right_hand" {
  if (idx <= 10) return "face";
  if (idx === 11 || idx === 12 || (idx >= 23 && idx <= 32)) return "body";
  if (idx === 13 || idx === 15 || idx === 17 || idx === 19 || idx === 21) return "left_hand";
  return "right_hand";
}

const PART_COLORS: Record<string, { stroke: string; fill: string }> = {
  body:    { stroke: "#22d3ee", fill: "#06b6d4" },
  left_hand:  { stroke: "#c084fc", fill: "#a855f7" },
  right_hand: { stroke: "#f472b6", fill: "#ec4899" },
  face:    { stroke: "#fbbf24", fill: "#f59e0b" },
};

export function LivePoseOverlay({ landmarksRef, connections, width, height }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);

  // Single rAF loop that reads landmarks from ref and updates SVG DOM directly
  // No React re-renders involved — pure imperative SVG manipulation
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;

    let running = true;
    let raf = 0;
    const DRAW_INTERVAL = 50; // ~20 fps

    // Pre-create / reuse DOM elements for performance
    const boneGroup = svg.querySelector<SVGGElement>("#bones")!;
    const kpGroup = svg.querySelector<SVGGElement>("#keypoints")!;

    const boneEls: { el: SVGGElement; glowLine: SVGLineElement; coreLine: SVGLineElement }[] = [];
    const kpEls: { el: SVGGElement; glow: SVGCircleElement; core: SVGCircleElement; highlight: SVGCircleElement }[] = [];

    // Initialize bone elements once
    for (const conn of connections) {
      const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
      const glow = document.createElementNS("http://www.w3.org/2000/svg", "line");
      const core = document.createElementNS("http://www.w3.org/2000/svg", "line");
      glow.setAttribute("stroke-linecap", "round");
      core.setAttribute("stroke-linecap", "round");
      g.appendChild(glow);
      g.appendChild(core);
      boneGroup.appendChild(g);
      boneEls.push({ el: g, glowLine: glow, coreLine: core });
    }

    // Initialize keypoint elements (33 landmarks)
    for (let i = 0; i < 33; i++) {
      const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
      const glow = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      const core = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      const highlight = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      g.appendChild(glow);
      g.appendChild(core);
      g.appendChild(highlight);
      kpGroup.appendChild(g);
      kpEls.push({ el: g, glow, core, highlight });
    }

    let lastDraw = 0;

    const draw = (now: number) => {
      if (!running) return;
      raf = requestAnimationFrame(draw);

      if (now - lastDraw < DRAW_INTERVAL) return;
      lastDraw = now;

      const landmarks = landmarksRef.current;
      if (!landmarks || landmarks.length === 0) {
        // Hide all elements
        for (const b of boneEls) b.el.setAttribute("opacity", "0");
        for (const k of kpEls) k.el.setAttribute("opacity", "0");
        return;
      }

      const pad = 4 / Math.min(width, height); // small padding

      // Update bones
      for (let i = 0; i < connections.length; i++) {
        const conn = connections[i];
        const a = landmarks[conn.start];
        const b = landmarks[conn.end];
        const bEl = boneEls[i];
        if (!bEl) continue;

        if (!a || !b || a.visibility < 0.2 || b.visibility < 0.2) {
          bEl.el.setAttribute("opacity", "0");
          continue;
        }

        const group = getGroup(conn.start);
        const colors = PART_COLORS[group];
        const minVis = Math.min(a.visibility, b.visibility);
        const ax = a.x;
        const ay = a.y;
        const bx = b.x;
        const by = b.y;

        // Glow
        bEl.glowLine.setAttribute("x1", String(ax));
        bEl.glowLine.setAttribute("y1", String(ay));
        bEl.glowLine.setAttribute("x2", String(bx));
        bEl.glowLine.setAttribute("y2", String(by));
        bEl.glowLine.setAttribute("stroke", colors.stroke);
        bEl.glowLine.setAttribute("stroke-width", "0.022");
        bEl.glowLine.setAttribute("opacity", String(minVis * 0.45));
        bEl.glowLine.setAttribute("filter", "url(#pose-glow)");

        // Core
        bEl.coreLine.setAttribute("x1", String(ax));
        bEl.coreLine.setAttribute("y1", String(ay));
        bEl.coreLine.setAttribute("x2", String(bx));
        bEl.coreLine.setAttribute("y2", String(by));
        bEl.coreLine.setAttribute("stroke", colors.stroke);
        bEl.coreLine.setAttribute("stroke-width", "0.010");
        bEl.coreLine.setAttribute("opacity", String(0.2 + minVis * 0.8));
        bEl.coreLine.removeAttribute("filter");

        bEl.el.setAttribute("opacity", "1");
      }

      // Update keypoints
      for (let i = 0; i < 33; i++) {
        const lm = landmarks[i];
        const kEl = kpEls[i];
        if (!kEl) continue;

        if (!lm || lm.visibility < 0.08) {
          kEl.el.setAttribute("opacity", "0");
          continue;
        }

        const group = getGroup(i);
        const colors = PART_COLORS[group];
        const cx = lm.x;
        const cy = lm.y;
        const r = group === "face" ? 0.008 : 0.013;

        // Glow
        kEl.glow.setAttribute("cx", String(cx));
        kEl.glow.setAttribute("cy", String(cy));
        kEl.glow.setAttribute("r", String(r * 3));
        kEl.glow.setAttribute("fill", colors.stroke);
        kEl.glow.setAttribute("opacity", String(lm.visibility * 0.3));
        kEl.glow.setAttribute("filter", "url(#pose-glow)");

        // Core
        kEl.core.setAttribute("cx", String(cx));
        kEl.core.setAttribute("cy", String(cy));
        kEl.core.setAttribute("r", String(r * (0.5 + lm.visibility * 0.5)));
        kEl.core.setAttribute("fill", colors.fill);
        kEl.core.setAttribute("opacity", String(0.5 + lm.visibility * 0.5));
        kEl.core.removeAttribute("filter");

        // Highlight for high-confidence points
        kEl.highlight.setAttribute("cx", String(cx));
        kEl.highlight.setAttribute("cy", String(cy));
        kEl.highlight.setAttribute("r", String(r * 0.35));
        kEl.highlight.setAttribute("fill", "white");
        kEl.highlight.setAttribute("opacity", lm.visibility >= 0.6 ? String(0.4 + lm.visibility * 0.35) : "0");

        kEl.el.setAttribute("opacity", "1");
      }
    };

    raf = requestAnimationFrame(draw);
    return () => {
      running = false;
      cancelAnimationFrame(raf);
      for (const b of boneEls) b.el.remove();
      for (const k of kpEls) k.el.remove();
    };
  }, [connections, width, height]);

  const vb = `0 0 1 1`;

  return (
    <svg
      ref={svgRef}
      viewBox={vb}
      className="absolute inset-0 w-full h-full pointer-events-none z-10"
      preserveAspectRatio="none"
    >
      <defs>
        <filter id="pose-glow" x="-0.5" y="-0.5" width="2" height="2">
          <feGaussianBlur stdDeviation="0.018" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
      <g id="bones" />
      <g id="keypoints" />
    </svg>
  );
}
