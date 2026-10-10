import * as React from "react";
import { cn } from "@/lib/utils";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "default" | "primary" | "outline" | "ghost";
  size?: "sm" | "default" | "lg";
}

function Button({
  className,
  variant = "default",
  size = "default",
  ...props
}: ButtonProps) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-2xl font-medium transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50",
        size === "sm" && "h-8 px-3 text-sm",
        size === "default" && "h-10 px-5 text-sm",
        size === "lg" && "h-12 px-6 text-base",
        variant === "default" &&
          "border border-border bg-white text-foreground shadow-sm hover:bg-muted hover:shadow-md",
        variant === "primary" &&
          "bg-gradient-to-r from-cyan-400 to-teal-400 text-slate-900 font-semibold shadow-md hover:shadow-lg hover:brightness-105",
        variant === "outline" &&
          "border-2 border-border bg-transparent hover:bg-muted",
        variant === "ghost" &&
          "bg-transparent hover:bg-muted",
        className,
      )}
      {...props}
    />
  );
}

export { Button, type ButtonProps };
