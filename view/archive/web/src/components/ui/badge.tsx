import { cn } from "@/lib/utils";

function Badge({
  className,
  variant = "default",
  ...props
}: React.HTMLAttributes<HTMLSpanElement> & { variant?: "default" | "outline" }) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-3 py-1 text-xs font-medium",
        variant === "default" &&
          "bg-white/15 border border-white/20 text-white",
        variant === "outline" &&
          "border border-border bg-transparent text-foreground",
        className,
      )}
      {...props}
    />
  );
}

export { Badge };
