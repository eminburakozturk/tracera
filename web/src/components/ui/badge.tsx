import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "../../lib/utils";

const badgeVariants = cva(
  "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium transition-colors",
  {
    variants: {
      variant: {
        default:
          "border-[var(--accent)]/30 bg-[var(--accent-soft)] text-[var(--accent-ink)]",
        secondary:
          "border-[var(--border-hairline)] bg-[var(--bg-elevated)] text-[var(--text-primary)]",
        outline:
          "border-[var(--border-strong)] text-[var(--text-primary)]",
        success:
          "border-emerald-700/35 bg-emerald-500/15 text-emerald-800 dark:border-emerald-500/30 dark:text-emerald-300",
        warning:
          "border-amber-700/35 bg-amber-500/15 text-amber-900 dark:border-amber-500/30 dark:text-amber-300",
        danger:
          "border-rose-700/35 bg-rose-500/15 text-rose-800 dark:border-rose-500/30 dark:text-rose-300",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  },
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

export function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <span className={cn(badgeVariants({ variant }), className)} {...props} />
  );
}
