import * as React from "react";
import { AlertTriangle } from "lucide-react";
import {
  LOW_CONF_REVIEW_LABEL_EN,
  LOW_CONF_REVIEW_LABEL_TR,
  LOW_CONF_REVIEW_TOOLTIP_EN,
  LOW_CONF_REVIEW_TOOLTIP_TR,
} from "../lib/v11Rules";
import type { LangCode } from "../store/useAppStore";
import { Badge } from "./ui/badge";

interface LowConfidenceReviewBadgeProps {
  idSuffix: string;
  lang?: LangCode;
  badgeTestId?: string;
  infoBtnTestId?: string;
  popoverTestId?: string;
}

export function LowConfidenceReviewBadge({
  idSuffix,
  lang = "tr",
  badgeTestId,
  infoBtnTestId,
  popoverTestId,
}: LowConfidenceReviewBadgeProps) {
  const [open, setOpen] = React.useState(false);
  const popoverId = `low-conf-popover-desc-${idSuffix}`;

  const labelText =
    lang === "en" ? LOW_CONF_REVIEW_LABEL_EN : LOW_CONF_REVIEW_LABEL_TR;
  const tooltipText =
    lang === "en" ? LOW_CONF_REVIEW_TOOLTIP_EN : LOW_CONF_REVIEW_TOOLTIP_TR;

  return (
    <div className="relative inline-flex flex-col items-start gap-1">
      <div className="inline-flex items-center gap-1.5">
        <Badge
          variant="warning"
          data-testid={badgeTestId ?? `low-conf-badge-${idSuffix}`}
          aria-describedby={open ? popoverId : undefined}
        >
          <AlertTriangle className="h-3 w-3 shrink-0" />
          <span>{labelText}</span>
        </Badge>
        <button
          type="button"
          data-testid={infoBtnTestId ?? `low-conf-info-btn-${idSuffix}`}
          aria-label={
            lang === "en"
              ? "Low confidence review explanation"
              : "Düşük güven etiketi açıklaması"
          }
          aria-expanded={open}
          aria-controls={popoverId}
          aria-describedby={open ? popoverId : undefined}
          onClick={(e) => {
            e.stopPropagation();
            setOpen((prev) => !prev);
          }}
          className="inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-full border border-amber-700/45 bg-amber-500/20 font-mono text-[11px] font-bold text-amber-900 transition-colors hover:bg-amber-500/30 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] dark:border-amber-400/40 dark:text-amber-200 cursor-pointer"
        >
          i
        </button>
      </div>

      {open && (
        <div
          id={popoverId}
          role="tooltip"
          data-testid={popoverTestId ?? `low-conf-popover-${idSuffix}`}
          onClick={(e) => e.stopPropagation()}
          className="z-20 max-w-full rounded-lg border border-amber-700/40 bg-[var(--bg-surface)] p-2.5 text-[11px] leading-relaxed font-normal text-[var(--text-primary)] shadow-md sm:max-w-xs dark:border-amber-500/40"
        >
          {tooltipText}
        </div>
      )}
    </div>
  );
}
