import * as React from "react";
import { useTranslation } from "react-i18next";
import type { PrCurveData, PrCurvePoint } from "../types";
import { Badge } from "./ui/badge";
import { fmtNum, fmtPct } from "../lib/format";

interface PrMiniChartProps {
  confThreshold: number;
  initialData?: PrCurveData | null;
}

export function PrMiniChart({ confThreshold, initialData }: PrMiniChartProps) {
  const { t, i18n } = useTranslation();
  const lang = i18n.language.startsWith("en") ? "en" : "tr";
  const [data, setData] = React.useState<PrCurveData | null>(
    initialData ?? null,
  );

  React.useEffect(() => {
    if (initialData !== undefined) {
      setData(initialData);
      return;
    }
    let active = true;
    void fetch("/data/pr_curve_val.json")
      .then(async (res) => {
        if (!res.ok) return null;
        return (await res.json()) as PrCurveData;
      })
      .then((json) => {
        if (active && json && Array.isArray(json.points) && json.points.length > 0) {
          setData(json);
        }
      })
      .catch(() => {
        // Dosya yoksa grafik gizlenir
      });
    return () => {
      active = false;
    };
  }, [initialData]);

  if (!data || !Array.isArray(data.points) || data.points.length === 0) {
    return null;
  }

  // Tespit olmayan eşiklerde (TP+FP=0) kesinlik tanımsızdır; dosyada 0 yazılı olsa da çizilmez.
  const definedPoints = data.points.filter((p) => p.TP + p.FP > 0);
  if (definedPoints.length === 0) {
    return null;
  }
  const sortedByRecall = [...definedPoints].sort((a, b) => a.R - b.R);
  const maxDefinedConf = Math.max(...definedPoints.map((p) => p.conf));
  const noDetectionsAtThreshold = confThreshold > maxDefinedConf + 0.005;
  const nearestPoint: PrCurvePoint = definedPoints.reduce((best, pt) =>
    Math.abs(pt.conf - confThreshold) < Math.abs(best.conf - confThreshold)
      ? pt
      : best,
  );
  const opPoint =
    data.operating_point ??
    definedPoints.find((p) => p.is_operating_point) ??
    nearestPoint;

  const w = 240;
  const h = 88;
  const padX = 16;
  const padY = 10;
  const innerW = w - padX * 2;
  const innerH = h - padY * 2;

  const toX = (r: number) =>
    padX + Math.max(0, Math.min(1, r)) * innerW;
  const toY = (p: number) =>
    padY + (1 - Math.max(0, Math.min(1, p))) * innerH;

  const pathD = sortedByRecall
    .map((pt, idx) => `${idx === 0 ? "M" : "L"}${toX(pt.R).toFixed(1)},${toY(pt.P).toFixed(1)}`)
    .join(" ");

  return (
    <div
      data-testid="pr-mini-chart"
      className="rounded-xl border border-[var(--border-hairline)] bg-[var(--bg-elevated)] p-3"
    >
      <div className="mb-1.5 flex items-center justify-between gap-2">
        <span className="text-[11px] font-semibold text-[var(--text-secondary)]">
          {t("detect.prCurveTitle")}
        </span>
        <Badge
          variant="secondary"
          className="px-2 py-0 font-mono text-[10px]"
          data-testid="pr-curve-val-badge"
        >
          {t("detect.prCurveBadge")}
        </Badge>
      </div>

      <svg
        viewBox={`0 0 ${w} ${h}`}
        className="h-20 w-full overflow-visible"
        role="img"
        aria-label={t("detect.prCurveTitle")}
      >
        <line
          x1={padX}
          y1={h - padY}
          x2={w - padX}
          y2={h - padY}
          stroke="var(--border-hairline)"
          strokeWidth="1"
        />
        <line
          x1={padX}
          y1={padY}
          x2={padX}
          y2={h - padY}
          stroke="var(--border-hairline)"
          strokeWidth="1"
        />
        <path
          d={pathD}
          fill="none"
          stroke="var(--accent)"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        {/* Kilitli çalışma noktası (0.395) */}
        <circle
          cx={toX(opPoint.R)}
          cy={toY(opPoint.P)}
          r="3.5"
          fill="var(--bg-surface)"
          stroke="var(--accent)"
          strokeWidth="2"
        />
        {/* Aktif eşik noktası */}
        {!noDetectionsAtThreshold && (
        <circle
          cx={toX(nearestPoint.R)}
          cy={toY(nearestPoint.P)}
          r="4.5"
          fill="var(--accent)"
        />
        )}
        <text
          x={w - padX}
          y={h - 1}
          textAnchor="end"
          fontSize="8"
          fill="var(--text-muted)"
        >
          {lang === "en" ? "Recall →" : "Duyarlılık →"}
        </text>
        <text
          x={2}
          y={padY - 2}
          fontSize="8"
          fill="var(--text-muted)"
        >
          {lang === "en" ? "↑ Precision" : "↑ Kesinlik"}
        </text>
      </svg>

      <div className="mt-1 flex items-center justify-between font-mono text-[11px] text-[var(--text-secondary)] tabular-nums">
        <span>
          {lang === "en" ? "conf" : "eşik"}≈{fmtNum(nearestPoint.conf, 3)}
        </span>
        {noDetectionsAtThreshold ? (
          <span data-testid="pr-no-detections">
            {lang === "en"
              ? "No validation detections at this threshold"
              : "Bu eşikte val setinde tespit yok"}
          </span>
        ) : (
          <span>
            P {fmtPct(nearestPoint.P, 1)} · R{" "}
            {fmtPct(nearestPoint.R, 1)} · F1{" "}
            {fmtNum(nearestPoint.F1, 3)}
          </span>
        )}
      </div>
    </div>
  );
}
