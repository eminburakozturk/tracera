import * as React from "react";
import { BookOpen, Info } from "lucide-react";
import { useTranslation } from "react-i18next";
import {
  CLASS_DICTIONARY,
  getClassColor,
  getClassDescription,
  getClassDisplayName,
  GUIDE_DATASET_NOTE_EN,
  GUIDE_DATASET_NOTE_TR,
} from "../lib/v11Rules";
import { useAppStore } from "../store/useAppStore";
import type {
  ClassMetricItem,
  ClassMetricsData,
  GuideCropItem,
  GuideIndexData,
} from "../types";
import { Badge } from "./ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./ui/card";
import { fmtNum, fmtPct } from "../lib/format";

interface GuideViewProps {
  initialGuideIndex?: GuideIndexData | null;
  initialClassMetrics?: ClassMetricsData | null;
}

const ORDERED_CLASSES = Object.keys(CLASS_DICTIONARY);

export function GuideView({
  initialGuideIndex,
  initialClassMetrics,
}: GuideViewProps) {
  const { t } = useTranslation();
  const lang = useAppStore((s) => s.lang);

  const [guideIndex, setGuideIndex] = React.useState<GuideIndexData | null>(
    initialGuideIndex ?? null,
  );
  const [classMetrics, setClassMetrics] =
    React.useState<ClassMetricsData | null>(initialClassMetrics ?? null);

  React.useEffect(() => {
    if (
      initialGuideIndex !== undefined ||
      initialClassMetrics !== undefined
    ) {
      setGuideIndex(initialGuideIndex ?? null);
      setClassMetrics(initialClassMetrics ?? null);
      return;
    }
    let active = true;
    void fetch("/guide/guide_index.json")
      .then(async (r) => (r.ok ? ((await r.json()) as GuideIndexData) : null))
      .then((d) => {
        if (active && d) setGuideIndex(d);
      })
      .catch(() => {});

    void fetch("/data/class_metrics.json")
      .then(async (r) => (r.ok ? ((await r.json()) as ClassMetricsData) : null))
      .then((d) => {
        if (active && d) setClassMetrics(d);
      })
      .catch(() => {});

    return () => {
      active = false;
    };
  }, [initialGuideIndex, initialClassMetrics]);

  const cropsByClass = React.useMemo(() => {
    const map: Record<string, GuideCropItem[]> = {};
    for (const item of guideIndex?.items ?? []) {
      if (!["05", "11"].includes(item.kart_no)) continue;
      map[item.class] = [...(map[item.class] ?? []), item];
    }
    return map;
  }, [guideIndex]);

  const metricsByClass = React.useMemo(() => {
    const map: Record<string, ClassMetricItem> = {};
    for (const item of classMetrics?.classes ?? []) {
      map[item.class] = item;
    }
    return map;
  }, [classMetrics]);

  return (
    <div className="space-y-5" data-testid="guide-view">
      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <BookOpen className="h-5 w-5 text-[var(--accent-ink)]" />
              <CardTitle>{t("guide.title")}</CardTitle>
            </div>
            <Badge variant="secondary" className="font-mono">
              Val: Kart 05 & Kart 11
            </Badge>
          </div>
          <CardDescription>{t("guide.subtitle")}</CardDescription>
        </CardHeader>

        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            {ORDERED_CLASSES.map((cls) => {
              const color = getClassColor(cls);
              const crops = cropsByClass[cls] ?? [];
              const m = metricsByClass[cls];
              const cropSize = guideIndex?.crop_size ?? 320;

              return (
                <div
                  key={cls}
                  data-testid={`guide-card-${cls}`}
                  className="elevated-card flex flex-col justify-between gap-3 rounded-xl p-4"
                >
                  <div className="space-y-2">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <span
                          className="h-3 w-3 rounded-full"
                          style={{ backgroundColor: color }}
                        />
                        <h3 className="text-sm font-bold text-[var(--text-primary)]">
                          {getClassDisplayName(cls, lang)}
                        </h3>
                      </div>
                      <Badge variant="secondary" className="font-mono">
                        {cls}
                      </Badge>
                    </div>

                    <p
                      data-testid={`guide-desc-${cls}`}
                      className="text-xs leading-relaxed text-[var(--text-secondary)]"
                    >
                      {getClassDescription(cls, lang)}
                    </p>
                  </div>

                  {/* Val Kart 05 & 11 Kırpımları */}
                  {crops.length > 0 && (
                    <div className="grid grid-cols-2 gap-2.5">
                      {crops.slice(0, 2).map((crop) => {
                        const [bx1, by1, bx2, by2] = crop.box_in_crop;
                        return (
                          <figure key={crop.id} className="space-y-1">
                            <div className="relative aspect-square w-full overflow-hidden rounded-lg border border-[var(--border-hairline)] bg-[var(--bg-canvas)]">
                              <img
                                src={crop.path}
                                alt={`${cls} Kart ${crop.kart_no}`}
                                className="h-full w-full object-cover"
                                loading="lazy"
                              />
                              <svg
                                viewBox={`0 0 ${cropSize} ${cropSize}`}
                                className="pointer-events-none absolute inset-0 h-full w-full"
                              >
                                <rect
                                  x={bx1}
                                  y={by1}
                                  width={Math.max(4, bx2 - bx1)}
                                  height={Math.max(4, by2 - by1)}
                                  fill="none"
                                  stroke={color}
                                  strokeWidth="4"
                                />
                              </svg>
                            </div>
                            <figcaption className="font-mono text-[10px] text-[var(--text-muted)]">
                              Kart {crop.kart_no} · {crop.source_image}
                            </figcaption>
                          </figure>
                        );
                      })}
                    </div>
                  )}

                  {/* Sınıf Bazlı Metrikler (class_metrics.json) */}
                  {m && (
                    <div className="grid grid-cols-3 gap-2 rounded-lg border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-2.5 font-mono text-[11px] tabular-nums">
                      <div>
                        <p className="font-sans text-[10px] text-[var(--text-muted)]">
                          {t("guide.valMetrics")}
                        </p>
                        <p className="font-semibold text-[var(--text-primary)]">
                          F1 {fmtNum(m.val.F1, 3)}
                        </p>
                        <p className="text-[10px] text-[var(--text-secondary)]">
                          P {fmtPct(m.val.P, 0)} · R{" "}
                          {fmtPct(m.val.R, 0)}
                        </p>
                      </div>
                      <div>
                        <p className="font-sans text-[10px] text-[var(--text-muted)]">
                          {t("guide.testMetrics")}
                        </p>
                        <p className="font-semibold text-[var(--text-primary)]">
                          F1 {fmtNum(m.test.F1, 3)}
                        </p>
                        <p className="text-[10px] text-[var(--text-secondary)]">
                          P {fmtPct(m.test.P, 0)} · R{" "}
                          {fmtPct(m.test.R, 0)}
                        </p>
                      </div>
                      <div>
                        <p className="font-sans text-[10px] text-[var(--text-muted)]">
                          {t("guide.cvMetrics")}
                        </p>
                        <p className="font-semibold text-[var(--accent-ink)]">
                          {m.cv_1024_last
                            ? `${fmtNum(m.cv_1024_last.mAP50_95_mean, 3)}`
                            : "—"}
                        </p>
                        <p className="text-[10px] text-[var(--text-secondary)]">
                          {m.cv_1024_last
                            ? `±${fmtNum(m.cv_1024_last.mAP50_95_std, 3)}`
                            : ""}
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Zorunlu Alt Not (docs/UI_V11_RULES.md) */}
          <div
            data-testid="guide-dataset-note"
            className="flex items-start gap-2.5 rounded-xl border border-[var(--border-hairline)] bg-[var(--bg-elevated)] p-3.5 text-xs text-[var(--text-secondary)]"
          >
            <Info className="mt-0.5 h-4 w-4 shrink-0 text-[var(--accent-ink)]" />
            <span>
              {lang === "en" ? GUIDE_DATASET_NOTE_EN : GUIDE_DATASET_NOTE_TR}
            </span>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
