import * as React from "react";
import { AlertTriangle, CheckCircle2, Download, Flame, Info, Loader2 } from "lucide-react";
import { useTranslation } from "react-i18next";
import { downloadBase64Png } from "../lib/download";
import { buildReliabilitySummary } from "../lib/reliability";
import {
  getClassColor,
  getClassDescription,
  getClassDisplayName,
} from "../lib/v11Rules";
import { useAppStore } from "../store/useAppStore";
import type { ReliabilityData } from "../types";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { Skeleton } from "./ui/skeleton";
import { Slider } from "./ui/slider";
import { fmtNum, fmtPct, fmtMs } from "../lib/format";

interface ExplainViewProps {
  onRunExplain: () => Promise<void>;
  initialValReliability?: ReliabilityData | null;
  initialTestReliability?: ReliabilityData | null;
}

export function ExplainView({
  onRunExplain,
  initialValReliability,
  initialTestReliability,
}: ExplainViewProps) {
  const { t } = useTranslation();
  const lang = useAppStore((s) => s.lang);
  const meta = useAppStore((s) => s.meta);
  const metrics = useAppStore((s) => s.metrics);
  const backendChoice = useAppStore((s) => s.backendChoice);
  const localApiAvailable = useAppStore((s) => s.localApiAvailable);
  const selectedImage = useAppStore((s) => s.selectedImage);
  const confThreshold = useAppStore((s) => s.confThreshold);
  const detectResult = useAppStore((s) => s.detectResult);
  const selectedDetectionId = useAppStore((s) => s.selectedDetectionId);
  const setSelectedDetectionId = useAppStore((s) => s.setSelectedDetectionId);
  const explainResult = useAppStore((s) => s.explainResult);
  const explainSource = useAppStore((s) => s.explainSource);
  const explainDetectionId = useAppStore((s) => s.explainDetectionId);
  const explainStatus = useAppStore((s) => s.explainStatus);
  const cloudSpaceId = useAppStore((s) => s.cloudSpaceId);
  const explainLoading = useAppStore((s) => s.explainLoading);
  const explainError = useAppStore((s) => s.explainError);
  const heatmapOpacity = useAppStore((s) => s.heatmapOpacity);
  const setHeatmapOpacity = useAppStore((s) => s.setHeatmapOpacity);

  const [valReliability, setValReliability] =
    React.useState<ReliabilityData | null>(initialValReliability ?? null);
  const [testReliability, setTestReliability] =
    React.useState<ReliabilityData | null>(initialTestReliability ?? null);

  React.useEffect(() => {
    if (
      initialValReliability !== undefined ||
      initialTestReliability !== undefined
    ) {
      setValReliability(initialValReliability ?? null);
      setTestReliability(initialTestReliability ?? null);
      return;
    }
    let active = true;
    void fetch("/data/reliability_val.json")
      .then(async (res) => (res.ok ? ((await res.json()) as ReliabilityData) : null))
      .then((data) => {
        if (active && data) setValReliability(data);
      })
      .catch(() => {});

    void fetch("/data/reliability_test.json")
      .then(async (res) => (res.ok ? ((await res.json()) as ReliabilityData) : null))
      .then((data) => {
        if (active && data) setTestReliability(data);
      })
      .catch(() => {});

    return () => {
      active = false;
    };
  }, [initialValReliability, initialTestReliability]);

  const renderExplainSourceLabel = () => {
    if (!explainSource) return null;
    if (explainSource === "api") return t("explain.sourceApi");
    if (explainSource === "cloud")
      return t("explain.sourceCloud", { space: cloudSpaceId });
    return t("explain.sourceMock");
  };

  const detections = detectResult?.detections ?? [];
  const selectedDet =
    detections.find((d) => d.id === selectedDetectionId) ??
    detections[0] ??
    null;

  const hasMatchingExplain = Boolean(
    !explainLoading &&
      !explainError &&
      explainStatus !== "error" &&
      explainResult &&
      selectedImage &&
      selectedDet &&
      (explainDetectionId === null || explainDetectionId === selectedDet.id),
  );
  const showErrorBanner = Boolean(
    !explainLoading && explainError && explainStatus !== "success",
  );

  const reliabilitySummary = React.useMemo(() => {
    if (!selectedDet) return null;
    return buildReliabilitySummary(
      selectedDet.conf,
      valReliability,
      testReliability,
      lang,
    );
  }, [selectedDet, valReliability, testReliability, lang]);

  const energyInBox =
    hasMatchingExplain &&
    typeof explainResult?.energy_in_box === "number" &&
    !Number.isNaN(explainResult.energy_in_box)
      ? Math.max(0, Math.min(1, explainResult.energy_in_box))
      : null;

  const layercamRow =
    metrics?.xai.find(
      (r) =>
        r.model === "g_y11s_1024_full" &&
        r.imgsz === 1024 &&
        r.katman === 4 &&
        r.method === "layercam",
    ) ??
    metrics?.xai.find((r) => r.katman === 4 && r.method === "layercam") ??
    null;

  return (
    <div className="space-y-5">
      <Card>
        <CardHeader className="pb-3">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <CardTitle>{t("explain.title")}</CardTitle>
            </div>
            {hasMatchingExplain && explainResult && selectedDet && (
              <div className="flex flex-wrap items-center gap-2">
                <Badge variant="outline" data-testid="explain-detection-badge">
                  #{selectedDet.id} · {selectedDet.cls} (
                  {fmtPct(selectedDet.conf, 1)})
                </Badge>
                {explainSource && (
                  <Badge variant="secondary" data-testid="explain-source-badge">
                    {t("explain.sourceLabel")}: {renderExplainSourceLabel()}
                  </Badge>
                )}
                <Badge variant="default" className="font-mono tabular-nums">
                  Layer {explainResult.layer} ·{" "}
                  {fmtMs(explainResult.timing.total_ms, 1)}
                </Badge>
              </div>
            )}
          </div>
        </CardHeader>

        <CardContent className="space-y-4">
          {(backendChoice === "ort" ||
            (!localApiAvailable && backendChoice !== "mock")) && (
            <div
              data-testid="explain-cloud-note"
              className="flex items-start gap-2 rounded-lg border border-[var(--accent)]/40 bg-[var(--accent-soft)] p-3 text-xs text-[var(--text-primary)]"
            >
              <Info className="mt-0.5 h-4 w-4 shrink-0 text-[var(--accent)]" />
              <span>
                {t(
                  localApiAvailable
                    ? "explain.apiRequiredNote"
                    : "explain.cloudOnlyNote",
                )}
              </span>
            </div>
          )}

          {!selectedImage || !selectedDet ? (
            <p className="rounded-lg border border-[var(--border-hairline)] bg-[var(--bg-elevated)] p-4 text-sm text-[var(--text-muted)]">
              {t("explain.noImageOrDet")}
            </p>
          ) : (
            <div className="grid grid-cols-1 gap-3 rounded-xl border border-[var(--border-hairline)] bg-[var(--bg-elevated)] p-3.5 md:grid-cols-12 md:items-center">
              <div className="space-y-1 md:col-span-5">
                <label className="text-xs font-medium text-[var(--text-secondary)]">
                  {t("explain.selectedDet")}
                </label>
                <Select
                  value={String(selectedDet.id)}
                  onValueChange={(val) => setSelectedDetectionId(Number(val))}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {detections.map((d) => (
                      <SelectItem key={d.id} value={String(d.id)}>
                        #{d.id} · {getClassDisplayName(d.cls, lang, true)} (
                        {fmtPct(d.conf, 1)})
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-1 md:col-span-4">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-[var(--text-secondary)]">
                    {t("explain.opacityLabel")}
                  </span>
                  <span className="font-mono text-[var(--accent-ink)] tabular-nums">
                    {Math.round(heatmapOpacity * 100)}%
                  </span>
                </div>
                <Slider
                  min={0.1}
                  max={1}
                  step={0.05}
                  value={[heatmapOpacity]}
                  onValueChange={([v]) => setHeatmapOpacity(v)}
                  aria-label={t("explain.opacityLabel")}
                />
              </div>

              <div className="flex justify-end md:col-span-3">
                <Button
                  type="button"
                  className="w-full"
                  disabled={explainLoading}
                  onClick={() => void onRunExplain()}
                  data-testid="generate-heatmap-btn"
                >
                  <Flame className="h-4 w-4" />
                  <span>
                    {explainLoading
                      ? explainStatus === "waking"
                        ? t("explain.wakingCloud")
                        : t("explain.generating")
                      : t("explain.generateBtn")}
                  </span>
                </Button>
              </div>
            </div>
          )}

          {showErrorBanner && (
            <div
              role="alert"
              data-testid="explain-error-banner"
              className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-rose-500/40 bg-rose-500/10 p-3 text-sm text-rose-800 dark:text-rose-200"
            >
              <div className="flex items-center gap-2">
                <AlertTriangle className="h-4 w-4 shrink-0" />
                <span>{explainError}</span>
              </div>
              <Button
                type="button"
                size="sm"
                variant="outline"
                onClick={() => void onRunExplain()}
              >
                {t("controls.retry")}
              </Button>
            </div>
          )}

          {explainLoading && (
            <div className="space-y-3" data-testid="explain-loading-state">
              <div
                role="status"
                data-testid="explain-status-banner"
                className="flex items-center gap-2 rounded-lg border border-[var(--accent)]/40 bg-[var(--accent-soft)] p-3 text-xs font-medium text-[var(--accent-ink)]"
              >
                <Loader2 className="h-4 w-4 shrink-0 animate-spin text-[var(--accent)]" />
                <span>
                  {explainStatus === "waking"
                    ? t("explain.wakingCloud")
                    : t("explain.generating")}
                </span>
              </div>
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                <Skeleton className="h-72 w-full" />
                <Skeleton className="h-72 w-full" />
              </div>
            </div>
          )}

          {hasMatchingExplain &&
            explainResult &&
            selectedImage &&
            selectedDet && (
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                <Card className="overflow-hidden">
                  <CardHeader className="flex flex-row items-center justify-between py-3">
                    <CardTitle className="text-sm">
                      {t("explain.zoomPanel")}
                    </CardTitle>
                    <Button
                      type="button"
                      size="sm"
                      variant="secondary"
                      onClick={() =>
                        downloadBase64Png(
                          explainResult.zoom_png,
                          `${selectedImage.name.replace(/\.[^.]+$/, "")}_layercam_zoom.png`,
                        )
                      }
                    >
                      <Download className="h-3.5 w-3.5" />
                      <span>{t("explain.downloadZoom")}</span>
                    </Button>
                  </CardHeader>
                  <CardContent>
                    <img
                      src={`data:image/png;base64,${explainResult.zoom_png}`}
                      alt={t("explain.zoomPanel")}
                      data-testid="explain-zoom-img"
                      className="w-full rounded-lg border border-[var(--border-hairline)] object-contain"
                    />
                  </CardContent>
                </Card>

                <Card className="overflow-hidden">
                  <CardHeader className="flex flex-row items-center justify-between py-3">
                    <CardTitle className="text-sm">
                      {t("explain.fullPanel")}
                    </CardTitle>
                    <Button
                      type="button"
                      size="sm"
                      variant="secondary"
                      data-testid="download-overlay-btn"
                      onClick={() =>
                        downloadBase64Png(
                          explainResult.overlay_png,
                          `${selectedImage.name.replace(/\.[^.]+$/, "")}_layercam_overlay.png`,
                        )
                      }
                    >
                      <Download className="h-3.5 w-3.5" />
                      <span>{t("explain.downloadOverlay")}</span>
                    </Button>
                  </CardHeader>
                  <CardContent>
                    <div className="relative overflow-hidden rounded-lg border border-[var(--border-hairline)]">
                      <img
                        src={selectedImage.url}
                        alt={selectedImage.name}
                        className="block h-auto w-full"
                      />
                      <img
                        src={`data:image/png;base64,${explainResult.overlay_png}`}
                        alt={t("explain.fullPanel")}
                        data-testid="explain-overlay-img"
                        className="absolute inset-0 h-full w-full object-cover transition-opacity"
                        style={{
                          opacity: heatmapOpacity,
                          borderColor:
                            meta.colors[selectedDet.cls_id] ?? "transparent",
                        }}
                      />
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}
        </CardContent>
      </Card>

      {/* "Bu karar nasıl verildi?" Kartı — Yalnız veriden cümleler (LLM YOK) */}
      {selectedDet && (
        <Card data-testid="how-decided-card">
          <CardHeader className="pb-2">
            <CardTitle>{t("explain.howDecidedTitle")}</CardTitle>
            <CardDescription>{t("explain.howDecidedSub")}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-sm text-[var(--text-primary)]">
            {/* 1. Güven skoru ve eşik farkı */}
            <div
              data-testid="how-decided-conf"
              className="flex items-start gap-2.5 rounded-xl border border-[var(--border-hairline)] bg-[var(--bg-elevated)] p-3"
            >
              <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-[var(--accent-ink)]" />
              <div className="font-mono text-xs tabular-nums">
                {lang === "en" ? (
                  <span>
                    Confidence score{" "}
                    <strong>{fmtPct(selectedDet.conf, 1)}</strong>{" "}
                    (threshold{" "}
                    <strong>{fmtPct(confThreshold, 1)}</strong>, diff{" "}
                    <strong>
                      {selectedDet.conf - confThreshold >= 0 ? "+" : ""}
                      {fmtNum((selectedDet.conf - confThreshold) * 100, 1)} pt
                    </strong>
                    ).
                  </span>
                ) : (
                  <span>
                    Güven skoru{" "}
                    <strong>{fmtPct(selectedDet.conf, 1)}</strong>{" "}
                    (aktif eşik{" "}
                    <strong>{fmtPct(confThreshold, 1)}</strong>, fark{" "}
                    <strong>
                      {selectedDet.conf - confThreshold >= 0 ? "+" : ""}
                      {fmtNum((selectedDet.conf - confThreshold) * 100, 1)}{" "}
                      puan
                    </strong>
                    ).
                  </span>
                )}
              </div>
            </div>

            {/* 2. Reliability (Val & Test yan yana veya tek bölme; n=0/null ise gizli) */}
            {reliabilitySummary && (
              <div
                data-testid="reliability-sentence"
                className="space-y-1.5 rounded-xl border border-[var(--border-hairline)] bg-[var(--bg-elevated)] p-3"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="text-xs font-medium text-[var(--text-secondary)]">
                    {lang === "en"
                      ? `Share of correct detections at confidence ${reliabilitySummary.binRange}`
                      : `${reliabilitySummary.binRange} güven aralığında doğru tespit oranı`}
                  </span>
                  {(reliabilitySummary.valPart?.lowSample ||
                    reliabilitySummary.testPart?.lowSample) && (
                    <Badge
                      variant="warning"
                      data-testid="reliability-low-sample-badge"
                    >
                      {t("explain.lowSampleBadge")}
                    </Badge>
                  )}
                </div>
                <p
                  className="font-mono text-xs font-semibold text-[var(--text-primary)] tabular-nums"
                  data-testid="reliability-headline"
                >
                  {reliabilitySummary.headlineText}
                </p>
                {reliabilitySummary.valPart && reliabilitySummary.testPart && (
                  <p
                    className="text-xs text-[var(--text-secondary)]"
                    data-testid="reliability-val-caveat"
                  >
                    {lang === "en"
                      ? "The threshold was chosen on the validation split, so its rate is optimistic; the test rate comes from unseen boards (2 boards)."
                      : "Eşik doğrulama (val) setinde seçildiği için val oranı iyimserdir; test oranı modelin hiç görmediği 2 karttan gelir."}
                  </p>
                )}
              </div>
            )}

            {/* 3. energy_in_box varsa yatay çubuk */}
            {energyInBox !== null && (
              <div
                data-testid="energy-in-box-row"
                className="space-y-1.5 rounded-xl border border-[var(--border-hairline)] bg-[var(--bg-elevated)] p-3"
              >
                <div className="flex items-center justify-between text-xs font-medium">
                  <span>
                    {t("explain.energyInBoxLabel", {
                      pct: fmtNum(energyInBox * 100, 1),
                    })}
                  </span>
                  <span className="font-mono text-[var(--accent-ink)] tabular-nums">
                    {fmtPct(energyInBox, 1)}
                  </span>
                </div>
                <div className="h-2 w-full overflow-hidden rounded-full bg-[var(--bg-subtle)]">
                  <div
                    className="h-full bg-[var(--accent)] transition-all"
                    style={{ width: `${Math.round(energyInBox * 100)}%` }}
                  />
                </div>
              </div>
            )}

            {/* 4. Resmi sınıf tanımı (docs/UI_V11_RULES.md) */}
            <div
              data-testid="explain-class-def"
              className="flex items-start gap-2.5 rounded-xl border border-[var(--border-hairline)] bg-[var(--bg-elevated)] p-3"
            >
              <span
                className="mt-1 h-2.5 w-2.5 shrink-0 rounded-full"
                style={{ backgroundColor: getClassColor(selectedDet.cls) }}
              />
              <div>
                <p className="text-xs font-semibold text-[var(--text-primary)]">
                  {t("explain.classDefLabel")}:{" "}
                  {getClassDisplayName(selectedDet.cls, lang)} (
                  <span className="font-mono">{selectedDet.cls}</span>)
                </p>
                <p className="mt-0.5 text-xs text-[var(--text-secondary)]">
                  {getClassDescription(selectedDet.cls, lang)}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader className="pb-2">
          <CardTitle>{t("explain.whatItShowsTitle")}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2.5 text-sm leading-relaxed text-[var(--text-secondary)]">
          <p>{t("explain.whatItShowsBody1")}</p>
          {layercamRow && (
            <>
              <p>
                {t("explain.whatItShowsMetrics", {
                  n: layercamRow.n_tespit,
                  imgsz: layercamRow.imgsz,
                  drop1x: fmtNum(layercamRow.goreli_dusus_harita_ort_1x ?? 0, 3),
                  ctrl1x: fmtNum(layercamRow.goreli_dusus_kontrol_ort_1x ?? 0, 3),
                  win1x: fmtNum((layercamRow.goreli_kazanma_orani_1x ?? 0) * 100, 1),
                  drop2x: fmtNum(layercamRow.goreli_dusus_harita_ort_2x ?? 0, 3),
                  ctrl2x: fmtNum(layercamRow.goreli_dusus_kontrol_ort_2x ?? 0, 3),
                  win2x: fmtNum((layercamRow.goreli_kazanma_orani_2x ?? 0) * 100, 1),
                  energy: fmtNum(layercamRow.enerji_kutuda_pencere * 100, 1),
                })}
              </p>
              <p className="text-xs text-[var(--text-muted)]">
                {t("explain.whatItShowsCaveat", {
                  pointing: fmtNum(layercamRow.kendi_kutu_tum_goruntu * 100, 1),
                  randPointing: fmtNum(layercamRow.pointing_rastgele_model * 100, 1),
                })}
              </p>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
