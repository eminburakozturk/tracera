import * as React from "react";
import {
  AlertTriangle,
  Camera,
  CheckCircle2,
  Download,
  FileJson,
  FileText,
  Flame,
  ImagePlus,
  Info,
  Play,
  RotateCcw,
  Upload,
  XCircle,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { motion, useReducedMotion } from "motion/react";
import { useTranslation } from "react-i18next";
import { VAL_EXAMPLES, type ValExampleItem } from "../fixtures";
import {
  downloadAnnotatedPng,
  downloadDetectJson,
} from "../lib/download";
import { getDownloadStats } from "../lib/downloadInfo";
import { cn } from "../lib/utils";
import {
  getClassColor,
  getClassDisplayName,
  isLowConfidenceReview,
  LOW_CONF_REVIEW_LABEL_EN,
  LOW_CONF_REVIEW_LABEL_TR,
  LOW_CONF_REVIEW_TOOLTIP_EN,
  LOW_CONF_REVIEW_TOOLTIP_TR,
  VERDICT_RULE_NOTE_EN,
  VERDICT_RULE_NOTE_TR,
} from "../lib/v11Rules";
import { useAppStore } from "../store/useAppStore";
import type { Detection } from "../types";
import { PrMiniChart } from "./PrMiniChart";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./ui/card";
import { Skeleton } from "./ui/skeleton";
import { Slider } from "./ui/slider";

interface DetectViewProps {
  onSelectExample: (example: ValExampleItem) => Promise<void>;
  onSelectFile: (file: File) => Promise<void>;
  onRunDetect: (confOverride?: number) => Promise<void>;
  onInspectDetection: (detId: number) => void;
  onExportPdf?: () => Promise<void>;
}

function useDefectCropUrl(
  imageUrl: string | undefined,
  box: [number, number, number, number],
  color: string,
): string | null {
  const [cropUrl, setCropUrl] = React.useState<string | null>(null);
  const [x1, y1, x2, y2] = box;

  React.useEffect(() => {
    if (!imageUrl || typeof document === "undefined") {
      setCropUrl(null);
      return;
    }
    let active = true;
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      if (!active) return;
      try {
        const bw = Math.max(8, x2 - x1);
        const bh = Math.max(8, y2 - y1);
        const pad = Math.max(24, Math.round(Math.max(bw, bh) * 0.45));
        const sx = Math.max(0, Math.floor(x1 - pad));
        const sy = Math.max(0, Math.floor(y1 - pad));
        const sw = Math.min(img.naturalWidth - sx, Math.ceil(bw + pad * 2));
        const sh = Math.min(img.naturalHeight - sy, Math.ceil(bh + pad * 2));
        if (sw <= 0 || sh <= 0) return;

        const canvas = document.createElement("canvas");
        canvas.width = 96;
        canvas.height = 96;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        ctx.drawImage(img, sx, sy, sw, sh, 0, 0, 96, 96);
        // Kırpım içinde kusur kutusunu çiz
        const rx = ((x1 - sx) / sw) * 96;
        const ry = ((y1 - sy) / sh) * 96;
        const rw = (bw / sw) * 96;
        const rh = (bh / sh) * 96;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(rx, ry, rw, rh);
        setCropUrl(canvas.toDataURL("image/jpeg", 0.85));
      } catch {
        setCropUrl(null);
      }
    };
    img.onerror = () => {
      if (active) setCropUrl(null);
    };
    img.src = imageUrl;
    return () => {
      active = false;
    };
  }, [imageUrl, x1, y1, x2, y2, color]);

  return cropUrl;
}

function DefectCardThumb({
  imageUrl,
  det,
  color,
  sharedLayoutId,
}: {
  imageUrl: string | undefined;
  det: Detection;
  color: string;
  sharedLayoutId?: string;
}) {
  const cropUrl = useDefectCropUrl(imageUrl, det.box, color);
  return (
    <motion.div
      layoutId={sharedLayoutId}
      className="relative flex h-12 w-12 shrink-0 items-center justify-center overflow-hidden rounded-lg border border-[var(--border-hairline)] bg-[var(--bg-canvas)]"
    >
      {cropUrl ? (
        <img
          src={cropUrl}
          alt={det.cls}
          className="h-full w-full object-cover"
        />
      ) : (
        <span
          className="h-3.5 w-3.5 rounded-full"
          style={{ backgroundColor: color }}
        />
      )}
    </motion.div>
  );
}

export function DetectView({
  onSelectExample,
  onSelectFile,
  onRunDetect,
  onInspectDetection,
  onExportPdf,
}: DetectViewProps) {
  const { t } = useTranslation();
  const prefersReducedMotion = useReducedMotion();
  const fileInputRef = React.useRef<HTMLInputElement>(null);
  const cameraInputRef = React.useRef<HTMLInputElement>(null);

  const lang = useAppStore((s) => s.lang);
  const meta = useAppStore((s) => s.meta);
  const backendChoice = useAppStore((s) => s.backendChoice);
  const selectedImage = useAppStore((s) => s.selectedImage);
  const confThreshold = useAppStore((s) => s.confThreshold);
  const setConfThreshold = useAppStore((s) => s.setConfThreshold);
  const classFilter = useAppStore((s) => s.classFilter);
  const setClassFilter = useAppStore((s) => s.setClassFilter);
  const detectResult = useAppStore((s) => s.detectResult);
  const detectLoading = useAppStore((s) => s.detectLoading);
  const detectError = useAppStore((s) => s.detectError);
  const selectedDetectionId = useAppStore((s) => s.selectedDetectionId);
  const setSelectedDetectionId = useAppStore((s) => s.setSelectedDetectionId);
  const ortProgress = useAppStore((s) => s.ortProgress);
  const ortModelCached = useAppStore((s) => s.ortModelCached);

  const [dragOver, setDragOver] = React.useState(false);
  const [hoveredId, setHoveredId] = React.useState<number | null>(null);
  const [zoom, setZoom] = React.useState(1);
  const [pan, setPan] = React.useState({ x: 0, y: 0 });
  const [panStart, setPanStart] = React.useState<{
    x: number;
    y: number;
  } | null>(null);

  React.useEffect(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, [selectedImage?.url]);

  const allDetections = React.useMemo(
    () =>
      (detectResult?.detections ?? []).filter(
        (d) => d.conf >= confThreshold,
      ),
    [detectResult, confThreshold],
  );
  const filteredDetections = React.useMemo(() => {
    if (classFilter === "all") return allDetections;
    return allDetections.filter((d) => d.cls === classFilter);
  }, [allDetections, classFilter]);

  const classCounts = React.useMemo(() => {
    const counts: Record<string, number> = {};
    for (const det of allDetections) {
      counts[det.cls] = (counts[det.cls] ?? 0) + 1;
    }
    return counts;
  }, [allDetections]);

  const selectedDet = React.useMemo(
    () =>
      filteredDetections.find((d) => d.id === selectedDetectionId) ??
      allDetections.find((d) => d.id === selectedDetectionId) ??
      null,
    [allDetections, filteredDetections, selectedDetectionId],
  );

  const isPass = allDetections.length === 0;

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      await onSelectFile(file);
    }
    e.target.value = "";
  };

  const handleDrop = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      await onSelectFile(file);
    }
  };

  const renderBox = (det: Detection) => {
    const color = meta.colors[det.cls_id] ?? getClassColor(det.cls);
    const [x1, y1, x2, y2] = det.box;
    const width = Math.max(1, x2 - x1);
    const height = Math.max(1, y2 - y1);
    const isSelected = selectedDetectionId === det.id;
    const isHovered = hoveredId === det.id;
    const imgWidth = detectResult?.width ?? 3056;
    const strokeW = Math.max(3, Math.round(imgWidth / 420));
    const fontSize = Math.max(18, Math.round(imgWidth / 75));
    const labelText = `${det.cls} ${(det.conf * 100).toFixed(1)}%`;
    const pillW = Math.max(120, labelText.length * fontSize * 0.58 + 16);
    const pillH = fontSize + 10;
    const labelY = y1 - pillH - 2 >= 0 ? y1 - pillH - 2 : y1 + 2;

    return (
      <g
        key={det.id}
        role="button"
        tabIndex={0}
        aria-label={`${det.cls} ${(det.conf * 100).toFixed(1)}%`}
        data-testid={`bbox-${det.id}`}
        style={{ pointerEvents: "all" }}
        className="cursor-pointer focus:outline-none"
        onClick={(e) => {
          e.stopPropagation();
          setSelectedDetectionId(det.id);
        }}
        onMouseEnter={() => setHoveredId(det.id)}
        onMouseLeave={() => setHoveredId(null)}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            setSelectedDetectionId(det.id);
          }
        }}
      >
        <rect
          x={x1}
          y={y1}
          width={width}
          height={height}
          fill={isSelected || isHovered ? `${color}33` : "rgba(0,0,0,0.001)"}
          stroke={color}
          strokeWidth={isSelected ? strokeW * 1.6 : strokeW}
        />
        <rect
          x={x1}
          y={labelY}
          width={pillW}
          height={pillH}
          rx={4}
          fill={color}
          opacity={isSelected || isHovered ? 1 : 0.9}
        />
        <text
          x={x1 + 8}
          y={labelY + fontSize + 1}
          fill="#06131f"
          fontSize={fontSize}
          fontWeight={700}
          fontFamily="JetBrains Mono, IBM Plex Mono, monospace"
        >
          {labelText}
        </text>
      </g>
    );
  };

  return (
    <div className="space-y-5">
      {/* GEÇTİ / KALDI Bandı (kural notu docs/UI_V11_RULES.md'den) */}
      {detectResult && (
        <div
          data-testid="verdict-banner"
          className={cn(
            "flex flex-wrap items-center justify-between gap-3 rounded-xl border px-4 py-3",
            isPass
              ? "border-emerald-600/40 bg-emerald-500/10 text-[var(--text-primary)]"
              : "border-rose-600/40 bg-rose-500/10 text-[var(--text-primary)]",
          )}
        >
          <div className="flex flex-wrap items-center gap-3">
            <Badge
              variant={isPass ? "success" : "danger"}
              data-testid="verdict-badge"
              className="px-3 py-1 font-mono text-xs font-bold tracking-wider"
            >
              {isPass ? (
                <CheckCircle2 className="h-3.5 w-3.5" />
              ) : (
                <XCircle className="h-3.5 w-3.5" />
              )}
              <span>
                {isPass ? t("detect.verdictPass") : t("detect.verdictFail")}
              </span>
            </Badge>
            <div>
              <p className="text-xs font-semibold text-[var(--text-primary)]">
                {isPass
                  ? t("detect.verdictPassDesc")
                  : t("detect.verdictFailDesc", { count: allDetections.length })}
              </p>
              <p
                data-testid="verdict-rule-note"
                className="text-[11px] text-[var(--text-secondary)]"
              >
                {lang === "en" ? VERDICT_RULE_NOTE_EN : VERDICT_RULE_NOTE_TR}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 font-mono text-xs text-[var(--text-secondary)] tabular-nums">
            <span>Conf ≥ {confThreshold.toFixed(3)}</span>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-12">
        <div className="space-y-4 lg:col-span-8">
          <Card>
            <CardHeader className="pb-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <CardTitle>{t("detect.title")}</CardTitle>
                  <CardDescription>{t("detect.dropHint")}</CardDescription>
                </div>
                {detectResult && (
                  <div className="flex flex-wrap items-center gap-1.5">
                    <Badge
                      variant="default"
                      className="font-mono tabular-nums"
                      data-testid="timing-badge"
                    >
                      {t("detect.modelTime")}:{" "}
                      {detectResult.timing.model_ms.toFixed(1)} ms ·{" "}
                      {t("detect.totalTime")}:{" "}
                      {detectResult.timing.total_ms.toFixed(1)} ms
                    </Badge>
                    <Badge variant="secondary" className="font-mono">
                      {detectResult.backend}
                    </Badge>
                  </div>
                )}
              </div>
            </CardHeader>

            <CardContent className="space-y-4">
              {backendChoice === "ort" &&
                ortModelCached === false &&
                ortProgress.stage === "idle" && (
                  <div
                    role="status"
                    data-testid="model-download-notice"
                    className="flex items-center gap-2 rounded-lg border border-sky-500/35 bg-sky-500/10 px-3 py-2 text-xs text-sky-900 dark:text-sky-200"
                  >
                    <Info className="h-4 w-4 shrink-0 text-[var(--accent)]" />
                    <span>
                      {t("detect.modelDownloadNotice", getDownloadStats())}
                    </span>
                  </div>
                )}

              {ortProgress.stage !== "idle" &&
                ortProgress.stage !== "ready" &&
                backendChoice === "ort" && (
                  <div
                    className="rounded-lg border border-[var(--accent)]/40 bg-[var(--accent-soft)] p-3"
                    role="status"
                    data-testid="ort-progress"
                  >
                    <div className="mb-1.5 flex items-center justify-between text-xs font-medium">
                      <span>{ortProgress.message}</span>
                      <span className="font-mono tabular-nums">
                        {Math.round(ortProgress.percent)}%
                      </span>
                    </div>
                    <div className="h-2 w-full overflow-hidden rounded-full bg-[var(--bg-subtle)]">
                      <div
                        className="h-full bg-[var(--accent)] transition-all"
                        style={{
                          width: `${Math.min(100, Math.max(0, ortProgress.percent))}%`,
                        }}
                      />
                    </div>
                  </div>
                )}

              {detectError && (
                <div
                  role="alert"
                  data-testid="detect-error-banner"
                  className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-rose-500/40 bg-rose-500/10 p-3 text-sm text-rose-700 dark:text-rose-200"
                >
                  <div className="flex items-start gap-2">
                    <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                    <span>{detectError}</span>
                  </div>
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    onClick={() => onRunDetect()}
                  >
                    {t("controls.retry")}
                  </Button>
                </div>
              )}

              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragOver(true);
                }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
                className={cn(
                  "flex flex-wrap items-center justify-between gap-3 rounded-xl border border-dashed p-3.5 transition-colors",
                  dragOver
                    ? "border-[var(--accent)] bg-[var(--accent-soft)]"
                    : "border-[var(--border-strong)] bg-[var(--bg-elevated)]",
                )}
              >
                <div className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[var(--bg-surface)] text-[var(--accent-ink)]">
                    <Upload className="h-4 w-4" />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-[var(--text-primary)]">
                      {selectedImage ? selectedImage.name : t("detect.dropTitle")}
                    </p>
                    <p className="text-xs text-[var(--text-muted)]">
                      {t("detect.dropHint")}
                    </p>
                  </div>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={handleFileChange}
                    data-testid="file-upload-input"
                  />
                  <input
                    ref={cameraInputRef}
                    type="file"
                    accept="image/*"
                    capture="environment"
                    className="hidden"
                    onChange={handleFileChange}
                    data-testid="camera-upload-input"
                  />
                  <Button
                    type="button"
                    size="sm"
                    variant="secondary"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <ImagePlus className="h-3.5 w-3.5" />
                    <span>{t("detect.pickFile")}</span>
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    variant="secondary"
                    onClick={() => cameraInputRef.current?.click()}
                  >
                    <Camera className="h-3.5 w-3.5" />
                    <span>{t("detect.cameraCapture")}</span>
                  </Button>
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="text-xs font-semibold uppercase tracking-wider text-[var(--text-secondary)]">
                    {t("detect.examplesTitle")}
                  </span>
                  <span className="text-xs text-[var(--text-muted)]">
                    {t("detect.examplesNote")}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-4">
                  {VAL_EXAMPLES.map((ex) => {
                    const active = selectedImage?.name === ex.filename;
                    const clsIdx = meta.classes.indexOf(ex.defectClass);
                    const dotColor =
                      clsIdx >= 0 ? meta.colors[clsIdx] : "#38bdf8";
                    return (
                      <button
                        key={ex.id}
                        type="button"
                        data-testid={`example-${ex.id}`}
                        onClick={() => onSelectExample(ex)}
                        className={cn(
                          "group flex flex-col overflow-hidden rounded-lg border text-left transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] cursor-pointer",
                          active
                            ? "border-[var(--accent)] ring-1 ring-[var(--accent)]"
                            : "border-[var(--border-hairline)] bg-[var(--bg-elevated)] hover:border-[var(--border-strong)]",
                        )}
                      >
                        <img
                          src={ex.url}
                          alt={ex.filename}
                          className="h-16 w-full object-cover"
                          loading="lazy"
                        />
                        <div className="flex items-center gap-1.5 px-2 py-1.5 text-[11px]">
                          <span
                            className="h-2 w-2 shrink-0 rounded-full"
                            style={{ backgroundColor: dotColor }}
                          />
                          <span className="truncate font-mono text-[var(--text-primary)]">
                            {ex.board}_{ex.defectClass}
                          </span>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-1.5">
                    <Button
                      type="button"
                      size="sm"
                      variant="secondary"
                      onClick={() =>
                        setZoom((z) => Math.min(6, +(z * 1.25).toFixed(2)))
                      }
                      aria-label={t("detect.zoomIn")}
                    >
                      <ZoomIn className="h-3.5 w-3.5" />
                    </Button>
                    <Button
                      type="button"
                      size="sm"
                      variant="secondary"
                      onClick={() =>
                        setZoom((z) => Math.max(1, +(z / 1.25).toFixed(2)))
                      }
                      aria-label={t("detect.zoomOut")}
                    >
                      <ZoomOut className="h-3.5 w-3.5" />
                    </Button>
                    <Button
                      type="button"
                      size="sm"
                      variant="secondary"
                      onClick={() => {
                        setZoom(1);
                        setPan({ x: 0, y: 0 });
                      }}
                      aria-label={t("detect.zoomReset")}
                    >
                      <RotateCcw className="h-3.5 w-3.5" />
                    </Button>
                    <span className="ml-1 font-mono text-xs text-[var(--text-muted)] tabular-nums">
                      {Math.round(zoom * 100)}%
                    </span>
                  </div>

                  {detectResult && selectedImage && (
                    <div className="flex flex-wrap items-center gap-2">
                      {onExportPdf && (
                        <Button
                          type="button"
                          size="sm"
                          variant="default"
                          data-testid="download-pdf-btn"
                          onClick={() => void onExportPdf()}
                        >
                          <FileText className="h-3.5 w-3.5" />
                          <span>{t("detect.downloadReportPdf")}</span>
                        </Button>
                      )}
                      <Button
                        type="button"
                        size="sm"
                        variant="secondary"
                        data-testid="download-json-btn"
                        onClick={() =>
                          downloadDetectJson(
                            detectResult,
                            selectedImage.name.replace(/\.[^.]+$/, ""),
                          )
                        }
                      >
                        <FileJson className="h-3.5 w-3.5" />
                        <span>{t("detect.downloadJson")}</span>
                      </Button>
                      <Button
                        type="button"
                        size="sm"
                        variant="secondary"
                        data-testid="download-png-btn"
                        onClick={() =>
                          downloadAnnotatedPng(
                            selectedImage.url,
                            detectResult,
                            meta,
                            selectedImage.name.replace(/\.[^.]+$/, ""),
                          )
                        }
                      >
                        <Download className="h-3.5 w-3.5" />
                        <span>{t("detect.downloadPng")}</span>
                      </Button>
                    </div>
                  )}
                </div>

                {/* Seçili kusur için ortak öğe (shared layoutId) önizleme çubuğu */}
                {selectedDet && selectedImage && (
                  <div
                    data-testid="selected-defect-spotlight"
                    className="flex items-center justify-between gap-3 rounded-xl border border-[var(--accent)]/40 bg-[var(--accent-soft)] px-3 py-2"
                  >
                    <div className="flex items-center gap-3">
                      <DefectCardThumb
                        imageUrl={selectedImage.url}
                        det={selectedDet}
                        color={
                          meta.colors[selectedDet.cls_id] ??
                          getClassColor(selectedDet.cls)
                        }
                        sharedLayoutId={
                          !prefersReducedMotion
                            ? `defect-shared-${selectedDet.id}`
                            : undefined
                        }
                      />
                      <div>
                        <div className="flex flex-wrap items-center gap-2">
                          <p className="text-xs font-semibold text-[var(--text-primary)]">
                            #{selectedDet.id} ·{" "}
                            {getClassDisplayName(selectedDet.cls, lang)} (
                            {(selectedDet.conf * 100).toFixed(1)}%)
                          </p>
                          {isLowConfidenceReview(
                            selectedDet.conf,
                            confThreshold,
                          ) && (
                            <Badge
                              variant="warning"
                              data-testid="selected-low-conf-badge"
                              title={
                                lang === "en"
                                  ? LOW_CONF_REVIEW_TOOLTIP_EN
                                  : LOW_CONF_REVIEW_TOOLTIP_TR
                              }
                            >
                              <AlertTriangle className="h-3 w-3 shrink-0" />
                              <span>
                                {lang === "en"
                                  ? LOW_CONF_REVIEW_LABEL_EN
                                  : LOW_CONF_REVIEW_LABEL_TR}
                              </span>
                            </Badge>
                          )}
                        </div>
                        <p className="font-mono text-[11px] text-[var(--text-secondary)] tabular-nums">
                          [{selectedDet.box.map((v) => Math.round(v)).join(", ")}]
                        </p>
                      </div>
                    </div>
                    <Button
                      type="button"
                      size="sm"
                      variant="secondary"
                      onClick={() => onInspectDetection(selectedDet.id)}
                    >
                      <Flame className="h-3.5 w-3.5 text-[var(--accent-ink)]" />
                      <span>{t("detect.inspectHeatmap")}</span>
                    </Button>
                  </div>
                )}

                <div
                  className="relative min-h-[260px] overflow-hidden rounded-xl border border-[var(--border-hairline)] bg-[var(--bg-canvas)]"
                  onMouseDown={(e) => {
                    if (zoom <= 1) return;
                    setPanStart({
                      x: e.clientX - pan.x,
                      y: e.clientY - pan.y,
                    });
                  }}
                  onMouseMove={(e) => {
                    if (!panStart) return;
                    setPan({
                      x: e.clientX - panStart.x,
                      y: e.clientY - panStart.y,
                    });
                  }}
                  onMouseUp={() => setPanStart(null)}
                  onMouseLeave={() => setPanStart(null)}
                >
                  {detectLoading && (
                    <div className="space-y-3 p-6">
                      <Skeleton className="h-64 w-full" />
                    </div>
                  )}

                  {!detectLoading && !selectedImage && (
                    <div className="flex h-64 items-center justify-center p-6 text-center text-sm text-[var(--text-muted)]">
                      {t("detect.noImage")}
                    </div>
                  )}

                  {!detectLoading && selectedImage && (
                    <div
                      className="relative mx-auto w-full select-none"
                      style={{
                        transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                        transformOrigin: "center center",
                        transition: panStart
                          ? "none"
                          : "transform 120ms ease-out",
                      }}
                    >
                      <img
                        src={selectedImage.url}
                        alt={selectedImage.name}
                        className="pointer-events-none block h-auto w-full"
                        draggable={false}
                      />
                      {detectResult && (
                        <svg
                          viewBox={`0 0 ${detectResult.width} ${detectResult.height}`}
                          className="pointer-events-none absolute inset-0 h-full w-full"
                          preserveAspectRatio="none"
                          data-testid="bbox-svg"
                        >
                          {filteredDetections.map(renderBox)}
                        </svg>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="space-y-4 lg:col-span-4">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle>{t("detect.confLabel")}</CardTitle>
              <CardDescription>{t("detect.confLockedNote")}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span
                    className="font-mono text-lg font-semibold text-[var(--accent-ink)] tabular-nums"
                    data-testid="conf-value"
                  >
                    {confThreshold.toFixed(3)}
                  </span>
                  {Math.abs(confThreshold - meta.conf) > 0.001 && (
                    <Button
                      type="button"
                      size="sm"
                      variant="ghost"
                      onClick={() => {
                        setConfThreshold(meta.conf);
                        if (selectedImage) {
                          void onRunDetect(meta.conf);
                        }
                      }}
                    >
                      {t("detect.resetLocked")} ({meta.conf.toFixed(3)})
                    </Button>
                  )}
                </div>
                <Slider
                  min={0.05}
                  max={0.95}
                  step={0.005}
                  value={[confThreshold]}
                  onValueChange={([v]) => setConfThreshold(v)}
                  onValueCommit={([v]) => {
                    if (selectedImage) {
                      void onRunDetect(v);
                    }
                  }}
                  aria-label={t("detect.confLabel")}
                />
              </div>

              <PrMiniChart confThreshold={confThreshold} />

              <Button
                type="button"
                className="w-full"
                disabled={!selectedImage || detectLoading}
                onClick={() => onRunDetect()}
                data-testid="run-detect-btn"
              >
                <Play className="h-4 w-4" />
                <span>
                  {detectLoading ? t("detect.running") : t("detect.runDetect")}
                </span>
              </Button>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle>{t("detect.detectionsTitle")}</CardTitle>
                <Badge
                  variant="default"
                  className="font-mono tabular-nums"
                  data-testid="det-count-badge"
                >
                  {filteredDetections.length}
                </Badge>
              </div>
              <CardDescription>
                {t("detect.selectHint")}
                {backendChoice === "ort" && ` ${t("explain.apiRequiredNote")}`}
              </CardDescription>
            </CardHeader>

            <CardContent className="space-y-3">
              <div
                className="flex flex-wrap gap-1.5"
                role="group"
                aria-label={t("detect.filterLabel")}
              >
                <button
                  type="button"
                  onClick={() => setClassFilter("all")}
                  className={cn(
                    "rounded-full border px-2.5 py-1 text-xs font-medium transition-colors cursor-pointer",
                    classFilter === "all"
                      ? "border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--accent-ink)]"
                      : "border-[var(--border-hairline)] bg-[var(--bg-elevated)] text-[var(--text-secondary)]",
                  )}
                >
                  {t("detect.filterAll")} ({allDetections.length})
                </button>
                {meta.classes.map((cls, idx) => {
                  const count = classCounts[cls] ?? 0;
                  if (count === 0 && classFilter !== cls) return null;
                  const color = meta.colors[idx] ?? getClassColor(cls);
                  return (
                    <button
                      key={cls}
                      type="button"
                      onClick={() => setClassFilter(cls)}
                      className={cn(
                        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium transition-colors cursor-pointer",
                        classFilter === cls
                          ? "border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--text-primary)]"
                          : "border-[var(--border-hairline)] bg-[var(--bg-elevated)] text-[var(--text-secondary)]",
                      )}
                    >
                      <span
                        className="h-2 w-2 rounded-full"
                        style={{ backgroundColor: color }}
                      />
                      <span>{getClassDisplayName(cls, lang, true)}</span>
                      <span className="font-mono opacity-80">({count})</span>
                    </button>
                  );
                })}
              </div>

              {filteredDetections.length === 0 ? (
                <p className="rounded-lg border border-[var(--border-hairline)] bg-[var(--bg-elevated)] p-3 text-xs text-[var(--text-muted)]">
                  {t("detect.noDetections")}
                </p>
              ) : (
                <div className="max-h-[560px] space-y-2 overflow-y-auto pr-1">
                  {filteredDetections.map((det) => {
                    const color =
                      meta.colors[det.cls_id] ?? getClassColor(det.cls);
                    const selected = selectedDetectionId === det.id;
                    return (
                      <div
                        key={det.id}
                        data-testid={`det-card-${det.id}`}
                        onMouseEnter={() => setHoveredId(det.id)}
                        onMouseLeave={() => setHoveredId(null)}
                        onClick={() => setSelectedDetectionId(det.id)}
                        className={cn(
                          "rounded-lg border p-3 transition-all cursor-pointer",
                          selected
                            ? "border-[var(--accent)] bg-[var(--accent-soft)]"
                            : "border-[var(--border-hairline)] bg-[var(--bg-elevated)] hover:border-[var(--border-strong)]",
                        )}
                      >
                        <div className="flex items-start gap-3">
                          {!selected && (
                            <DefectCardThumb
                              imageUrl={selectedImage?.url}
                              det={det}
                              color={color}
                              sharedLayoutId={
                                !prefersReducedMotion
                                  ? `defect-shared-${det.id}`
                                  : undefined
                              }
                            />
                          )}
                          <div className="min-w-0 flex-1">
                            <div className="flex items-center justify-between gap-2">
                              <div className="flex min-w-0 items-center gap-2">
                                <span
                                  className="h-2.5 w-2.5 shrink-0 rounded-full"
                                  style={{ backgroundColor: color }}
                                />
                                <span className="truncate text-sm font-semibold text-[var(--text-primary)]">
                                  {getClassDisplayName(det.cls, lang, true)}
                                </span>
                                <span className="truncate font-mono text-[11px] text-[var(--text-muted)]">
                                  ({det.cls})
                                </span>
                              </div>
                              <span className="font-mono text-xs font-semibold text-[var(--accent-ink)] tabular-nums">
                                {(det.conf * 100).toFixed(1)}%
                              </span>
                            </div>
                            <p className="mt-1 font-mono text-[11px] text-[var(--text-muted)] tabular-nums">
                              [{det.box.map((v) => Math.round(v)).join(", ")}]
                            </p>
                            {isLowConfidenceReview(det.conf, confThreshold) && (
                              <div className="mt-1.5">
                                <Badge
                                  variant="warning"
                                  data-testid={`low-conf-badge-${det.id}`}
                                  title={
                                    lang === "en"
                                      ? LOW_CONF_REVIEW_TOOLTIP_EN
                                      : LOW_CONF_REVIEW_TOOLTIP_TR
                                  }
                                >
                                  <AlertTriangle className="h-3 w-3 shrink-0" />
                                  <span>
                                    {lang === "en"
                                      ? LOW_CONF_REVIEW_LABEL_EN
                                      : LOW_CONF_REVIEW_LABEL_TR}
                                  </span>
                                </Badge>
                              </div>
                            )}
                          </div>
                        </div>
                        <div className="mt-2 flex justify-end">
                          <Button
                            type="button"
                            size="sm"
                            variant="secondary"
                            data-testid={`explain-det-${det.id}`}
                            onClick={(e) => {
                              e.stopPropagation();
                              onInspectDetection(det.id);
                            }}
                          >
                            <Flame className="h-3.5 w-3.5 text-[var(--accent-ink)]" />
                            <span>{t("detect.inspectHeatmap")}</span>
                          </Button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
