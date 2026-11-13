import * as React from "react";
import {
  AlertTriangle,
  Camera,
  Download,
  FileJson,
  Flame,
  ImagePlus,
  Play,
  RotateCcw,
  Upload,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import { VAL_EXAMPLES, type ValExampleItem } from "../fixtures";
import {
  downloadAnnotatedPng,
  downloadDetectJson,
} from "../lib/download";
import { cn } from "../lib/utils";
import { useAppStore } from "../store/useAppStore";
import type { Detection } from "../types";
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
}

export function DetectView({
  onSelectExample,
  onSelectFile,
  onRunDetect,
  onInspectDetection,
}: DetectViewProps) {
  const { t } = useTranslation();
  const fileInputRef = React.useRef<HTMLInputElement>(null);
  const cameraInputRef = React.useRef<HTMLInputElement>(null);

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

  const [dragOver, setDragOver] = React.useState(false);
  const [hoveredId, setHoveredId] = React.useState<number | null>(null);
  const [zoom, setZoom] = React.useState(1);
  const [pan, setPan] = React.useState({ x: 0, y: 0 });
  const [panStart, setPanStart] = React.useState<{ x: number; y: number } | null>(
    null,
  );

  React.useEffect(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, [selectedImage?.url]);

  const allDetections = React.useMemo(
    () => detectResult?.detections ?? [],
    [detectResult],
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
    const color = meta.colors[det.cls_id] ?? "#38bdf8";
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
          fontFamily="IBM Plex Mono, monospace"
        >
          {labelText}
        </text>
      </g>
    );
  };

  return (
    <div className="space-y-5">
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
                    <Badge variant="default" className="font-mono tabular-nums" data-testid="timing-badge">
                      {t("detect.modelTime")}: {detectResult.timing.model_ms.toFixed(1)} ms ·{" "}
                      {t("detect.totalTime")}: {detectResult.timing.total_ms.toFixed(1)} ms
                    </Badge>
                    <Badge variant="secondary" className="font-mono">
                      {detectResult.backend}
                    </Badge>
                  </div>
                )}
              </div>
            </CardHeader>

            <CardContent className="space-y-4">
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
                        style={{ width: `${Math.min(100, Math.max(0, ortProgress.percent))}%` }}
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
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[var(--bg-surface)] text-[var(--accent)]">
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
                      onClick={() => setZoom((z) => Math.min(6, +(z * 1.25).toFixed(2)))}
                      aria-label={t("detect.zoomIn")}
                    >
                      <ZoomIn className="h-3.5 w-3.5" />
                    </Button>
                    <Button
                      type="button"
                      size="sm"
                      variant="secondary"
                      onClick={() => setZoom((z) => Math.max(1, +(z / 1.25).toFixed(2)))}
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

                <div
                  className="relative min-h-[260px] overflow-hidden rounded-xl border border-[var(--border-hairline)] bg-[var(--bg-canvas)]"
                  onMouseDown={(e) => {
                    if (zoom <= 1) return;
                    setPanStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
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
                        transition: panStart ? "none" : "transform 120ms ease-out",
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
                  <span className="font-mono text-lg font-semibold text-[var(--accent)] tabular-nums" data-testid="conf-value">
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
                <Badge variant="default" className="font-mono tabular-nums" data-testid="det-count-badge">
                  {filteredDetections.length}
                </Badge>
              </div>
              <CardDescription>
                {t("detect.selectHint")}
                {backendChoice === "ort" && ` ${t("explain.apiRequiredNote")}`}
              </CardDescription>
            </CardHeader>

            <CardContent className="space-y-3">
              <div className="flex flex-wrap gap-1.5" role="group" aria-label={t("detect.filterLabel")}>
                <button
                  type="button"
                  onClick={() => setClassFilter("all")}
                  className={cn(
                    "rounded-full border px-2.5 py-1 text-xs font-medium transition-colors cursor-pointer",
                    classFilter === "all"
                      ? "border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--accent)]"
                      : "border-[var(--border-hairline)] bg-[var(--bg-elevated)] text-[var(--text-secondary)]",
                  )}
                >
                  {t("detect.filterAll")} ({allDetections.length})
                </button>
                {meta.classes.map((cls, idx) => {
                  const count = classCounts[cls] ?? 0;
                  if (count === 0 && classFilter !== cls) return null;
                  const color = meta.colors[idx] ?? "#38bdf8";
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
                      <span>{cls}</span>
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
                    const color = meta.colors[det.cls_id] ?? "#38bdf8";
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
                        <div className="flex items-center justify-between gap-2">
                          <div className="flex items-center gap-2">
                            <span
                              className="h-2.5 w-2.5 rounded-full"
                              style={{ backgroundColor: color }}
                            />
                            <span className="text-sm font-semibold text-[var(--text-primary)]">
                              {det.cls}
                            </span>
                          </div>
                          <span className="font-mono text-xs font-semibold text-[var(--accent)] tabular-nums">
                            {(det.conf * 100).toFixed(1)}%
                          </span>
                        </div>
                        <p className="mt-1 font-mono text-[11px] text-[var(--text-muted)] tabular-nums">
                          [{det.box.map((v) => Math.round(v)).join(", ")}]
                        </p>
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
                            <Flame className="h-3.5 w-3.5 text-[var(--accent)]" />
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
