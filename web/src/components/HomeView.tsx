import * as React from "react";
import {
  ArrowRight,
  Camera,
  CheckCircle2,
  Clock,
  Cpu,
  ImagePlus,
  Layers,
  ShieldCheck,
  Target,
  Upload,
  XCircle,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import { VAL_EXAMPLES, type ValExampleItem } from "../fixtures";
import { fmtMeanStd, fmtMs, fmtNum } from "../lib/format";
import { listHistoryRecords } from "../lib/historyDb";
import { selectOpenVinoInt8Row } from "../lib/metrics";
import { cn } from "../lib/utils";
import { getClassColor, getClassDisplayName } from "../lib/v11Rules";
import { useAppStore } from "../store/useAppStore";
import type { HistoryRecord } from "../types";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./ui/card";

interface HomeViewProps {
  onSelectExample: (example: ValExampleItem) => Promise<void>;
  onSelectFile: (file: File) => Promise<void>;
  onOpenHistoryRecord?: (record: HistoryRecord) => void;
  onRunDetect?: () => Promise<void>;
}

export function HomeView({
  onSelectExample,
  onSelectFile,
  onOpenHistoryRecord,
  onRunDetect,
}: HomeViewProps) {
  const { t } = useTranslation();
  const fileInputRef = React.useRef<HTMLInputElement>(null);
  const cameraInputRef = React.useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = React.useState(false);
  const [recentRecords, setRecentRecords] = React.useState<HistoryRecord[]>([]);

  const lang = useAppStore((s) => s.lang);
  const metrics = useAppStore((s) => s.metrics);
  const setActiveTab = useAppStore((s) => s.setActiveTab);
  const historyVersion = useAppStore((s) => s.historyVersion);

  React.useEffect(() => {
    let active = true;
    void listHistoryRecords().then((records) => {
      if (active) {
        setRecentRecords(records.slice(0, 4));
      }
    });
    return () => {
      active = false;
    };
  }, [historyVersion]);

  const cv1024 = metrics?.cv.resolution_summary["1024_last"];
  const opTest = metrics?.operating_point.find((r) => r.split === "test");
  const ovInt8Row = selectOpenVinoInt8Row(metrics?.edge_benchmark);
  const webGpuRow = metrics?.web_spike?.find(
    (r) =>
      r.backend.toLowerCase().includes("webgpu") &&
      r.model === "FP32 ONNX" &&
      r.medyan_ms !== null,
  );

  const handleFilePick = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setActiveTab("detect");
      await onSelectFile(file);
    }
    e.target.value = "";
  };

  const handleDrop = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      setActiveTab("detect");
      await onSelectFile(file);
    }
  };

  return (
    <div className="space-y-6" data-testid="home-view">
      {/* Hero & Hızlı Yükleme */}
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-12">
        <Card className="lg:col-span-7">
          <CardHeader>
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="default" className="font-mono">
                YOLO11s · 1024×1024 · LayerCAM L4
              </Badge>
              <Badge variant="secondary" className="font-mono">
                {lang === "en" ? "Conf" : "Eşik"} = {fmtNum(0.395, 3)}
              </Badge>
            </div>
            <CardTitle className="mt-2 text-xl sm:text-2xl">
              {t("home.heroTitle")}
            </CardTitle>
            <CardDescription className="text-sm leading-relaxed text-[var(--text-secondary)]">
              {t("home.heroSubtitle")}
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-wrap items-center gap-3">
            <Button
              type="button"
              onClick={() => {
                setActiveTab("detect");
                void onRunDetect?.();
              }}
              data-testid="run-detect-btn"
            >
              <span>{t("home.openInspect")}</span>
              <ArrowRight className="h-4 w-4" />
            </Button>
            <Button
              type="button"
              variant="secondary"
              onClick={() => setActiveTab("batch")}
              data-testid="home-go-batch-btn"
            >
              <Layers className="h-4 w-4" />
              <span>{t("home.openBatch")}</span>
            </Button>
          </CardContent>
        </Card>

        <Card className="lg:col-span-5">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">{t("home.quickUploadTitle")}</CardTitle>
            <CardDescription>{t("home.quickUploadHint")}</CardDescription>
          </CardHeader>
          <CardContent>
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              data-testid="home-dropzone"
              className={cn(
                "flex flex-col items-center justify-center gap-3 rounded-xl border border-dashed p-5 text-center transition-colors",
                dragOver
                  ? "border-[var(--accent)] bg-[var(--accent-soft)]"
                  : "border-[var(--border-strong)] bg-[var(--bg-elevated)]",
              )}
            >
              <div className="flex h-11 w-11 items-center justify-center rounded-xl border border-[var(--accent)]/35 bg-[var(--accent-soft)] text-[var(--accent-ink)]">
                <Upload className="h-5 w-5" />
              </div>
              <p className="text-xs font-medium text-[var(--text-primary)]">
                {t("detect.dropTitle")}
              </p>
              <div className="flex flex-wrap items-center justify-center gap-2">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={handleFilePick}
                  data-testid="home-file-input"
                />
                <input
                  ref={cameraInputRef}
                  type="file"
                  accept="image/*"
                  capture="environment"
                  className="hidden"
                  onChange={handleFilePick}
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
          </CardContent>
        </Card>
      </div>

      {/* 3 Metrik Kartı (metrics.json kaynağından) */}
      <div
        className="grid grid-cols-1 gap-4 md:grid-cols-3"
        data-testid="home-metrics-grid"
      >
        <Card>
          <CardContent className="flex items-start justify-between gap-3 pt-5">
            <div>
              <p className="text-xs font-medium text-[var(--text-secondary)]">
                {t("home.metricCvTitle")}
              </p>
              <p
                className="mt-1.5 font-mono text-2xl font-bold text-[var(--accent-ink)] tabular-nums"
                data-testid="home-metric-cv"
              >
                {cv1024 ? fmtMeanStd(cv1024.mAP50_mean, cv1024.mAP50_std) : "—"}
              </p>
              <p className="mt-1 font-mono text-xs text-[var(--text-muted)] tabular-nums">
                mAP50-95:{" "}
                {cv1024
                  ? fmtMeanStd(cv1024.mAP50_95_mean, cv1024.mAP50_95_std)
                  : "—"}
              </p>
              <p className="mt-1 text-[11px] text-[var(--text-muted)]">
                {t("home.metricCvSub")}
              </p>
            </div>
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-[var(--accent)]/30 bg-[var(--accent-soft)] text-[var(--accent-ink)]">
              <ShieldCheck className="h-5 w-5" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="flex items-start justify-between gap-3 pt-5">
            <div>
              <p className="text-xs font-medium text-[var(--text-secondary)]">
                {t("home.metricTestTitle")}
              </p>
              <p
                className="mt-1.5 font-mono text-2xl font-bold text-[var(--text-primary)] tabular-nums"
                data-testid="home-metric-test"
              >
                {opTest ? fmtNum(opTest.F1, 3) : "—"}
              </p>
              <p className="mt-1 font-mono text-xs text-[var(--text-muted)] tabular-nums">
                P: {opTest ? fmtNum(opTest.P, 3) : "—"} · R:{" "}
                {opTest ? fmtNum(opTest.R, 3) : "—"}
              </p>
              <p className="mt-1 text-[11px] text-[var(--text-muted)]">
                {t("home.metricTestSub")}
              </p>
            </div>
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-[var(--accent)]/30 bg-[var(--accent-soft)] text-[var(--accent-ink)]">
              <Target className="h-5 w-5" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="flex items-start justify-between gap-3 pt-5">
            <div>
              <p className="text-xs font-medium text-[var(--text-secondary)]">
                {t("home.metricSpeedTitle")}
              </p>
              <p
                className="mt-1.5 font-mono text-2xl font-bold text-[var(--text-primary)] tabular-nums"
                data-testid="home-metric-speed"
              >
                {ovInt8Row ? fmtMs(ovInt8Row.gecikme_inf_medyan_ms) : "—"}
              </p>
              <p className="mt-1 font-mono text-xs text-[var(--text-muted)] tabular-nums">
                {ovInt8Row
                  ? `${lang === "en" ? "Total incl. pre/post" : "Ön/son işlem dahil"}: ${fmtMs(ovInt8Row.gecikme_toplam_medyan_ms)}`
                  : "—"}
                {webGpuRow && webGpuRow.medyan_ms !== null
                  ? ` · WebGPU (${lang === "en" ? "browser" : "tarayıcı"}): ${fmtMs(webGpuRow.medyan_ms)}`
                  : ""}
              </p>
              <p className="mt-1 text-[11px] text-[var(--text-muted)]">
                {t("home.metricSpeedSub")}
              </p>
            </div>
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-[var(--accent)]/30 bg-[var(--accent-soft)] text-[var(--accent-ink)]">
              <Cpu className="h-5 w-5" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Val Örnek Galerisi (Kart 05 & 11) + Son Muayeneler */}
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-12">
        <Card className="lg:col-span-7">
          <CardHeader className="pb-3">
            <CardTitle>{t("home.valGalleryTitle")}</CardTitle>
            <CardDescription>{t("home.valGallerySub")}</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              {VAL_EXAMPLES.map((ex) => {
                const dotColor = getClassColor(ex.defectClass);
                return (
                  <button
                    key={ex.id}
                    type="button"
                    data-testid={`home-example-${ex.id}`}
                    onClick={async () => {
                      setActiveTab("detect");
                      await onSelectExample(ex);
                    }}
                    className="group flex flex-col overflow-hidden rounded-xl border border-[var(--border-hairline)] bg-[var(--bg-elevated)] text-left transition-all hover:border-[var(--accent)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] cursor-pointer"
                  >
                    <img
                      src={ex.url}
                      alt={ex.filename}
                      className="h-24 w-full object-cover transition-transform duration-200 group-hover:scale-105"
                      loading="lazy"
                    />
                    <div className="space-y-0.5 p-2.5">
                      <div className="flex items-center gap-1.5">
                        <span
                          className="h-2 w-2 shrink-0 rounded-full"
                          style={{ backgroundColor: dotColor }}
                        />
                        <span className="truncate text-xs font-semibold text-[var(--text-primary)]">
                          {getClassDisplayName(ex.defectClass, lang, true)}
                        </span>
                      </div>
                      <p className="truncate font-mono text-[11px] text-[var(--text-muted)]">
                        Kart {ex.board} · {ex.filename}
                      </p>
                    </div>
                  </button>
                );
              })}
            </div>
          </CardContent>
        </Card>

        <Card className="lg:col-span-5">
          <CardHeader className="flex flex-row items-center justify-between pb-3">
            <div>
              <CardTitle>{t("home.recentTitle")}</CardTitle>
            </div>
            <Button
              type="button"
              size="sm"
              variant="ghost"
              onClick={() => setActiveTab("history")}
              data-testid="home-view-all-history"
            >
              <Clock className="h-3.5 w-3.5" />
              <span>{t("home.viewAllHistory")}</span>
            </Button>
          </CardHeader>
          <CardContent>
            {recentRecords.length === 0 ? (
              <div
                data-testid="home-recent-empty"
                className="rounded-xl border border-[var(--border-hairline)] bg-[var(--bg-elevated)] p-4 text-xs text-[var(--text-muted)]"
              >
                {t("home.recentEmpty")}
              </div>
            ) : (
              <div className="space-y-2.5" data-testid="home-recent-list">
                {recentRecords.map((rec) => (
                  <button
                    key={rec.id}
                    type="button"
                    onClick={() => {
                      if (onOpenHistoryRecord) {
                        onOpenHistoryRecord(rec);
                      } else {
                        setActiveTab("history");
                      }
                    }}
                    className="flex w-full items-center justify-between gap-3 rounded-xl border border-[var(--border-hairline)] bg-[var(--bg-elevated)] p-2.5 text-left transition-colors hover:border-[var(--accent)] cursor-pointer"
                  >
                    <div className="flex min-w-0 items-center gap-3">
                      <img
                        src={rec.thumbnailDataUrl}
                        alt={rec.filename}
                        className="h-10 w-12 shrink-0 rounded-md border border-[var(--border-hairline)] object-cover"
                      />
                      <div className="min-w-0">
                        <p className="truncate font-mono text-xs font-semibold text-[var(--text-primary)]">
                          {rec.filename}
                        </p>
                        <p className="font-mono text-[11px] text-[var(--text-muted)] tabular-nums">
                          {new Date(rec.createdAt).toLocaleTimeString(
                            lang === "en" ? "en-US" : "tr-TR",
                            { hour: "2-digit", minute: "2-digit" },
                          )}{" "}
                          · {rec.defectCount} kusur
                        </p>
                      </div>
                    </div>
                    <Badge
                      variant={rec.verdict === "PASS" ? "success" : "danger"}
                      className="shrink-0 font-mono"
                    >
                      {rec.verdict === "PASS" ? (
                        <CheckCircle2 className="h-3 w-3" />
                      ) : (
                        <XCircle className="h-3 w-3" />
                      )}
                      <span>
                        {rec.verdict === "PASS"
                          ? t("detect.verdictPass")
                          : t("detect.verdictFail")}
                      </span>
                    </Badge>
                  </button>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
