import * as React from "react";
import {
  ArrowUpDown,
  CheckCircle2,
  Download,
  FileSpreadsheet,
  FileText,
  FolderOpen,
  ImagePlus,
  Loader2,
  Pause,
  Play,
  Square,
  Trash2,
  XCircle,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import { VAL_EXAMPLES } from "../fixtures";
import { saveInspectionToHistory } from "../lib/historyDb";
import {
  generateCombinedBatchPdfBytes,
  triggerPdfDownload,
} from "../lib/pdfReport";
import {
  getClassColor,
  getClassDisplayName,
} from "../lib/v11Rules";
import { useAppStore } from "../store/useAppStore";
import type { DetectResult } from "../types";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./ui/card";
import { fmtNum } from "../lib/format";

export interface BatchQueueItem {
  id: string;
  file: File;
  filename: string;
  status: "queued" | "running" | "done" | "error" | "cancelled";
  verdict?: "PASS" | "FAIL";
  defectCount?: number;
  result?: DetectResult;
  error?: string;
}

// eslint-disable-next-line react-refresh/only-export-components
export async function runBatchQueueSequentially(params: {
  items: BatchQueueItem[];
  confThreshold: number;
  detectFn: (file: File, conf: number) => Promise<DetectResult>;
  onUpdateItem: (id: string, patch: Partial<BatchQueueItem>) => void;
  shouldCancel?: () => boolean;
  waitWhilePaused?: () => Promise<void>;
}): Promise<BatchQueueItem[]> {
  const updated: BatchQueueItem[] = params.items.map((it) => ({ ...it }));

  for (let i = 0; i < updated.length; i++) {
    const item = updated[i];
    if (item.status === "done") continue;

    if (params.shouldCancel?.()) {
      item.status = "cancelled";
      params.onUpdateItem(item.id, { status: "cancelled" });
      continue;
    }

    if (params.waitWhilePaused) {
      await params.waitWhilePaused();
    }

    if (params.shouldCancel?.()) {
      item.status = "cancelled";
      params.onUpdateItem(item.id, { status: "cancelled" });
      continue;
    }

    item.status = "running";
    params.onUpdateItem(item.id, { status: "running" });

    try {
      const res = await params.detectFn(item.file, params.confThreshold);
      const activeDets = res.detections.filter(
        (d) => d.conf >= params.confThreshold,
      );
      const verdict: "PASS" | "FAIL" =
        activeDets.length > 0 ? "FAIL" : "PASS";
      item.status = "done";
      item.result = res;
      item.verdict = verdict;
      item.defectCount = activeDets.length;
      params.onUpdateItem(item.id, {
        status: "done",
        result: res,
        verdict,
        defectCount: activeDets.length,
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      item.status = "error";
      item.error = msg;
      params.onUpdateItem(item.id, { status: "error", error: msg });
    }
  }

  return updated;
}

interface BatchViewProps {
  onDetectFile: (file: File, conf: number) => Promise<DetectResult>;
  onOpenInInspect?: (item: BatchQueueItem) => void;
}

type SortKey = "filename" | "defectCount" | "time";

export function BatchView({ onDetectFile, onOpenInInspect }: BatchViewProps) {
  const { t } = useTranslation();
  const lang = useAppStore((s) => s.lang);
  const meta = useAppStore((s) => s.meta);
  const confThreshold = useAppStore((s) => s.confThreshold);
  const bumpHistoryVersion = useAppStore((s) => s.bumpHistoryVersion);

  const filesInputRef = React.useRef<HTMLInputElement>(null);
  const folderInputRef = React.useRef<HTMLInputElement>(null);

  const [items, setItems] = React.useState<BatchQueueItem[]>([]);
  const [running, setRunning] = React.useState(false);
  const [paused, setPaused] = React.useState(false);
  const [sortKey, setSortKey] = React.useState<SortKey>("filename");
  const [sortAsc, setSortAsc] = React.useState(true);

  const pausedRef = React.useRef(false);
  const cancelledRef = React.useRef(false);

  const appendFiles = React.useCallback((fileList: File[]) => {
    const imageFiles = fileList.filter(
      (f) =>
        f.type.startsWith("image/") ||
        /\.(jpe?g|png|bmp|webp)$/i.test(f.name),
    );
    if (imageFiles.length === 0) return;
    setItems((prev) => [
      ...prev,
      ...imageFiles.map((file, idx) => ({
        id: `batch_${Date.now()}_${idx}_${Math.random().toString(36).slice(2, 6)}`,
        file,
        filename: file.name,
        status: "queued" as const,
      })),
    ]);
  }, []);

  const handleAddValSamples = async () => {
    const loaded: File[] = [];
    for (const ex of VAL_EXAMPLES) {
      try {
        const res = await fetch(ex.url);
        if (!res.ok) continue;
        const blob = await res.blob();
        loaded.push(
          new File([blob], ex.filename, {
            type: blob.type || "image/jpeg",
          }),
        );
      } catch {
        // ignore fetch error
      }
    }
    appendFiles(loaded);
  };

  const handleStartQueue = async () => {
    if (running || items.length === 0) return;
    setRunning(true);
    setPaused(false);
    pausedRef.current = false;
    cancelledRef.current = false;

    await runBatchQueueSequentially({
      items,
      confThreshold,
      detectFn: async (file, conf) => {
        const res = await onDetectFile(file, conf);
        void saveInspectionToHistory({
          filename: file.name,
          fileOrUrl: file,
          confThreshold: conf,
          result: res,
        }).then(() => bumpHistoryVersion());
        return res;
      },
      onUpdateItem: (id, patch) => {
        setItems((prev) =>
          prev.map((it) => (it.id === id ? { ...it, ...patch } : it)),
        );
      },
      shouldCancel: () => cancelledRef.current,
      waitWhilePaused: async () => {
        while (pausedRef.current && !cancelledRef.current) {
          await new Promise((r) => setTimeout(r, 80));
        }
      },
    });

    setRunning(false);
    setPaused(false);
  };

  const handleTogglePause = () => {
    const next = !paused;
    setPaused(next);
    pausedRef.current = next;
  };

  const handleCancel = () => {
    cancelledRef.current = true;
    pausedRef.current = false;
    setPaused(false);
  };

  const doneItems = React.useMemo(
    () => items.filter((it) => it.status === "done" && it.result),
    [items],
  );
  const passCount = React.useMemo(
    () => doneItems.filter((it) => it.verdict === "PASS").length,
    [doneItems],
  );
  const failCount = React.useMemo(
    () => doneItems.filter((it) => it.verdict === "FAIL").length,
    [doneItems],
  );

  const classDistribution = React.useMemo(() => {
    const counts: Record<string, number> = {};
    for (const it of doneItems) {
      for (const d of it.result?.detections ?? []) {
        if (d.conf >= confThreshold) {
          counts[d.cls] = (counts[d.cls] ?? 0) + 1;
        }
      }
    }
    return Object.entries(counts).map(([cls, count]) => ({
      cls,
      name: getClassDisplayName(cls, lang, true),
      value: count,
      color: getClassColor(cls),
    }));
  }, [doneItems, confThreshold, lang]);

  const sortedItems = React.useMemo(() => {
    const copy = [...items];
    copy.sort((a, b) => {
      let cmp = 0;
      if (sortKey === "filename") {
        cmp = a.filename.localeCompare(b.filename);
      } else if (sortKey === "defectCount") {
        cmp = (a.defectCount ?? -1) - (b.defectCount ?? -1);
      } else if (sortKey === "time") {
        cmp =
          (a.result?.timing.total_ms ?? -1) -
          (b.result?.timing.total_ms ?? -1);
      }
      return sortAsc ? cmp : -cmp;
    });
    return copy;
  }, [items, sortKey, sortAsc]);

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortAsc(!sortAsc);
    } else {
      setSortKey(key);
      setSortAsc(true);
    }
  };

  const handleExportCsv = () => {
    if (doneItems.length === 0 || typeof document === "undefined") return;
    const header = [
      "filename",
      "verdict",
      "defect_count",
      "classes",
      "conf_threshold",
      "model_ms",
      "total_ms",
      "backend",
    ];
    const rows = doneItems.map((it) => {
      const dets =
        it.result?.detections.filter((d) => d.conf >= confThreshold) ?? [];
      const classes = dets.map((d) => d.cls).join(";");
      return [
        `"${it.filename.replace(/"/g, '""')}"`,
        it.verdict ?? "",
        String(dets.length),
        `"${classes}"`,
        confThreshold.toFixed(3),
        it.result ? it.result.timing.model_ms.toFixed(2) : "",
        it.result ? it.result.timing.total_ms.toFixed(2) : "",
        it.result?.backend ?? "",
      ].join(",");
    });
    const csv = [header.join(","), ...rows].join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "tracera_batch_results.csv";
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  const handleExportCombinedPdf = async () => {
    if (doneItems.length === 0) return;
    const bytes = await generateCombinedBatchPdfBytes({
      items: doneItems.map((it) => ({
        filename: it.filename,
        confThreshold,
        result: it.result!,
      })),
      meta,
      lang,
    });
    triggerPdfDownload(bytes, "tracera_toplu_rapor.pdf");
  };

  return (
    <div className="space-y-5" data-testid="batch-view">
      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <CardTitle>{t("batch.title")}</CardTitle>
              <CardDescription>{t("batch.subtitle")}</CardDescription>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <input
                ref={filesInputRef}
                type="file"
                accept="image/*"
                multiple
                className="hidden"
                data-testid="batch-files-input"
                onChange={(e) => {
                  if (e.target.files) {
                    appendFiles(Array.from(e.target.files));
                  }
                  e.target.value = "";
                }}
              />
              <input
                ref={folderInputRef}
                type="file"
                accept="image/*"
                multiple
                // @ts-expect-error webkitdirectory standard dışı öznitelik
                webkitdirectory=""
                className="hidden"
                data-testid="batch-folder-input"
                onChange={(e) => {
                  if (e.target.files) {
                    appendFiles(Array.from(e.target.files));
                  }
                  e.target.value = "";
                }}
              />
              <Button
                type="button"
                size="sm"
                variant="secondary"
                onClick={() => filesInputRef.current?.click()}
              >
                <ImagePlus className="h-3.5 w-3.5" />
                <span>{t("batch.selectFiles")}</span>
              </Button>
              <Button
                type="button"
                size="sm"
                variant="secondary"
                onClick={() => folderInputRef.current?.click()}
              >
                <FolderOpen className="h-3.5 w-3.5" />
                <span>{t("batch.selectFolder")}</span>
              </Button>
              <Button
                type="button"
                size="sm"
                variant="secondary"
                data-testid="batch-add-val-btn"
                onClick={() => void handleAddValSamples()}
              >
                <span>{t("batch.addValSamples")}</span>
              </Button>
            </div>
          </div>
        </CardHeader>

        <CardContent className="space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-[var(--border-hairline)] bg-[var(--bg-elevated)] p-3">
            <div className="flex flex-wrap items-center gap-2">
              <Button
                type="button"
                size="sm"
                disabled={running || items.length === 0}
                onClick={() => void handleStartQueue()}
                data-testid="batch-start-btn"
              >
                {running ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <Play className="h-3.5 w-3.5" />
                )}
                <span>{t("batch.startBtn")}</span>
              </Button>
              {running && (
                <>
                  <Button
                    type="button"
                    size="sm"
                    variant="secondary"
                    onClick={handleTogglePause}
                    data-testid="batch-pause-btn"
                  >
                    <Pause className="h-3.5 w-3.5" />
                    <span>
                      {paused ? t("batch.resumeBtn") : t("batch.pauseBtn")}
                    </span>
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    variant="destructive"
                    onClick={handleCancel}
                    data-testid="batch-cancel-btn"
                  >
                    <Square className="h-3.5 w-3.5" />
                    <span>{t("batch.cancelBtn")}</span>
                  </Button>
                </>
              )}
              {!running && items.length > 0 && (
                <Button
                  type="button"
                  size="sm"
                  variant="ghost"
                  onClick={() => setItems([])}
                  data-testid="batch-clear-btn"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                  <span>{t("batch.clearBtn")}</span>
                </Button>
              )}
            </div>

            {doneItems.length > 0 && (
              <div className="flex flex-wrap items-center gap-2">
                <Button
                  type="button"
                  size="sm"
                  variant="secondary"
                  onClick={handleExportCsv}
                  data-testid="batch-export-csv-btn"
                >
                  <FileSpreadsheet className="h-3.5 w-3.5" />
                  <span>{t("batch.exportCsv")}</span>
                </Button>
                <Button
                  type="button"
                  size="sm"
                  variant="default"
                  onClick={() => void handleExportCombinedPdf()}
                  data-testid="batch-export-pdf-btn"
                >
                  <FileText className="h-3.5 w-3.5" />
                  <span>{t("batch.exportPdf")}</span>
                </Button>
              </div>
            )}
          </div>

          {/* Özet Kartları (Toplam / Geçti / Kaldı) + Sınıf Dağılım Halkası */}
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
            <div className="grid grid-cols-3 gap-3 lg:col-span-7">
              <div className="elevated-card rounded-xl p-3.5">
                <p className="text-xs text-[var(--text-secondary)]">
                  {t("batch.totalCards")}
                </p>
                <p
                  className="mt-1 font-mono text-2xl font-bold text-[var(--text-primary)] tabular-nums"
                  data-testid="batch-total-count"
                >
                  {items.length}
                </p>
              </div>
              <div className="elevated-card rounded-xl p-3.5">
                <p className="text-xs text-[var(--text-secondary)]">
                  {t("batch.passCards")}
                </p>
                <p
                  className="mt-1 font-mono text-2xl font-bold text-emerald-700 dark:text-emerald-400 tabular-nums"
                  data-testid="batch-pass-count"
                >
                  {passCount}
                </p>
              </div>
              <div className="elevated-card rounded-xl p-3.5">
                <p className="text-xs text-[var(--text-secondary)]">
                  {t("batch.failCards")}
                </p>
                <p
                  className="mt-1 font-mono text-2xl font-bold text-rose-700 dark:text-rose-400 tabular-nums"
                  data-testid="batch-fail-count"
                >
                  {failCount}
                </p>
              </div>
            </div>

            <div className="elevated-card flex flex-col justify-between rounded-xl p-3.5 lg:col-span-5">
              <p className="text-xs font-semibold text-[var(--text-secondary)]">
                {t("batch.classDistTitle")}
              </p>
              {classDistribution.length === 0 ? (
                <p className="py-4 text-xs text-[var(--text-muted)]">
                  —
                </p>
              ) : (
                <div className="mt-2 flex items-center gap-4">
                  <div className="h-24 w-24 shrink-0">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={classDistribution}
                          dataKey="value"
                          nameKey="name"
                          innerRadius={24}
                          outerRadius={42}
                          strokeWidth={1}
                        >
                          {classDistribution.map((entry) => (
                            <Cell key={entry.cls} fill={entry.color} />
                          ))}
                        </Pie>
                        <Tooltip />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="flex flex-wrap gap-1.5 text-xs">
                    {classDistribution.map((c) => (
                      <span
                        key={c.cls}
                        className="inline-flex items-center gap-1.5 rounded-full border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 py-0.5 font-mono text-[11px]"
                      >
                        <span
                          className="h-2 w-2 rounded-full"
                          style={{ backgroundColor: c.color }}
                        />
                        <span>
                          {c.name}: {c.value}
                        </span>
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Sıralanabilir Sonuç Tablosu */}
          {sortedItems.length === 0 ? (
            <div className="rounded-xl border border-[var(--border-hairline)] bg-[var(--bg-elevated)] p-6 text-center text-xs text-[var(--text-muted)]">
              {t("batch.emptyQueue")}
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table
                className="w-full border-collapse text-left text-xs"
                data-testid="batch-results-table"
              >
                <thead>
                  <tr className="border-b border-[var(--border-hairline)] text-[var(--text-secondary)]">
                    <th className="py-2 pr-3">
                      <button
                        type="button"
                        onClick={() => handleSort("filename")}
                        className="inline-flex items-center gap-1 font-semibold hover:text-[var(--text-primary)] cursor-pointer"
                      >
                        <span>{t("batch.tableFile")}</span>
                        <ArrowUpDown className="h-3 w-3" />
                      </button>
                    </th>
                    <th className="py-2 pr-3">{t("batch.tableStatus")}</th>
                    <th className="py-2 pr-3">{t("batch.tableVerdict")}</th>
                    <th className="py-2 pr-3">
                      <button
                        type="button"
                        onClick={() => handleSort("defectCount")}
                        className="inline-flex items-center gap-1 font-semibold hover:text-[var(--text-primary)] cursor-pointer"
                      >
                        <span>{t("batch.tableDefects")}</span>
                        <ArrowUpDown className="h-3 w-3" />
                      </button>
                    </th>
                    <th className="py-2 pr-3">{t("batch.tableClasses")}</th>
                    <th className="py-2 pr-3">
                      <button
                        type="button"
                        onClick={() => handleSort("time")}
                        className="inline-flex items-center gap-1 font-semibold hover:text-[var(--text-primary)] cursor-pointer"
                      >
                        <span>{t("batch.tableTime")}</span>
                        <ArrowUpDown className="h-3 w-3" />
                      </button>
                    </th>
                    <th className="py-2">{t("batch.tableAction")}</th>
                  </tr>
                </thead>
                <tbody className="font-mono tabular-nums">
                  {sortedItems.map((it) => {
                    const dets =
                      it.result?.detections.filter(
                        (d) => d.conf >= confThreshold,
                      ) ?? [];
                    return (
                      <tr
                        key={it.id}
                        data-testid={`batch-row-${it.filename}`}
                        className="border-b border-[var(--border-hairline)]/60 text-[var(--text-primary)]"
                      >
                        <td className="py-2.5 pr-3 font-medium">
                          {it.filename}
                        </td>
                        <td className="py-2.5 pr-3">
                          <Badge variant="secondary">{it.status}</Badge>
                        </td>
                        <td className="py-2.5 pr-3">
                          {it.verdict ? (
                            <Badge
                              variant={
                                it.verdict === "PASS" ? "success" : "danger"
                              }
                            >
                              {it.verdict === "PASS" ? (
                                <CheckCircle2 className="h-3 w-3" />
                              ) : (
                                <XCircle className="h-3 w-3" />
                              )}
                              <span>
                                {it.verdict === "PASS"
                                  ? t("detect.verdictPass")
                                  : t("detect.verdictFail")}
                              </span>
                            </Badge>
                          ) : (
                            "—"
                          )}
                        </td>
                        <td className="py-2.5 pr-3">
                          {it.defectCount !== undefined ? it.defectCount : "—"}
                        </td>
                        <td className="py-2.5 pr-3 font-sans">
                          {dets.length > 0
                            ? dets
                                .map((d) =>
                                  getClassDisplayName(d.cls, lang, true),
                                )
                                .join(", ")
                            : "—"}
                        </td>
                        <td className="py-2.5 pr-3">
                          {it.result
                            ? fmtNum(it.result.timing.total_ms, 1)
                            : "—"}
                        </td>
                        <td className="py-2.5">
                          {it.result && onOpenInInspect && (
                            <Button
                              type="button"
                              size="sm"
                              variant="ghost"
                              onClick={() => onOpenInInspect(it)}
                            >
                              <Download className="h-3 w-3" />
                              <span>{t("batch.openInInspect")}</span>
                            </Button>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
