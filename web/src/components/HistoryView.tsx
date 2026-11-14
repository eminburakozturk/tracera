import * as React from "react";
import {
  CheckCircle2,
  Columns2,
  Database,
  ExternalLink,
  Trash2,
  X,
  XCircle,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import {
  clearAllHistoryRecords,
  deleteHistoryRecord,
  estimateHistoryStorageBytes,
  listHistoryRecords,
} from "../lib/historyDb";
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

interface HistoryViewProps {
  onOpenRecord: (record: HistoryRecord) => void;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function HistoryView({ onOpenRecord }: HistoryViewProps) {
  const { t } = useTranslation();
  const lang = useAppStore((s) => s.lang);
  const historyVersion = useAppStore((s) => s.historyVersion);
  const bumpHistoryVersion = useAppStore((s) => s.bumpHistoryVersion);

  const [records, setRecords] = React.useState<HistoryRecord[]>([]);
  const [storageBytes, setStorageBytes] = React.useState(0);
  const [compareIds, setCompareIds] = React.useState<string[]>([]);

  const reload = React.useCallback(async () => {
    const [list, bytes] = await Promise.all([
      listHistoryRecords(),
      estimateHistoryStorageBytes(),
    ]);
    setRecords(list);
    setStorageBytes(bytes);
  }, []);

  React.useEffect(() => {
    void reload();
  }, [reload, historyVersion]);

  const handleDelete = async (id: string) => {
    await deleteHistoryRecord(id);
    setCompareIds((prev) => prev.filter((x) => x !== id));
    bumpHistoryVersion();
    await reload();
  };

  const handleClearAll = async () => {
    await clearAllHistoryRecords();
    setCompareIds([]);
    bumpHistoryVersion();
    await reload();
  };

  const toggleCompare = (id: string) => {
    setCompareIds((prev) => {
      if (prev.includes(id)) {
        return prev.filter((x) => x !== id);
      }
      if (prev.length >= 2) {
        return [prev[1], id];
      }
      return [...prev, id];
    });
  };

  const comparedRecords = React.useMemo(
    () =>
      compareIds
        .map((id) => records.find((r) => r.id === id))
        .filter((r): r is HistoryRecord => Boolean(r)),
    [compareIds, records],
  );

  return (
    <div className="space-y-5" data-testid="history-view">
      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <CardTitle>{t("history.title")}</CardTitle>
              <CardDescription>{t("history.subtitle")}</CardDescription>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Badge
                variant="secondary"
                className="font-mono tabular-nums"
                data-testid="history-storage-badge"
              >
                <Database className="h-3 w-3" />
                <span>
                  {t("history.storageUsed")}: {formatBytes(storageBytes)} (
                  {records.length})
                </span>
              </Badge>
              {records.length > 0 && (
                <Button
                  type="button"
                  size="sm"
                  variant="destructive"
                  onClick={() => void handleClearAll()}
                  data-testid="history-clear-all-btn"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                  <span>{t("history.clearAll")}</span>
                </Button>
              )}
            </div>
          </div>
        </CardHeader>

        <CardContent className="space-y-4">
          {/* 2 Kaydı Yan Yana Karşılaştırma Paneli */}
          {comparedRecords.length === 2 && (
            <div
              data-testid="history-compare-panel"
              className="space-y-3 rounded-xl border border-[var(--accent)]/45 bg-[var(--accent-soft)] p-4"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-sm font-semibold text-[var(--text-primary)]">
                  <Columns2 className="h-4 w-4 text-[var(--accent-ink)]" />
                  <span>{t("history.compareTitle")}</span>
                </div>
                <Button
                  type="button"
                  size="sm"
                  variant="ghost"
                  onClick={() => setCompareIds([])}
                >
                  <X className="h-3.5 w-3.5" />
                  <span>{t("history.closeCompare")}</span>
                </Button>
              </div>

              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                {comparedRecords.map((rec) => (
                  <div
                    key={rec.id}
                    className="rounded-xl border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="truncate font-mono text-xs font-semibold text-[var(--text-primary)]">
                        {rec.filename}
                      </span>
                      <Badge
                        variant={rec.verdict === "PASS" ? "success" : "danger"}
                      >
                        {rec.verdict === "PASS"
                          ? t("detect.verdictPass")
                          : t("detect.verdictFail")}
                      </Badge>
                    </div>
                    <img
                      src={rec.thumbnailDataUrl}
                      alt={rec.filename}
                      className="mt-2 h-36 w-full rounded-lg border border-[var(--border-hairline)] object-cover"
                    />
                    <div className="mt-2 flex flex-wrap items-center justify-between font-mono text-[11px] text-[var(--text-secondary)] tabular-nums">
                      <span>Conf: {rec.confThreshold.toFixed(3)}</span>
                      <span>Kusur: {rec.defectCount}</span>
                      <span>{rec.result.timing.total_ms.toFixed(1)} ms</span>
                    </div>
                    <div className="mt-2 flex flex-wrap gap-1">
                      {rec.result.detections.map((d) => (
                        <span
                          key={d.id}
                          className="inline-flex items-center gap-1 rounded-full border border-[var(--border-hairline)] bg-[var(--bg-elevated)] px-2 py-0.5 font-mono text-[10px]"
                        >
                          <span
                            className="h-1.5 w-1.5 rounded-full"
                            style={{ backgroundColor: getClassColor(d.cls) }}
                          />
                          <span>
                            {getClassDisplayName(d.cls, lang, true)}{" "}
                            {(d.conf * 100).toFixed(0)}%
                          </span>
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {records.length === 0 ? (
            <div
              data-testid="history-empty-state"
              className="rounded-xl border border-[var(--border-hairline)] bg-[var(--bg-elevated)] p-6 text-center text-xs text-[var(--text-muted)]"
            >
              {t("history.emptyHistory")}
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
              {records.map((rec) => {
                const selectedForCompare = compareIds.includes(rec.id);
                return (
                  <div
                    key={rec.id}
                    data-testid={`history-card-${rec.id}`}
                    className={cn(
                      "flex flex-col justify-between gap-3 rounded-xl border p-3.5 transition-colors",
                      selectedForCompare
                        ? "border-[var(--accent)] bg-[var(--accent-soft)]"
                        : "border-[var(--border-hairline)] bg-[var(--bg-elevated)]",
                    )}
                  >
                    <div className="flex items-start gap-3">
                      <img
                        src={rec.thumbnailDataUrl}
                        alt={rec.filename}
                        className="h-16 w-20 shrink-0 rounded-lg border border-[var(--border-hairline)] object-cover"
                      />
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center justify-between gap-2">
                          <p className="truncate font-mono text-xs font-semibold text-[var(--text-primary)]">
                            {rec.filename}
                          </p>
                          <Badge
                            variant={
                              rec.verdict === "PASS" ? "success" : "danger"
                            }
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
                        </div>
                        <p className="mt-1 font-mono text-[11px] text-[var(--text-secondary)] tabular-nums">
                          {new Date(rec.createdAt).toLocaleString(
                            lang === "en" ? "en-US" : "tr-TR",
                          )}{" "}
                          · Conf {rec.confThreshold.toFixed(3)} ·{" "}
                          {rec.defectCount} kusur
                        </p>
                        <div className="mt-1.5 flex flex-wrap gap-1">
                          {rec.result.detections.slice(0, 4).map((d) => (
                            <span
                              key={d.id}
                              className="inline-flex items-center gap-1 rounded-full border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 py-0.5 text-[10px]"
                            >
                              <span
                                className="h-1.5 w-1.5 rounded-full"
                                style={{
                                  backgroundColor: getClassColor(d.cls),
                                }}
                              />
                              <span>
                                {getClassDisplayName(d.cls, lang, true)}
                              </span>
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>

                    <div className="flex flex-wrap items-center justify-end gap-2 border-t border-[var(--border-hairline)]/60 pt-2">
                      <Button
                        type="button"
                        size="sm"
                        variant={selectedForCompare ? "default" : "secondary"}
                        onClick={() => toggleCompare(rec.id)}
                        data-testid={`history-compare-${rec.id}`}
                      >
                        <Columns2 className="h-3.5 w-3.5" />
                        <span>{t("history.selectForCompare")}</span>
                      </Button>
                      <Button
                        type="button"
                        size="sm"
                        variant="secondary"
                        onClick={() => onOpenRecord(rec)}
                        data-testid={`history-open-${rec.id}`}
                      >
                        <ExternalLink className="h-3.5 w-3.5" />
                        <span>{t("history.openRecord")}</span>
                      </Button>
                      <Button
                        type="button"
                        size="sm"
                        variant="ghost"
                        onClick={() => void handleDelete(rec.id)}
                        data-testid={`history-delete-${rec.id}`}
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                        <span>{t("history.deleteRecord")}</span>
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
  );
}
