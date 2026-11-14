import * as React from "react";
import {
  Check,
  Cpu,
  Globe,
  Moon,
  RotateCcw,
  Settings,
  Sparkles,
  Sun,
  Trash2,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import type { BackendChoice } from "../inference/types";
import { clearAllHistoryRecords } from "../lib/historyDb";
import { useAppStore } from "../store/useAppStore";
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
import { Slider } from "./ui/slider";

interface SettingsViewProps {
  onReopenOnboarding?: () => void;
}

export function SettingsView({ onReopenOnboarding }: SettingsViewProps) {
  const { t } = useTranslation();
  const theme = useAppStore((s) => s.theme);
  const setTheme = useAppStore((s) => s.setTheme);
  const lang = useAppStore((s) => s.lang);
  const setLang = useAppStore((s) => s.setLang);
  const backendChoice = useAppStore((s) => s.backendChoice);
  const setBackendChoice = useAppStore((s) => s.setBackendChoice);
  const localApiAvailable = useAppStore((s) => s.localApiAvailable);
  const apiBaseUrl = useAppStore((s) => s.apiBaseUrl);
  const setApiBaseUrl = useAppStore((s) => s.setApiBaseUrl);
  const confThreshold = useAppStore((s) => s.confThreshold);
  const setConfThreshold = useAppStore((s) => s.setConfThreshold);
  const meta = useAppStore((s) => s.meta);
  const bumpHistoryVersion = useAppStore((s) => s.bumpHistoryVersion);

  const [clearedBanner, setClearedBanner] = React.useState(false);

  const handleClearStorage = async () => {
    await clearAllHistoryRecords();
    if (typeof caches !== "undefined") {
      try {
        const keys = await caches.keys();
        await Promise.all(keys.map((k) => caches.delete(k)));
      } catch {
        // ignore cache error
      }
    }
    bumpHistoryVersion();
    setClearedBanner(true);
    setTimeout(() => setClearedBanner(false), 3000);
  };

  return (
    <div className="space-y-5" data-testid="settings-view">
      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <Settings className="h-5 w-5 text-[var(--accent-ink)]" />
              <CardTitle>{t("settings.title")}</CardTitle>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="secondary" className="font-mono">
                {t("settings.versionLabel")}
              </Badge>
              <Badge variant="default" className="font-mono">
                {t("settings.licenseLabel")}
              </Badge>
            </div>
          </div>
          <CardDescription>{t("settings.subtitle")}</CardDescription>
        </CardHeader>

        <CardContent className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {/* Görünüm ve Dil */}
          <div className="elevated-card space-y-3 rounded-xl p-4">
            <h4 className="text-sm font-semibold text-[var(--text-primary)]">
              {t("settings.appearanceTitle")}
            </h4>
            <div className="flex flex-wrap items-center gap-2">
              <Button
                type="button"
                size="sm"
                variant={theme === "dark" ? "default" : "secondary"}
                onClick={() => setTheme("dark")}
              >
                <Moon className="h-3.5 w-3.5" />
                <span>{t("controls.themeDark")}</span>
              </Button>
              <Button
                type="button"
                size="sm"
                variant={theme === "light" ? "default" : "secondary"}
                onClick={() => setTheme("light")}
              >
                <Sun className="h-3.5 w-3.5" />
                <span>{t("controls.themeLight")}</span>
              </Button>
              <Button
                type="button"
                size="sm"
                variant="secondary"
                onClick={() => setLang(lang === "tr" ? "en" : "tr")}
              >
                <Globe className="h-3.5 w-3.5" />
                <span>{lang === "tr" ? "Türkçe (TR)" : "English (EN)"}</span>
              </Button>
            </div>
            {onReopenOnboarding && (
              <div className="pt-1">
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  onClick={onReopenOnboarding}
                  data-testid="settings-reopen-onboarding-btn"
                >
                  <Sparkles className="h-3.5 w-3.5" />
                  <span>{t("settings.reopenOnboarding")}</span>
                </Button>
              </div>
            )}
          </div>

          {/* Çıkarım Motoru ve Eşik */}
          <div className="elevated-card space-y-3 rounded-xl p-4">
            <h4 className="text-sm font-semibold text-[var(--text-primary)]">
              {t("settings.engineTitle")}
            </h4>
            <div className="space-y-2">
              <Select
                value={backendChoice}
                onValueChange={(val) => setBackendChoice(val as BackendChoice)}
              >
                <SelectTrigger aria-label={t("backend.label")}>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="mock">{t("backend.optMock")}</SelectItem>
                  {localApiAvailable && (
                    <SelectItem value="api">{t("backend.optApi")}</SelectItem>
                  )}
                  <SelectItem value="cloud">{t("backend.optCloud")}</SelectItem>
                  <SelectItem value="ort">{t("backend.optOrt")}</SelectItem>
                </SelectContent>
              </Select>

              {localApiAvailable && backendChoice === "api" && (
                <input
                  type="url"
                  value={apiBaseUrl}
                  onChange={(e) => setApiBaseUrl(e.target.value)}
                  aria-label={t("backend.apiUrlLabel")}
                  className="h-9 w-full rounded-lg border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-3 font-mono text-xs text-[var(--text-primary)]"
                />
              )}
            </div>

            <div className="space-y-1.5 pt-1">
              <div className="flex items-center justify-between text-xs">
                <span className="font-medium text-[var(--text-secondary)]">
                  {t("detect.confLabel")}:{" "}
                  <strong className="font-mono text-[var(--accent-ink)]">
                    {confThreshold.toFixed(3)}
                  </strong>
                </span>
                <Button
                  type="button"
                  size="sm"
                  variant="ghost"
                  onClick={() => setConfThreshold(meta.conf)}
                  data-testid="settings-reset-threshold-btn"
                >
                  <RotateCcw className="h-3 w-3" />
                  <span>{t("settings.resetThreshold")}</span>
                </Button>
              </div>
              <Slider
                min={0.05}
                max={0.95}
                step={0.005}
                value={[confThreshold]}
                onValueChange={([v]) => setConfThreshold(v)}
                aria-label={t("detect.confLabel")}
              />
            </div>
          </div>

          {/* Model Bilgisi (meta.json) */}
          <div className="elevated-card space-y-2 rounded-xl p-4">
            <div className="flex items-center gap-2">
              <Cpu className="h-4 w-4 text-[var(--accent-ink)]" />
              <h4 className="text-sm font-semibold text-[var(--text-primary)]">
                {t("settings.modelInfoTitle")}
              </h4>
            </div>
            <dl className="grid grid-cols-2 gap-2 font-mono text-xs text-[var(--text-secondary)] tabular-nums">
              <div>
                <dt className="text-[10px] text-[var(--text-muted)]">Model</dt>
                <dd className="font-semibold text-[var(--text-primary)]">
                  {meta.model}
                </dd>
              </div>
              <div>
                <dt className="text-[10px] text-[var(--text-muted)]">Boyut</dt>
                <dd className="font-semibold text-[var(--text-primary)]">
                  {meta.imgsz}×{meta.imgsz}
                </dd>
              </div>
              <div>
                <dt className="text-[10px] text-[var(--text-muted)]">
                  Kilitli Eşik
                </dt>
                <dd className="font-semibold text-[var(--text-primary)]">
                  {meta.conf} (NMS IoU {meta.iou_nms})
                </dd>
              </div>
              <div>
                <dt className="text-[10px] text-[var(--text-muted)]">ONNX</dt>
                <dd className="truncate font-semibold text-[var(--text-primary)]">
                  {meta.onnx}
                </dd>
              </div>
            </dl>
          </div>

          {/* Önbellek ve Geçmiş Temizliği */}
          <div className="elevated-card flex flex-col justify-between space-y-3 rounded-xl p-4">
            <div>
              <h4 className="text-sm font-semibold text-[var(--text-primary)]">
                {t("settings.storageTitle")}
              </h4>
              <p className="mt-1 text-xs text-[var(--text-secondary)]">
                GNU AGPL-3.0 · Sürüm v1.1.0
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Button
                type="button"
                size="sm"
                variant="destructive"
                onClick={() => void handleClearStorage()}
                data-testid="settings-clear-cache-btn"
              >
                <Trash2 className="h-3.5 w-3.5" />
                <span>{t("settings.clearCacheBtn")}</span>
              </Button>
              {clearedBanner && (
                <span className="inline-flex items-center gap-1 text-xs font-medium text-emerald-700 dark:text-emerald-300">
                  <Check className="h-3.5 w-3.5" />
                  <span>{t("settings.cacheCleared")}</span>
                </span>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
