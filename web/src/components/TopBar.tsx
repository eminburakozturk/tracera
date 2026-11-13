import { Cpu, Globe, Info, Moon, RefreshCw, Server, Sparkles, Sun, Wifi, WifiOff } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { Tabs, TabsList, TabsTrigger } from "./ui/tabs";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "./ui/tooltip";
import type { BackendChoice } from "../inference/types";
import { useAppStore, type AppTab } from "../store/useAppStore";

interface TopBarProps {
  onRecheckHealth: () => void;
  ortEnabled?: boolean;
}

export function TopBar({ onRecheckHealth, ortEnabled = false }: TopBarProps) {
  const { t } = useTranslation();
  const activeTab = useAppStore((s) => s.activeTab);
  const setActiveTab = useAppStore((s) => s.setActiveTab);
  const theme = useAppStore((s) => s.theme);
  const setTheme = useAppStore((s) => s.setTheme);
  const lang = useAppStore((s) => s.lang);
  const setLang = useAppStore((s) => s.setLang);
  const backendChoice = useAppStore((s) => s.backendChoice);
  const setBackendChoice = useAppStore((s) => s.setBackendChoice);
  const apiBaseUrl = useAppStore((s) => s.apiBaseUrl);
  const setApiBaseUrl = useAppStore((s) => s.setApiBaseUrl);
  const apiHealth = useAppStore((s) => s.apiHealth);

  const renderBackendBadge = () => {
    if (backendChoice === "ort") {
      return (
        <Badge variant="default" data-testid="backend-badge">
          <Cpu className="h-3 w-3" />
          <span>{t("backend.badgeDevice")}</span>
        </Badge>
      );
    }
    if (backendChoice === "api") {
      return (
        <Badge
          variant={apiHealth.status === "offline" ? "danger" : "success"}
          data-testid="backend-badge"
        >
          <Server className="h-3 w-3" />
          <span>{t("backend.badgeLocalApi")}</span>
        </Badge>
      );
    }
    return (
      <Badge variant="secondary" data-testid="backend-badge">
        <Sparkles className="h-3 w-3" />
        <span>{t("backend.badgeMock")}</span>
      </Badge>
    );
  };

  return (
    <header className="sticky top-0 z-40 border-b border-[var(--border-hairline)] bg-[var(--bg-surface)]/92 backdrop-blur-md">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 px-4 py-3">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-[var(--accent)]/40 bg-[var(--accent-soft)] text-[var(--accent)]">
            <svg
              viewBox="0 0 24 24"
              className="h-5 w-5"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.9"
              aria-hidden="true"
            >
              <path d="M4 7h6l3 3h7" />
              <path d="M4 17h5l4-4h7" />
              <circle cx="4" cy="7" r="1.5" fill="currentColor" />
              <circle cx="20" cy="10" r="1.5" fill="currentColor" />
              <circle cx="4" cy="17" r="1.5" fill="currentColor" />
              <circle cx="20" cy="13" r="1.5" fill="currentColor" />
            </svg>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-base font-semibold tracking-tight text-[var(--text-primary)]">
                {t("app.name")}
              </span>
              {renderBackendBadge()}
            </div>
            <p className="hidden text-xs text-[var(--text-muted)] md:block">
              {t("app.subtitle")}
            </p>
          </div>
        </div>

        <Tabs
          value={activeTab}
          onValueChange={(val) => setActiveTab(val as AppTab)}
          className="hidden sm:block"
        >
          <TabsList aria-label="Ana gezinme">
            <TabsTrigger value="detect" data-testid="tab-detect">
              {t("nav.detect")}
            </TabsTrigger>
            <TabsTrigger value="explain" data-testid="tab-explain">
              {t("nav.explain")}
            </TabsTrigger>
            <TabsTrigger value="results" data-testid="tab-results">
              {t("nav.results")}
            </TabsTrigger>
            <TabsTrigger value="about" data-testid="tab-about">
              {t("nav.about")}
            </TabsTrigger>
          </TabsList>
        </Tabs>

        <div className="flex flex-wrap items-center gap-2">
          <div className="flex items-center gap-1.5">
            <div className="w-[178px]">
              <Select
                value={backendChoice}
                onValueChange={(val) => setBackendChoice(val as BackendChoice)}
              >
                <SelectTrigger
                  aria-label={t("backend.label")}
                  data-testid="backend-select"
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="mock">{t("backend.optMock")}</SelectItem>
                  <SelectItem value="api">{t("backend.optApi")}</SelectItem>
                  <SelectItem value="ort" disabled={!ortEnabled}>
                    {t("backend.optOrt")}
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>

            <TooltipProvider delayDuration={150}>
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    type="button"
                    aria-label={t("backend.parityInfo")}
                    title={t("backend.parityInfo")}
                    data-testid="backend-parity-info"
                    className="inline-flex h-8 w-8 items-center justify-center rounded-lg border border-[var(--border-hairline)] bg-[var(--bg-elevated)] text-[var(--text-secondary)] hover:border-[var(--accent)] hover:text-[var(--accent)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] cursor-pointer"
                  >
                    <Info className="h-3.5 w-3.5" />
                  </button>
                </TooltipTrigger>
                <TooltipContent className="max-w-xs leading-relaxed">
                  {t("backend.parityInfo")}
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>

          {backendChoice === "api" && (
            <div className="flex items-center gap-1.5">
              <Badge
                variant={
                  apiHealth.status === "online"
                    ? "success"
                    : apiHealth.status === "offline"
                      ? "danger"
                      : "secondary"
                }
                data-testid="api-health-badge"
              >
                {apiHealth.status === "online" ? (
                  <Wifi className="h-3 w-3" />
                ) : (
                  <WifiOff className="h-3 w-3" />
                )}
                <span>
                  {apiHealth.status === "online"
                    ? t("backend.serverOnline")
                    : apiHealth.status === "offline"
                      ? t("backend.serverOffline")
                      : t("backend.serverChecking")}
                </span>
              </Badge>
              <input
                type="url"
                value={apiBaseUrl}
                onChange={(e) => setApiBaseUrl(e.target.value)}
                aria-label={t("backend.apiUrlLabel")}
                className="hidden h-8 w-36 rounded-md border border-[var(--border-hairline)] bg-[var(--bg-elevated)] px-2 font-mono text-xs text-[var(--text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--accent)] xl:block"
              />
              <Button
                type="button"
                variant="ghost"
                size="icon"
                onClick={onRecheckHealth}
                aria-label={t("backend.recheck")}
                title={t("backend.recheck")}
              >
                <RefreshCw className="h-3.5 w-3.5" />
              </Button>
            </div>
          )}

          <Button
            type="button"
            variant="secondary"
            size="sm"
            onClick={() => setLang(lang === "tr" ? "en" : "tr")}
            aria-label={t("controls.language")}
            data-testid="lang-toggle"
          >
            <Globe className="h-3.5 w-3.5" />
            <span>{lang.toUpperCase()}</span>
          </Button>

          <Button
            type="button"
            variant="secondary"
            size="icon"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            aria-label={theme === "dark" ? t("controls.themeLight") : t("controls.themeDark")}
            title={theme === "dark" ? t("controls.themeLight") : t("controls.themeDark")}
            data-testid="theme-toggle"
          >
            {theme === "dark" ? (
              <Sun className="h-4 w-4" />
            ) : (
              <Moon className="h-4 w-4" />
            )}
          </Button>
        </div>
      </div>
    </header>
  );
}
