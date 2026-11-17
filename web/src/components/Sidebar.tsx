import {
  BarChart3,
  BookOpen,
  ChevronLeft,
  ChevronRight,
  Clock,
  Flame,
  Home,
  Layers,
  ScanSearch,
  Settings,
  UserCheck,
  Video,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import { cn } from "../lib/utils";
import { useAppStore, type AppTab } from "../store/useAppStore";

const SIDEBAR_ITEMS: Array<{
  id: AppTab;
  icon: typeof Home;
  labelKey: string;
}> = [
  { id: "home", icon: Home, labelKey: "nav.home" },
  { id: "detect", icon: ScanSearch, labelKey: "nav.detect" },
  { id: "explain", icon: Flame, labelKey: "nav.explain" },
  { id: "live", icon: Video, labelKey: "nav.live" },
  { id: "batch", icon: Layers, labelKey: "nav.batch" },
  { id: "history", icon: Clock, labelKey: "nav.history" },
  { id: "guide", icon: BookOpen, labelKey: "nav.guide" },
  { id: "results", icon: BarChart3, labelKey: "nav.results" },
  { id: "about", icon: UserCheck, labelKey: "nav.about" },
  { id: "settings", icon: Settings, labelKey: "nav.settings" },
];

export function Sidebar() {
  const { t } = useTranslation();
  const activeTab = useAppStore((s) => s.activeTab);
  const setActiveTab = useAppStore((s) => s.setActiveTab);
  const collapsed = useAppStore((s) => s.sidebarCollapsed);
  const setCollapsed = useAppStore((s) => s.setSidebarCollapsed);

  return (
    <aside
      data-testid="desktop-sidebar"
      aria-label="Ana kenar çubuğu"
      className={cn(
        "hidden shrink-0 flex-col border-r border-[var(--border-hairline)] bg-[var(--bg-surface)] transition-[width] duration-200 sm:flex",
        collapsed ? "w-16" : "w-60",
      )}
    >
      <div className="flex h-16 items-center justify-between border-b border-[var(--border-hairline)] px-3.5">
        <button
          type="button"
          onClick={() => setActiveTab("home")}
          className="flex items-center gap-2.5 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] cursor-pointer"
        >
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-[var(--accent)]/40 bg-[var(--accent-soft)] text-[var(--accent-ink)]">
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
          {!collapsed && (
            <div className="min-w-0">
              <div className="flex items-center gap-1.5">
                <span className="truncate text-sm font-bold tracking-tight text-[var(--text-primary)]">
                  {t("app.name")}
                </span>
                <span className="rounded bg-[var(--bg-elevated)] px-1.5 py-0.5 font-mono text-[10px] font-semibold text-[var(--text-muted)]">
                  {t("app.version")}
                </span>
              </div>
              <p className="truncate text-[11px] text-[var(--text-muted)]">
                {t("app.subtitle")}
              </p>
            </div>
          )}
        </button>
      </div>

      <nav className="flex-1 space-y-1 overflow-y-auto p-2.5" aria-label="Ana gezinme">
        {SIDEBAR_ITEMS.map(({ id, icon: Icon, labelKey }) => {
          const isActive =
            activeTab === id || (id === "about" && activeTab === "developer");
          const label = t(labelKey);
          return (
            <button
              key={id}
              type="button"
              onClick={() => setActiveTab(id)}
              data-testid={`tab-${id}`}
              title={collapsed ? label : undefined}
              aria-label={label}
              aria-current={isActive ? "page" : undefined}
              className={cn(
                "flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] cursor-pointer",
                isActive
                  ? "border border-[var(--accent)]/40 bg-[var(--accent-soft)] text-[var(--accent-ink)] font-semibold"
                  : "border border-transparent text-[var(--text-secondary)] hover:bg-[var(--bg-elevated)] hover:text-[var(--text-primary)]",
                collapsed && "justify-center px-2",
              )}
            >
              <Icon className="h-4 w-4 shrink-0" />
              {!collapsed && <span className="truncate">{label}</span>}
            </button>
          );
        })}
      </nav>

      <div className="border-t border-[var(--border-hairline)] p-2.5">
        <button
          type="button"
          onClick={() => setCollapsed(!collapsed)}
          data-testid="sidebar-collapse-btn"
          aria-label={
            collapsed ? t("nav.expandSidebar") : t("nav.collapseSidebar")
          }
          title={collapsed ? t("nav.expandSidebar") : t("nav.collapseSidebar")}
          className="flex w-full items-center justify-center gap-2 rounded-lg border border-[var(--border-hairline)] bg-[var(--bg-elevated)] px-3 py-2 text-xs font-medium text-[var(--text-secondary)] hover:text-[var(--text-primary)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] cursor-pointer"
        >
          {collapsed ? (
            <ChevronRight className="h-4 w-4" />
          ) : (
            <>
              <ChevronLeft className="h-4 w-4" />
              <span>{t("nav.collapseSidebar")}</span>
            </>
          )}
        </button>
      </div>
    </aside>
  );
}
