import { Flame, Home, Layers, MoreHorizontal, ScanSearch } from "lucide-react";
import { useTranslation } from "react-i18next";
import { cn } from "../lib/utils";
import { useAppStore, type AppTab } from "../store/useAppStore";

const MOBILE_TABS: Array<{ id: AppTab; icon: typeof ScanSearch; key: string }> = [
  { id: "home", icon: Home, key: "nav.home" },
  { id: "detect", icon: ScanSearch, key: "nav.detect" },
  { id: "explain", icon: Flame, key: "nav.explain" },
  { id: "batch", icon: Layers, key: "nav.batch" },
  { id: "more", icon: MoreHorizontal, key: "nav.more" },
];

const MORE_SUBTABS = new Set<AppTab>([
  "more",
  "live",
  "history",
  "guide",
  "results",
  "about",
  "developer",
  "settings",
]);

export function MobileNav() {
  const { t } = useTranslation();
  const activeTab = useAppStore((s) => s.activeTab);
  const setActiveTab = useAppStore((s) => s.setActiveTab);

  return (
    <nav
      data-testid="mobile-nav"
      aria-label="Mobil alt gezinme"
      className="safe-area-bottom sticky bottom-0 z-40 grid grid-cols-5 border-t border-[var(--border-hairline)] bg-[var(--bg-surface)]/95 backdrop-blur-md sm:hidden"
    >
      {MOBILE_TABS.map(({ id, icon: Icon, key }) => {
        const active =
          id === "more" ? MORE_SUBTABS.has(activeTab) : activeTab === id;
        return (
          <button
            key={id}
            type="button"
            onClick={() => setActiveTab(id)}
            data-testid={`mobile-tab-${id}`}
            className={cn(
              "flex min-h-[52px] flex-col items-center justify-center gap-0.5 px-1.5 py-1.5 text-[11px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-[var(--accent)] cursor-pointer",
              active
                ? "text-[var(--accent-ink)] font-semibold"
                : "text-[var(--text-muted)] hover:text-[var(--text-primary)]",
            )}
            aria-current={active ? "page" : undefined}
          >
            <Icon className="h-4 w-4" />
            <span className="truncate">{t(key)}</span>
          </button>
        );
      })}
    </nav>
  );
}
