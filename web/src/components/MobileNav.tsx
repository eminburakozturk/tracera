import { BarChart3, Flame, Info, ScanSearch, Video } from "lucide-react";
import { useTranslation } from "react-i18next";
import { useAppStore, type AppTab } from "../store/useAppStore";
import { cn } from "../lib/utils";

const NAV_ITEMS: Array<{ id: AppTab; icon: typeof ScanSearch; key: string }> = [
  { id: "detect", icon: ScanSearch, key: "nav.detect" },
  { id: "live", icon: Video, key: "nav.live" },
  { id: "explain", icon: Flame, key: "nav.explain" },
  { id: "results", icon: BarChart3, key: "nav.results" },
  { id: "about", icon: Info, key: "nav.about" },
];

export function MobileNav() {
  const { t } = useTranslation();
  const activeTab = useAppStore((s) => s.activeTab);
  const setActiveTab = useAppStore((s) => s.setActiveTab);

  return (
    <nav
      aria-label="Mobil alt gezinme"
      className="sticky bottom-0 z-40 grid grid-cols-5 border-t border-[var(--border-hairline)] bg-[var(--bg-surface)]/95 backdrop-blur-md sm:hidden"
    >
      {NAV_ITEMS.map(({ id, icon: Icon, key }) => {
        const active = activeTab === id;
        return (
          <button
            key={id}
            type="button"
            onClick={() => setActiveTab(id)}
            data-testid={`mobile-tab-${id}`}
            className={cn(
              "flex min-h-[52px] flex-col items-center justify-center gap-0.5 px-2 py-1.5 text-[11px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-[var(--accent)]",
              active
                ? "text-[var(--accent)]"
                : "text-[var(--text-muted)] hover:text-[var(--text-primary)]",
            )}
            aria-current={active ? "page" : undefined}
          >
            <Icon className="h-4 w-4" />
            <span>{t(key)}</span>
          </button>
        );
      })}
    </nav>
  );
}
