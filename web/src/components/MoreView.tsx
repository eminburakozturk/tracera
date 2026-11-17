import {
  BarChart3,
  BookOpen,
  ChevronRight,
  Clock,
  Settings,
  UserCheck,
  Video,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import { useAppStore, type AppTab } from "../store/useAppStore";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";

const MORE_ITEMS: Array<{
  id: AppTab;
  icon: typeof Video;
  labelKey: string;
  descTr: string;
  descEn: string;
}> = [
  {
    id: "live",
    icon: Video,
    labelKey: "nav.live",
    descTr: "Canlı video ve kamera akışında kare atlamalı ONNX Runtime Web denetimi.",
    descEn: "Frame-dropping ONNX Runtime Web inspection on live video and camera streams.",
  },
  {
    id: "history",
    icon: Clock,
    labelKey: "nav.history",
    descTr: "Yerel IndexedDB muayene geçmişi, kayıt silme ve yan yana 2 kart karşılaştırma.",
    descEn: "Local IndexedDB inspection history, record deletion, and side-by-side comparison.",
  },
  {
    id: "guide",
    icon: BookOpen,
    labelKey: "nav.guide",
    descTr: "6 kusur sınıfının resmi tanımları, Kart 05 & 11 kırpımları ve doğruluk metrikleri.",
    descEn: "Official definitions for 6 defect classes, Board 05 & 11 crops, and accuracy metrics.",
  },
  {
    id: "results",
    icon: BarChart3,
    labelKey: "nav.results",
    descTr: "5-katlı CV, çözünürlük kıyası, hata analizi galerisi ve cihaz hız testi.",
    descEn: "5-fold CV, resolution comparison, error analysis gallery, and device speed test.",
  },
  {
    id: "about",
    icon: UserCheck,
    labelKey: "nav.about",
    descTr: "Geliştirici bilgileri, onaylı proje zaman çizelgesi, sınırlılıklar ve AGPL-3.0 lisansı.",
    descEn: "Developer profile, approved project timeline, limitations, and AGPL-3.0 license.",
  },
  {
    id: "settings",
    icon: Settings,
    labelKey: "nav.settings",
    descTr: "Tema, dil, çıkarım motoru, varsayılan güven eşiği (0,395) ve önbellek temizliği.",
    descEn: "Theme, language, inference engine, default threshold (0.395), and cache cleanup.",
  },
];

export function MoreView() {
  const { t } = useTranslation();
  const lang = useAppStore((s) => s.lang);
  const setActiveTab = useAppStore((s) => s.setActiveTab);

  return (
    <div className="space-y-4" data-testid="more-view">
      <Card>
        <CardHeader>
          <CardTitle>{t("nav.moreTitle")}</CardTitle>
          <CardDescription>{t("nav.moreSubtitle")}</CardDescription>
        </CardHeader>
        <CardContent className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          {MORE_ITEMS.map(({ id, icon: Icon, labelKey, descTr, descEn }) => (
            <button
              key={id}
              type="button"
              data-testid={`more-link-${id}`}
              onClick={() => setActiveTab(id)}
              className="flex items-center justify-between gap-3 rounded-xl border border-[var(--border-hairline)] bg-[var(--bg-elevated)] p-4 text-left transition-colors hover:border-[var(--accent)] cursor-pointer"
            >
              <div className="flex items-start gap-3">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-[var(--accent)]/30 bg-[var(--accent-soft)] text-[var(--accent-ink)]">
                  <Icon className="h-5 w-5" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-[var(--text-primary)]">
                    {t(labelKey)}
                  </p>
                  <p className="mt-0.5 text-xs text-[var(--text-secondary)]">
                    {lang === "en" ? descEn : descTr}
                  </p>
                </div>
              </div>
              <ChevronRight className="h-4 w-4 shrink-0 text-[var(--text-muted)]" />
            </button>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}
