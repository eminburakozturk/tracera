import * as React from "react";
import { ArrowLeft, ArrowRight, Check, Flame, ScanSearch, ShieldCheck, X } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Button } from "./ui/button";

export const ONBOARDING_STORAGE_KEY = "tracera_onboarding_v11_seen";

interface OnboardingModalProps {
  open: boolean;
  onClose: () => void;
}

export function OnboardingModal({ open, onClose }: OnboardingModalProps) {
  const { t } = useTranslation();
  const [step, setStep] = React.useState(0);

  React.useEffect(() => {
    if (open) setStep(0);
  }, [open]);

  if (!open) return null;

  const cards = [
    {
      icon: ScanSearch,
      title: t("onboarding.step1Title"),
      body: t("onboarding.step1Body"),
    },
    {
      icon: Flame,
      title: t("onboarding.step2Title"),
      body: t("onboarding.step2Body"),
    },
    {
      icon: ShieldCheck,
      title: t("onboarding.step3Title"),
      body: t("onboarding.step3Body"),
    },
  ];

  const current = cards[step] ?? cards[0];
  const Icon = current.icon;

  const handleDismiss = () => {
    try {
      localStorage.setItem(ONBOARDING_STORAGE_KEY, "1");
    } catch {
      // ignore storage errors
    }
    onClose();
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={current.title}
      data-testid="onboarding-modal"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-xs"
    >
      <div className="w-full max-w-md space-y-4 rounded-2xl border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-6 shadow-2xl">
        <div className="flex items-center justify-between">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-[var(--accent)]/40 bg-[var(--accent-soft)] text-[var(--accent-ink)]">
            <Icon className="h-5 w-5" />
          </div>
          <Button
            type="button"
            size="sm"
            variant="ghost"
            onClick={handleDismiss}
            data-testid="onboarding-skip-btn"
          >
            <X className="h-3.5 w-3.5" />
            <span>{t("onboarding.skip")}</span>
          </Button>
        </div>

        <div className="space-y-2">
          <h2 className="text-base font-bold text-[var(--text-primary)]">
            {current.title}
          </h2>
          <p className="text-xs leading-relaxed text-[var(--text-secondary)]">
            {current.body}
          </p>
        </div>

        <div className="flex items-center justify-between pt-2">
          <div className="flex items-center gap-1.5">
            {cards.map((_, idx) => (
              <span
                key={idx}
                className={`h-1.5 rounded-full transition-all ${
                  idx === step
                    ? "w-6 bg-[var(--accent)]"
                    : "w-2 bg-[var(--border-strong)]"
                }`}
              />
            ))}
          </div>

          <div className="flex items-center gap-2">
            {step > 0 && (
              <Button
                type="button"
                size="sm"
                variant="secondary"
                onClick={() => setStep((s) => Math.max(0, s - 1))}
              >
                <ArrowLeft className="h-3.5 w-3.5" />
                <span>{t("onboarding.prev")}</span>
              </Button>
            )}
            {step < cards.length - 1 ? (
              <Button
                type="button"
                size="sm"
                onClick={() => setStep((s) => Math.min(cards.length - 1, s + 1))}
                data-testid="onboarding-next-btn"
              >
                <span>{t("onboarding.next")}</span>
                <ArrowRight className="h-3.5 w-3.5" />
              </Button>
            ) : (
              <Button
                type="button"
                size="sm"
                onClick={handleDismiss}
                data-testid="onboarding-finish-btn"
              >
                <Check className="h-3.5 w-3.5" />
                <span>{t("onboarding.finish")}</span>
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
