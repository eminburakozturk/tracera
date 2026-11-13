import { ExternalLink, Scale, ShieldAlert } from "lucide-react";
import { useTranslation } from "react-i18next";
import { useAppStore } from "../store/useAppStore";
import { Badge } from "./ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./ui/card";

export function AboutView() {
  const { t } = useTranslation();
  const meta = useAppStore((s) => s.meta);

  return (
    <div className="grid grid-cols-1 gap-5 lg:grid-cols-12">
      <Card className="lg:col-span-7">
        <CardHeader>
          <CardTitle>{t("about.title")}</CardTitle>
          <CardDescription>
            {meta.model} · {meta.imgsz}×{meta.imgsz} · conf {meta.conf}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5 text-sm leading-relaxed text-[var(--text-secondary)]">
          <p>{t("about.description")}</p>

          <div className="flex flex-wrap gap-2">
            {meta.classes.map((cls, i) => (
              <Badge key={cls} variant="secondary" className="gap-2 py-1">
                <span
                  className="h-2.5 w-2.5 rounded-full"
                  style={{ backgroundColor: meta.colors[i] ?? "#38bdf8" }}
                />
                <span>{t(`classes.${cls}`, { defaultValue: cls })}</span>
              </Badge>
            ))}
          </div>

          <div className="space-y-2 rounded-xl border border-[var(--border-hairline)] bg-[var(--bg-elevated)] p-4">
            <div className="flex items-center gap-2 text-sm font-semibold text-[var(--text-primary)]">
              <ShieldAlert className="h-4 w-4 text-amber-400" />
              <span>{t("about.limitsTitle")}</span>
            </div>
            <ul className="list-disc space-y-1.5 pl-5 text-xs text-[var(--text-secondary)]">
              <li>{t("about.limit1")}</li>
              <li>{t("about.limit2")}</li>
              <li>{t("about.limit3")}</li>
              <li>{t("about.limit4")}</li>
              <li>{t("about.limit5")}</li>
            </ul>
          </div>
        </CardContent>
      </Card>

      <Card className="lg:col-span-5">
        <CardHeader>
          <div className="flex items-center gap-2">
            <Scale className="h-4 w-4 text-[var(--accent)]" />
            <CardTitle>{t("about.licenseTitle")}</CardTitle>
          </div>
        </CardHeader>
        <CardContent className="space-y-4 text-sm text-[var(--text-secondary)]">
          <p>{t("about.licenseBody")}</p>
          <p className="font-mono text-xs text-[var(--text-muted)]">
            Copyright (C) 2026 Emin Burak Öztürk — GNU AGPL-3.0 (LICENSE)
          </p>
          <div className="flex flex-col gap-2">
            <a
              href="https://github.com/eburakozturk/tracera"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center justify-between rounded-lg border border-[var(--border-hairline)] bg-[var(--bg-elevated)] px-3.5 py-2.5 text-xs font-medium text-[var(--text-primary)] hover:border-[var(--accent)]"
            >
              <span>{t("about.githubLink")}</span>
              <ExternalLink className="h-3.5 w-3.5 text-[var(--accent)]" />
            </a>
            <a
              href="https://huggingface.co/spaces/eburakozturk/tracera"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center justify-between rounded-lg border border-[var(--border-hairline)] bg-[var(--bg-elevated)] px-3.5 py-2.5 text-xs font-medium text-[var(--text-primary)] hover:border-[var(--accent)]"
            >
              <span>{t("about.hfLink")}</span>
              <ExternalLink className="h-3.5 w-3.5 text-[var(--accent)]" />
            </a>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
