import * as React from "react";
import { Calendar, ExternalLink, Mail, Scale, ShieldAlert } from "lucide-react";
import { useTranslation } from "react-i18next";
import { getApprovedTimelineItems } from "../lib/timeline";
import { getClassColor, getClassDisplayName } from "../lib/v11Rules";
import { useAppStore } from "../store/useAppStore";
import type { TimelineData } from "../types";
import { Badge } from "./ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./ui/card";

interface AboutViewProps {
  initialTimeline?: TimelineData | null;
}

export function AboutView({ initialTimeline }: AboutViewProps) {
  const { t } = useTranslation();
  const lang = useAppStore((s) => s.lang);
  const meta = useAppStore((s) => s.meta);
  const [timeline, setTimeline] = React.useState<TimelineData | null>(
    initialTimeline ?? null,
  );

  React.useEffect(() => {
    if (initialTimeline !== undefined) {
      setTimeline(initialTimeline);
      return;
    }
    let active = true;
    void fetch("/data/timeline.json")
      .then(async (res) =>
        res.ok ? ((await res.json()) as TimelineData) : null,
      )
      .then((data) => {
        if (active && data) {
          setTimeline(data);
        }
      })
      .catch(() => {});
    return () => {
      active = false;
    };
  }, [initialTimeline]);

  const approvedItems = React.useMemo(
    () => getApprovedTimelineItems(timeline),
    [timeline],
  );

  return (
    <div className="space-y-5" data-testid="developer-view">
      {/* Geliştirici Kartı (Fotoğraf / Telefon YOK; baş harf avatarı EB, e-posta mailto) */}
      <Card data-testid="developer-profile-card">
        <CardHeader>
          <CardTitle>{t("developer.title")}</CardTitle>
          <CardDescription>{t("developer.roleNote")}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-[var(--border-hairline)] bg-[var(--bg-elevated)] p-4">
            <div className="flex items-center gap-3.5">
              <div
                aria-hidden="true"
                data-testid="developer-avatar"
                className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl border border-[var(--accent)]/40 bg-[var(--accent-soft)] font-mono text-sm font-bold text-[var(--accent-ink)]"
              >
                EB
              </div>
              <div>
                <h3 className="text-base font-bold text-[var(--text-primary)]">
                  Emin Burak Öztürk
                </h3>
                <p className="text-xs font-medium text-[var(--text-secondary)]">
                  Yapay Zeka Operatörlüğü (Önlisans), Balıkesir Üniversitesi
                  Balıkesir MYO
                </p>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <a
                href="mailto:eminburak1410@gmail.com"
                data-testid="developer-email-link"
                className="inline-flex items-center gap-1.5 rounded-lg border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-3 py-2 text-xs font-medium text-[var(--text-primary)] hover:border-[var(--accent)]"
              >
                <Mail className="h-3.5 w-3.5 shrink-0 text-[var(--accent-ink)]" />
                <span>eminburak1410@gmail.com</span>
              </a>
              <a
                href="https://github.com/eminburakozturk"
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1.5 rounded-lg border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-3 py-2 text-xs font-medium text-[var(--text-primary)] hover:border-[var(--accent)]"
              >
                <span>github.com/eminburakozturk</span>
                <ExternalLink className="h-3.5 w-3.5 shrink-0 text-[var(--accent-ink)]" />
              </a>
              <a
                href="https://www.linkedin.com/in/emin-burak-%C3%B6zt%C3%BCrk-85a674395/"
                target="_blank"
                rel="noreferrer"
                data-testid="developer-linkedin-link"
                className="inline-flex max-w-full items-center gap-1.5 rounded-lg border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-3 py-2 text-xs font-medium break-all text-[var(--text-primary)] hover:border-[var(--accent)]"
              >
                <span>
                  https://www.linkedin.com/in/emin-burak-%C3%B6zt%C3%BCrk-85a674395/
                </span>
                <ExternalLink className="h-3.5 w-3.5 shrink-0 text-[var(--accent-ink)]" />
              </a>
              <a
                href="https://huggingface.co/ebboo"
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1.5 rounded-lg border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-3 py-2 text-xs font-medium text-[var(--text-primary)] hover:border-[var(--accent)]"
              >
                <span>huggingface.co/ebboo</span>
                <ExternalLink className="h-3.5 w-3.5 shrink-0 text-[var(--accent-ink)]" />
              </a>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Proje Zaman Çizelgesi — YALNIZ status === "approved" ise ve metne dokunmadan */}
      {approvedItems.length > 0 && (
        <Card data-testid="developer-timeline-card">
          <CardHeader>
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <Calendar className="h-4 w-4 text-[var(--accent-ink)]" />
                <CardTitle>{t("developer.timelineTitle")}</CardTitle>
              </div>
              <Badge variant="secondary" className="font-mono">
                {t("developer.timelineSource")}
              </Badge>
            </div>
          </CardHeader>
          <CardContent>
            <ol className="relative space-y-3 border-l border-[var(--border-hairline)] pl-4">
              {approvedItems.map((item) => (
                <li
                  key={item.id}
                  data-testid={`timeline-item-${item.id}`}
                  className="rounded-xl border border-[var(--border-hairline)] bg-[var(--bg-elevated)] p-3.5"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="text-xs font-bold text-[var(--text-primary)]">
                      {item.title}
                    </span>
                    <Badge variant="default" className="font-mono text-[11px]">
                      {item.date}
                    </Badge>
                  </div>
                  <p className="mt-1 text-xs leading-relaxed text-[var(--text-secondary)]">
                    {item.summary}
                  </p>
                </li>
              ))}
            </ol>
          </CardContent>
        </Card>
      )}

      {/* Proje Hakkında, Sınırlılıklar ve AGPL-3.0 Lisansı */}
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
                    style={{
                      backgroundColor: meta.colors[i] ?? getClassColor(cls),
                    }}
                  />
                  <span>{getClassDisplayName(cls, lang)}</span>
                </Badge>
              ))}
            </div>

            <div className="space-y-2 rounded-xl border border-[var(--border-hairline)] bg-[var(--bg-elevated)] p-4">
              <div className="flex items-center gap-2 text-sm font-semibold text-[var(--text-primary)]">
                <ShieldAlert className="h-4 w-4 text-amber-500" />
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
              <Scale className="h-4 w-4 text-[var(--accent-ink)]" />
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
                href="https://github.com/eminburakozturk/tracera"
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center justify-between rounded-lg border border-[var(--border-hairline)] bg-[var(--bg-elevated)] px-3.5 py-2.5 text-xs font-medium text-[var(--text-primary)] hover:border-[var(--accent)]"
              >
                <span>{t("about.githubLink")}</span>
                <ExternalLink className="h-3.5 w-3.5 text-[var(--accent-ink)]" />
              </a>
              <a
                href="https://huggingface.co/spaces/ebboo/tracera-web"
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center justify-between rounded-lg border border-[var(--border-hairline)] bg-[var(--bg-elevated)] px-3.5 py-2.5 text-xs font-medium text-[var(--text-primary)] hover:border-[var(--accent)]"
              >
                <span>{t("about.hfLink")}</span>
                <ExternalLink className="h-3.5 w-3.5 text-[var(--accent-ink)]" />
              </a>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

export const DeveloperView = AboutView;
