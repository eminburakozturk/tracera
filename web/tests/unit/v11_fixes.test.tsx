import fs from "node:fs";
import path from "node:path";
import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { HomeView } from "../../src/components/HomeView";
import { PrMiniChart } from "../../src/components/PrMiniChart";
import { METRICS_FALLBACK } from "../../src/fixtures";
import i18n from "../../src/i18n";
import { MOCK_ENABLED } from "../../src/lib/env";
import { fmtMeanStd, fmtMs, fmtNum, fmtPct } from "../../src/lib/format";
import { selectOpenVinoInt8Row } from "../../src/lib/metrics";
import { useAppStore } from "../../src/store/useAppStore";
import type { MetricsPayload, PrCurveData } from "../../src/types";

const webRoot = path.resolve(__dirname, "../..");
const PR_CURVE = JSON.parse(
  fs.readFileSync(path.join(webRoot, "public/data/pr_curve_val.json"), "utf-8"),
) as PrCurveData;
const METRICS = METRICS_FALLBACK as unknown as MetricsPayload;

describe("v1.1 düzeltmeleri (Claude incelemesi, 2026-11-14)", () => {
  beforeEach(async () => {
    await i18n.changeLanguage("tr");
    useAppStore.setState({ lang: "tr", metrics: METRICS });
  });

  it("Sayı biçimi: TR virgül ve önde yüzde, EN nokta ve sonda yüzde", () => {
    expect(fmtNum(0.395, 3, "tr")).toBe("0,395");
    expect(fmtPct(0.831, 1, "tr")).toBe("%83,1");
    expect(fmtMs(36.7, 1, "tr")).toBe("36,7 ms");
    expect(fmtMeanStd(0.858, 0.152, 3, "tr")).toBe("0,858 ± 0,152");
    expect(fmtNum(0.395, 3, "en")).toBe("0.395");
    expect(fmtPct(0.831, 1, "en")).toBe("83.1%");
    expect(fmtNum(Number.NaN, 2, "tr")).toBe("—");
    // Binlik ayırıcı kullanılmaz (2544 px gibi değerler bozulmasın)
    expect(fmtNum(2544, 0, "tr")).toBe("2544");
  });

  it("Ana Sayfa hız kartı YALNIZ 1024 OpenVINO INT8 satırından beslenir (640'ın 17,4 ms'i değil)", () => {
    const row = selectOpenVinoInt8Row(METRICS.edge_benchmark);
    expect(row).toBeDefined();
    expect(row!.imgsz).toBe(1024);
    expect(row!.format).toBe("OpenVINO INT8 (CPU)");

    render(
      <HomeView onSelectExample={async () => {}} onSelectFile={async () => {}} />,
    );
    const speed = screen.getByTestId("home-metric-speed");
    expect(speed).toHaveTextContent(fmtMs(row!.gecikme_inf_medyan_ms, 1, "tr"));
    expect(speed).not.toHaveTextContent("17,4");
    expect(speed).not.toHaveTextContent("17.4");
  });

  it("Ana Sayfa metrikleri veri yoksa uydurma yedek sayı değil '—' gösterir", () => {
    useAppStore.setState({ metrics: null });
    render(
      <HomeView onSelectExample={async () => {}} onSelectFile={async () => {}} />,
    );
    for (const id of ["home-metric-cv", "home-metric-test", "home-metric-speed"]) {
      expect(screen.getByTestId(id)).toHaveTextContent("—");
    }
    const text = document.body.textContent ?? "";
    for (const fake of ["0.493", "145.7", "54.7", "0.8101"]) {
      expect(text).not.toContain(fake);
    }
  });

  it("PR mini grafiği: tespit olmayan eşikte kesinlik 0 göstermez", () => {
    const { unmount } = render(
      <PrMiniChart confThreshold={0.92} initialData={PR_CURVE} />,
    );
    expect(screen.getByTestId("pr-no-detections")).toBeInTheDocument();
    expect(screen.getByTestId("pr-mini-chart")).not.toHaveTextContent("P %0,0");
    unmount();

    render(<PrMiniChart confThreshold={0.395} initialData={PR_CURVE} />);
    expect(screen.queryByTestId("pr-no-detections")).not.toBeInTheDocument();
    expect(screen.getByTestId("pr-mini-chart")).toHaveTextContent("%97,0");
    expect(screen.getByTestId("pr-mini-chart")).toHaveTextContent("Kesinlik");
  });

  it("Mock motoru test modunda açık (üretim derlemesinde kapalı; bkz. lib/env.ts)", () => {
    expect(MOCK_ENABLED).toBe(true);
  });

  it("Stitch kökenli ve ham ifadeler TR arayüz metinlerinde yok", () => {
    const tr = fs.readFileSync(path.join(webRoot, "src/i18n/tr.json"), "utf-8");
    for (const bad of [
      "Hassas Optik",
      "Hassasiyet–Duyarlılık",
      "Fixture",
      "(Mock)",
      "%{{pct}}'i",
    ]) {
      expect(tr.includes(bad), `tr.json içinde "${bad}" var`).toBe(false);
    }
    const about = fs.readFileSync(
      path.join(webRoot, "src/components/AboutView.tsx"),
      "utf-8",
    );
    // LinkedIn: href tam adres, görünen metin okunaklı
    expect(about).toContain(
      'href="https://www.linkedin.com/in/emin-burak-%C3%B6zt%C3%BCrk-85a674395/"',
    );
    expect(about).toContain("linkedin.com/in/emin-burak-öztürk");
  });
});
