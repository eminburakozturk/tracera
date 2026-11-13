import fs from "node:fs";
import path from "node:path";
import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { LiveView } from "../../src/components/LiveView";
import { ResultsView } from "../../src/components/ResultsView";
import { METRICS_FALLBACK } from "../../src/fixtures";
import { useAppStore } from "../../src/store/useAppStore";
import type { ErrorAnalysisPayload } from "../../src/types";

const ANALYSIS_PATH = path.resolve(__dirname, "../../public/analysis.json");
const SAMPLE_ANALYSIS = JSON.parse(
  fs.readFileSync(ANALYSIS_PATH, "utf-8"),
) as ErrorAnalysisPayload;

describe("LiveView & Error Analysis Gallery", () => {
  beforeEach(() => {
    useAppStore.setState({
      lang: "tr",
      metrics: METRICS_FALLBACK,
      metricsSource: "static",
      analysis: SAMPLE_ANALYSIS,
      ortModelCached: false,
      ortProgress: { stage: "idle", percent: 0, message: "" },
    });
  });

  it("renders LiveView with experimental camera disclaimer, ~65 MB notice, and canvas overlay", () => {
    render(<LiveView />);
    expect(screen.getByTestId("live-camera-note")).toHaveTextContent(
      "Deneysel — model yukarıdan çekilmiş yüksek çözünürlüklü kart fotoğraflarıyla eğitildi; telefon kamerasında doğruluk düşebilir.",
    );
    expect(screen.getByTestId("live-download-notice")).toHaveTextContent("~65 MB");
    expect(screen.getByTestId("live-video")).toBeInTheDocument();
    expect(screen.getByTestId("live-overlay-canvas")).toBeInTheDocument();
    expect(screen.getByTestId("live-pause-btn")).toBeInTheDocument();
  });

  it("renders Error Analysis Gallery dynamically from analysis.json", () => {
    render(<ResultsView />);
    const section = screen.getByTestId("error-analysis-section");
    expect(section).toBeInTheDocument();
    expect(section).toHaveTextContent("158");

    const fpTable = screen.getByTestId("fp-matrix-table");
    expect(fpTable).toHaveTextContent("138");
    expect(fpTable).toHaveTextContent("06");
    expect(fpTable).toHaveTextContent("10");

    const mbSpot = screen.getByTestId("template-spot-mouse_bite");
    expect(mbSpot).toHaveTextContent("Evet (0.5569)");
    expect(mbSpot).toHaveTextContent("0.4406–0.6331");

    const spurSpot = screen.getByTestId("template-spot-spur");
    expect(spurSpot).toHaveTextContent("Hayır (<0.395)");

    const quantTable = screen.getByTestId("quant-parity-table");
    expect(quantTable).toHaveTextContent("0.395");
    expect(quantTable).toHaveTextContent("43");
  });
});
