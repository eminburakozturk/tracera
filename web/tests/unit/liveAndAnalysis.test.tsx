import fs from "node:fs";
import path from "node:path";
import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LiveView } from "../../src/components/LiveView";
import { ResultsView } from "../../src/components/ResultsView";
import { METRICS_FALLBACK, VAL_EXAMPLES } from "../../src/fixtures";
import { runDeviceBenchmark } from "../../src/lib/deviceBenchmark";
import { FALLBACK_META } from "../../src/lib/meta";
import { useAppStore } from "../../src/store/useAppStore";
import type { ErrorAnalysisPayload } from "../../src/types";

const ANALYSIS_PATH = path.resolve(__dirname, "../../public/analysis.json");
const SAMPLE_ANALYSIS = JSON.parse(
  fs.readFileSync(ANALYSIS_PATH, "utf-8"),
) as ErrorAnalysisPayload;

describe("LiveView, Error Analysis Gallery & Device Benchmark", () => {
  beforeEach(() => {
    useAppStore.setState({
      lang: "tr",
      meta: FALLBACK_META,
      metrics: METRICS_FALLBACK,
      metricsSource: "static",
      analysis: SAMPLE_ANALYSIS,
      ortModelCached: false,
      ortProgress: { stage: "idle", percent: 0, message: "" },
    });
  });

  it("renders LiveView with camera note, ~65 MB notice, canvas overlay, and WASM slow warning without WebGPU", () => {
    render(<LiveView />);
    expect(screen.getByTestId("live-camera-note")).toHaveTextContent(
      "Deneysel — model yukarıdan çekilmiş yüksek çözünürlüklü kart fotoğraflarıyla eğitildi; telefon kamerasında doğruluk düşebilir.",
    );
    expect(screen.getByTestId("live-wasm-warning")).toHaveTextContent(
      "WebGPU yok; canlı mod bu cihazda çok yavaş (<1 FPS) çalışır.",
    );
    expect(screen.getByTestId("live-download-notice")).toHaveTextContent("~65 MB");
    expect(screen.getByTestId("live-video")).toBeInTheDocument();
    expect(screen.getByTestId("live-overlay-canvas")).toBeInTheDocument();
    expect(screen.getByTestId("live-pause-btn")).toBeInTheDocument();
  });

  it("renders Error Analysis Gallery with separated Test (142) and Val (16) FP totals", () => {
    render(<ResultsView />);
    const section = screen.getByTestId("error-analysis-section");
    expect(section).toBeInTheDocument();
    expect(section).not.toHaveTextContent("158");

    expect(screen.getByTestId("fp-total-test")).toHaveTextContent("142");
    expect(screen.getByTestId("fp-total-val")).toHaveTextContent("16");

    const mbSpot = screen.getByTestId("template-spot-mouse_bite");
    expect(mbSpot).toHaveTextContent("Evet (0.5569)");
    expect(mbSpot).toHaveTextContent("0.4406–0.6331");

    const spurSpot = screen.getByTestId("template-spot-spur");
    expect(spurSpot).toHaveTextContent("Hayır (<0.395)");

    const quantTable = screen.getByTestId("quant-parity-table");
    expect(quantTable).toHaveTextContent("0.395");
    expect(quantTable).toHaveTextContent("43");

    expect(screen.getByTestId("device-bench-btn")).toHaveTextContent(
      "Bu cihazda hız testi",
    );
  });

  it("runs 4 val images × 5 repeats after 1 warmup and reports medians cleanly", async () => {
    const modelTimings = [
      999, // warmup (excluded from median)
      10, 20, 30, 40,
      12, 22, 32, 42,
      14, 24, 34, 44,
      16, 26, 36, 46,
      18, 28, 38, 48,
    ];
    let idx = 0;
    const detectSpy = vi.fn(async () => {
      const m = modelTimings[idx++] ?? 30;
      return {
        width: 1024,
        height: 1024,
        detections: [],
        timing: { pre_ms: 5, model_ms: m, post_ms: 2, total_ms: m + 7 },
        backend: "ort-webgpu",
      };
    });
    const loadFileSpy = vi.fn(
      async (item: { filename: string }) =>
        new File([new Uint8Array([1, 2, 3])], item.filename, {
          type: "image/jpeg",
        }),
    );
    const progressCalls: Array<[number, number]> = [];

    const report = await runDeviceBenchmark({
      meta: FALLBACK_META,
      examples: VAL_EXAMPLES,
      repeats: 5,
      detector: { detect: detectSpy },
      loadExampleFile: loadFileSpy,
      onProgress: (done, total) => progressCalls.push([done, total]),
    });

    expect(detectSpy).toHaveBeenCalledTimes(21); // 1 warmup + 4x5
    expect(report.images).toBe(4);
    expect(report.repeats).toBe(5);
    expect(report.warmup_runs).toBe(1);
    expect(report.timed_runs).toBe(20);
    expect(report.backend).toBe("ort-webgpu");
    expect(report.median_model_ms).toBe(29);
    expect(report.median_total_ms).toBe(36);
    expect(report.user_agent.length).toBeGreaterThan(0);
    expect(progressCalls[progressCalls.length - 1]).toEqual([21, 21]);
  });
});
