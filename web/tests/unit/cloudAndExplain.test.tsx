import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "../../src/App";
import { DetectView } from "../../src/components/DetectView";
import { ExplainView } from "../../src/components/ExplainView";
import { METRICS_FALLBACK } from "../../src/fixtures";
import { ApiClientError } from "../../src/inference/api/ApiDetector";
import {
  CLOUD_UNAVAILABLE_MESSAGE,
  CloudUnavailableError,
  DEFAULT_CLOUD_SPACE,
  DEFAULT_CLOUD_TIMEOUT_MS,
  callCloudEndpoint,
} from "../../src/inference/cloud/CloudClient";
import { CloudDetector } from "../../src/inference/cloud/CloudDetector";
import { CloudExplainer } from "../../src/inference/cloud/CloudExplainer";
import { resolveExplain } from "../../src/inference/explainResolver";
import type { Explainer } from "../../src/inference/types";
import { useAppStore } from "../../src/store/useAppStore";
import type { DetectResult, Detection, ExplainResult } from "../../src/types";

const sampleDet: Detection = {
  id: 1,
  cls_id: 0,
  cls: "missing_hole",
  conf: 0.91,
  box: [100, 120, 180, 200],
};

const sampleFile = new File([new Uint8Array([1, 2, 3])], "board_05.jpg", {
  type: "image/jpeg",
});

const fakeExplain: ExplainResult = {
  overlay_png: "AAAA",
  zoom_png: "BBBB",
  method: "layercam",
  layer: 4,
  timing: { pre_ms: 1, model_ms: 10, post_ms: 1, total_ms: 12 },
};

const fakeDetect: DetectResult = {
  width: 640,
  height: 480,
  detections: [sampleDet],
  timing: { pre_ms: 1, model_ms: 20, post_ms: 1, total_ms: 22 },
  backend: "cloud:ebboo/tracera",
};

describe("Cloud inference & ORT explain fallback", () => {
  beforeEach(() => {
    useAppStore.setState({
      metrics: METRICS_FALLBACK,
      lang: "tr",
      backendChoice: "ort",
      cloudSpaceId: DEFAULT_CLOUD_SPACE,
      selectedImage: { file: sampleFile, url: "blob:test", name: "board_05.jpg" },
      detectResult: fakeDetect,
      selectedDetectionId: 1,
      explainResult: fakeExplain,
      explainSource: "cloud",
      explainError: null,
      explainLoading: false,
    });
  });

  it("wraps private/unreachable HF Space errors with explicit Turkish unavailable message", async () => {
    const failingConnector = vi.fn().mockRejectedValue(new Error("404 Private Space"));
    const cloudExplainer = new CloudExplainer(DEFAULT_CLOUD_SPACE, {
      connector: failingConnector,
    });
    const cloudDetector = new CloudDetector(DEFAULT_CLOUD_SPACE, {
      connector: failingConnector,
    });

    await expect(cloudExplainer.explain(sampleFile, sampleDet)).rejects.toThrow(
      CLOUD_UNAVAILABLE_MESSAGE,
    );
    await expect(cloudDetector.detect(sampleFile, 0.395)).rejects.toThrow(
      CLOUD_UNAVAILABLE_MESSAGE,
    );
    expect(failingConnector).toHaveBeenCalledWith("ebboo/tracera");
  });

  it("calls /detect_api and /explain_api on configured Space when accessible", async () => {
    const predict = vi.fn(async (endpoint: string | number) => {
      if (endpoint === "/detect_api") return { data: [fakeDetect] };
      if (endpoint === "/explain_api") return { data: [fakeExplain] };
      throw new Error("unexpected endpoint");
    });
    const connector = vi.fn(async () => ({ predict }));

    const detector = new CloudDetector("ebboo/tracera", { connector });
    const detRes = await detector.detect(sampleFile, 0.395);
    expect(detRes.detections).toHaveLength(1);
    expect(predict).toHaveBeenCalledWith(
      "/detect_api",
      expect.objectContaining({ conf: 0.395 }),
    );

    const explainer = new CloudExplainer("ebboo/tracera", { connector });
    const expRes = await explainer.explain(sampleFile, sampleDet);
    expect(expRes.overlay_png).toBe("AAAA");
    expect(predict).toHaveBeenCalledWith(
      "/explain_api",
      expect.objectContaining({ box: "100.00,120.00,180.00,200.00", cls_id: 0 }),
    );
  });

  it("tries local API first when backendChoice is ort, and does NOT touch cloud if API succeeds", async () => {
    const apiExplain = vi.fn<Explainer["explain"]>().mockResolvedValue(fakeExplain);
    const cloudExplain = vi.fn<Explainer["explain"]>().mockResolvedValue(fakeExplain);
    const mockExplain = vi.fn<Explainer["explain"]>().mockResolvedValue(fakeExplain);
    const onApiStatusChange = vi.fn();

    const out = await resolveExplain({
      backendChoice: "ort",
      apiBaseUrl: "http://127.0.0.1:8765",
      cloudSpaceId: DEFAULT_CLOUD_SPACE,
      file: sampleFile,
      detection: sampleDet,
      onApiStatusChange,
      factories: {
        api: () => ({ explain: apiExplain }),
        cloud: () => ({ explain: cloudExplain }),
        mock: () => ({ explain: mockExplain }),
      },
    });

    expect(out.source).toBe("api");
    expect(apiExplain).toHaveBeenCalledTimes(1);
    expect(cloudExplain).not.toHaveBeenCalled();
    expect(mockExplain).not.toHaveBeenCalled();
    expect(onApiStatusChange).toHaveBeenCalledWith("online");
  });

  it("falls back to Cloud (never Mock) when ORT local API is offline", async () => {
    const apiExplain = vi
      .fn<Explainer["explain"]>()
      .mockRejectedValue(new ApiClientError("Bağlantı kurulamadı", { isOffline: true }));
    const cloudExplain = vi.fn<Explainer["explain"]>().mockResolvedValue(fakeExplain);
    const mockExplain = vi.fn<Explainer["explain"]>().mockResolvedValue(fakeExplain);
    const onApiStatusChange = vi.fn();

    const out = await resolveExplain({
      backendChoice: "ort",
      apiBaseUrl: "http://127.0.0.1:8765",
      cloudSpaceId: DEFAULT_CLOUD_SPACE,
      file: sampleFile,
      detection: sampleDet,
      onApiStatusChange,
      factories: {
        api: () => ({ explain: apiExplain }),
        cloud: () => ({ explain: cloudExplain }),
        mock: () => ({ explain: mockExplain }),
      },
    });

    expect(out.source).toBe("cloud");
    expect(apiExplain).toHaveBeenCalledTimes(1);
    expect(cloudExplain).toHaveBeenCalledTimes(1);
    expect(mockExplain).not.toHaveBeenCalled();
    expect(onApiStatusChange).toHaveBeenCalledWith("offline", "Bağlantı kurulamadı");
  });

  it("surfaces CloudUnavailableError instead of silently falling back to Mock when both API and Cloud fail", async () => {
    const apiExplain = vi
      .fn<Explainer["explain"]>()
      .mockRejectedValue(new ApiClientError("Bağlantı kurulamadı", { isOffline: true }));
    const cloudExplain = vi
      .fn<Explainer["explain"]>()
      .mockRejectedValue(new CloudUnavailableError(CLOUD_UNAVAILABLE_MESSAGE));
    const mockExplain = vi.fn<Explainer["explain"]>().mockResolvedValue(fakeExplain);

    await expect(
      resolveExplain({
        backendChoice: "ort",
        apiBaseUrl: "http://127.0.0.1:8765",
        cloudSpaceId: DEFAULT_CLOUD_SPACE,
        file: sampleFile,
        detection: sampleDet,
        factories: {
          api: () => ({ explain: apiExplain }),
          cloud: () => ({ explain: cloudExplain }),
          mock: () => ({ explain: mockExplain }),
        },
      }),
    ).rejects.toThrow(CLOUD_UNAVAILABLE_MESSAGE);

    expect(mockExplain).not.toHaveBeenCalled();
  });

  it("renders explain-source-badge in ExplainView showing which backend answered", () => {
    render(<ExplainView onRunExplain={async () => {}} />);
    const badge = screen.getByTestId("explain-source-badge");
    expect(badge).toHaveTextContent("Bulut (ebboo/tracera)");
  });

  it("renders cloud-only note once and uses selected detection copy in ExplainView", () => {
    act(() => {
      useAppStore.setState({ localApiAvailable: false, backendChoice: "ort" });
    });
    render(<ExplainView onRunExplain={async () => {}} />);
    const notes = screen.getAllByText(
      "Bu cihazda ısı haritası bulutta hesaplanır (internet gerekir).",
    );
    expect(notes).toHaveLength(1);
    expect(screen.getByTestId("explain-cloud-note")).toBeInTheDocument();
    expect(screen.queryByText(/anchor/i)).not.toBeInTheDocument();
    expect(screen.getByText(/seçilen tespitin/i)).toBeInTheDocument();
  });

  it("sets cloud timeout >= 60s, calls onWaking, and retries once automatically on transient failure", async () => {
    expect(DEFAULT_CLOUD_TIMEOUT_MS).toBeGreaterThanOrEqual(60_000);

    let attempts = 0;
    const predict = vi.fn(async () => ({ data: [fakeExplain] }));
    const connector = vi.fn(async () => {
      attempts++;
      if (attempts === 1) {
        throw new Error("ZeroGPU cold start 503");
      }
      return { predict };
    });
    const onWaking = vi.fn();

    const out = await callCloudEndpoint<ExplainResult>(
      DEFAULT_CLOUD_SPACE,
      "/explain_api",
      { box: "0,0,10,10", cls_id: 0 },
      { connector, retries: 1, retryDelayMs: 0, wakingThresholdMs: 10, onWaking },
    );

    expect(out).toEqual(fakeExplain);
    expect(connector).toHaveBeenCalledTimes(2);
    expect(onWaking).toHaveBeenCalledTimes(1);
  });

  it("enforces explain state transitions: error->success, success->error, detection change", async () => {
    const det2: Detection = {
      id: 2,
      cls_id: 3,
      cls: "short",
      conf: 0.84,
      box: [10, 20, 40, 60],
    };

    useAppStore.setState({
      detectResult: { ...fakeDetect, detections: [sampleDet, det2] },
      selectedDetectionId: 1,
      explainResult: null,
      explainSource: null,
      explainDetectionId: null,
      explainStatus: "idle",
      explainLoading: false,
      explainError: null,
    });

    let mode: "fail" | "ok" = "fail";
    const onRunExplain = vi.fn(async () => {
      const st = useAppStore.getState();
      const detId = st.selectedDetectionId ?? 1;
      st.startExplainRequest(detId);
      if (mode === "fail") {
        st.setExplainError(CLOUD_UNAVAILABLE_MESSAGE);
      } else {
        st.setExplainResult(fakeExplain, "cloud", detId);
      }
    });

    render(<ExplainView onRunExplain={onRunExplain} />);

    // 1) error -> success: hata önce banner açar, yeni başarılı istekte hata kaybolur
    await act(async () => {
      await onRunExplain();
    });
    expect(screen.getByTestId("explain-error-banner")).toHaveTextContent(
      CLOUD_UNAVAILABLE_MESSAGE,
    );
    expect(screen.queryByTestId("explain-overlay-img")).not.toBeInTheDocument();

    mode = "ok";
    await act(async () => {
      fireEvent.click(screen.getByTestId("generate-heatmap-btn"));
    });
    await waitFor(() => {
      expect(screen.getByTestId("explain-overlay-img")).toBeInTheDocument();
    });
    expect(screen.queryByTestId("explain-error-banner")).not.toBeInTheDocument();
    expect(screen.getByTestId("explain-detection-badge")).toHaveTextContent(
      "#1 · missing_hole",
    );

    // 2) success -> error: hata oluşunca eski harita kaldırılır (aynı anda görünmez)
    mode = "fail";
    await act(async () => {
      fireEvent.click(screen.getByTestId("generate-heatmap-btn"));
    });
    await waitFor(() => {
      expect(screen.getByTestId("explain-error-banner")).toBeInTheDocument();
    });
    expect(screen.queryByTestId("explain-overlay-img")).not.toBeInTheDocument();
    expect(screen.queryByTestId("explain-detection-badge")).not.toBeInTheDocument();

    // Tekrar başarı + 3) tespit değişimi: eski tespitin haritası ve hatası temizlenir
    mode = "ok";
    await act(async () => {
      fireEvent.click(screen.getByTestId("generate-heatmap-btn"));
    });
    await waitFor(() => {
      expect(screen.getByTestId("explain-overlay-img")).toBeInTheDocument();
    });

    act(() => {
      useAppStore.getState().setSelectedDetectionId(2);
    });
    expect(screen.queryByTestId("explain-overlay-img")).not.toBeInTheDocument();
    expect(screen.queryByTestId("explain-error-banner")).not.toBeInTheDocument();

    // Waking ara durumu UI metnini doğrula
    act(() => {
      useAppStore.getState().startExplainRequest(2);
      useAppStore.getState().setExplainWaking();
    });
    expect(screen.getByTestId("explain-status-banner")).toHaveTextContent(
      "Bulut sunucusu uyanıyor…",
    );
  });

  it("App reference stays importable without side effects in explain state tests", () => {
    expect(typeof App).toBe("function");
  });

  it("shows first-detect model download notice only when ORT model is uncached", () => {
    useAppStore.setState({ backendChoice: "ort", ortModelCached: false });
    const { rerender } = render(
      <DetectView
        onSelectExample={async () => {}}
        onSelectFile={async () => {}}
        onRunDetect={async () => {}}
        onInspectDetection={() => {}}
      />,
    );
    expect(screen.getByTestId("model-download-notice")).toHaveTextContent(
      "~65 MB",
    );

    act(() => {
      useAppStore.setState({ backendChoice: "ort", ortModelCached: true });
    });
    rerender(
      <DetectView
        onSelectExample={async () => {}}
        onSelectFile={async () => {}}
        onRunDetect={async () => {}}
        onInspectDetection={() => {}}
      />,
    );
    expect(screen.queryByTestId("model-download-notice")).not.toBeInTheDocument();
  });
});
