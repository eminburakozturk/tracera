import "fake-indexeddb/auto";
import fs from "node:fs";
import path from "node:path";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AboutView } from "../../src/components/AboutView";
import {
  type BatchQueueItem,
  runBatchQueueSequentially,
} from "../../src/components/BatchView";
import { DetectView } from "../../src/components/DetectView";
import { ExplainView } from "../../src/components/ExplainView";
import { GuideView } from "../../src/components/GuideView";
import { HistoryView } from "../../src/components/HistoryView";
import { HomeView } from "../../src/components/HomeView";
import {
  ONBOARDING_STORAGE_KEY,
  OnboardingModal,
} from "../../src/components/OnboardingModal";
import { METRICS_FALLBACK } from "../../src/fixtures";
import {
  clearAllHistoryRecords,
  listHistoryRecords,
  saveInspectionToHistory,
} from "../../src/lib/historyDb";
import { FALLBACK_META } from "../../src/lib/meta";
import { generateInspectionPdfBytes } from "../../src/lib/pdfReport";
import { buildReliabilitySummary } from "../../src/lib/reliability";
import { getApprovedTimelineItems } from "../../src/lib/timeline";
import {
  isLowConfidenceReview,
  LOW_CONF_REVIEW_LABEL_TR,
  LOW_CONF_REVIEW_TOOLTIP_TR,
  LOW_CONF_REVIEW_UPPER_BOUND,
} from "../../src/lib/v11Rules";
import { useAppStore } from "../../src/store/useAppStore";
import type {
  DetectResult,
  Detection,
  ExplainResult,
  ReliabilityData,
  TimelineData,
} from "../../src/types";

const FORBIDDEN_V11_STRINGS = [
  "IPC-A-610",
  "ISO 9001",
  "ISO/IEC 17025",
  "Sertifika",
  "Sertifikalı",
  "Kabul uyumluluğu",
  "mikron",
  "µm",
  "µm/px",
  "48 MPix",
  "FOV",
  "45 mikronluk köprü",
  "YOLOv8",
  "Conv5_3",
  "WebGPU FP16",
  "v2.4",
  "12.4 TFLOPS",
  "58 FPS",
  "97.4",
  "98.1",
  "0.964",
  "1.420 PCB",
  "99,92",
  "PCB-9842-X",
  "LOT-4089",
  "SMT-HAT-04",
  "Rev A2",
  "pin R14",
  "Baş Kalite Güvence",
  "Golden Board",
  "soğuk lehim",
  "eksik bileşen",
  "Bilgisayar Müh.",
  "YZ Araştırmacısı",
];

function collectFilesRecursively(dir: string): string[] {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  const files: string[] = [];
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...collectFilesRecursively(full));
    } else if (/\.(ts|tsx|json|css)$/i.test(entry.name)) {
      files.push(full);
    }
  }
  return files;
}

const VAL_RELIABILITY_PATH = path.resolve(
  __dirname,
  "../../public/data/reliability_val.json",
);
const TEST_RELIABILITY_PATH = path.resolve(
  __dirname,
  "../../public/data/reliability_test.json",
);
const TIMELINE_PATH = path.resolve(
  __dirname,
  "../../public/data/timeline.json",
);

const REAL_VAL_RELIABILITY = JSON.parse(
  fs.readFileSync(VAL_RELIABILITY_PATH, "utf-8"),
) as ReliabilityData;
const REAL_TEST_RELIABILITY = JSON.parse(
  fs.readFileSync(TEST_RELIABILITY_PATH, "utf-8"),
) as ReliabilityData;
const REAL_TIMELINE = JSON.parse(
  fs.readFileSync(TIMELINE_PATH, "utf-8"),
) as TimelineData;

describe("Tracera v1.1 UI Rules, Reliability, Timeline, Batch, History & PDF", () => {
  beforeEach(async () => {
    await clearAllHistoryRecords();
    useAppStore.setState({
      activeTab: "home",
      lang: "tr",
      theme: "dark",
      meta: FALLBACK_META,
      metrics: METRICS_FALLBACK,
      metricsSource: "static",
      confThreshold: 0.395,
    });
  });

  it("1) Yasaklı kelime grep testi: src/ ve i18n/ içinde hiçbir Stitch uydurması geçmez", () => {
    const srcDir = path.resolve(__dirname, "../../src");
    const files = collectFilesRecursively(srcDir);
    expect(files.length).toBeGreaterThan(10);

    for (const file of files) {
      const content = fs.readFileSync(file, "utf-8");
      for (const forbidden of FORBIDDEN_V11_STRINGS) {
        expect(
          content.includes(forbidden),
          `Yasaklı ifade "${forbidden}" bulundu: ${path.relative(srcDir, file)}`,
        ).toBe(false);
      }
    }
  });

  it("2) Reliability kartı testi: null/n=0 gizlenir, n<20 'az örnek' etiketi alır, Val+Test yan yana ve tek dosya yedeği çalışır", () => {
    // a) 0.95 güven aralığında n=0 ve precision=null -> null döner (cümle gizlenir)
    const emptyBinSummary = buildReliabilitySummary(
      0.95,
      REAL_VAL_RELIABILITY,
      REAL_TEST_RELIABILITY,
      "tr",
    );
    expect(emptyBinSummary).toBeNull();

    // b) 0.45 güven aralığında Val n=4 (<20 -> az örnek) ve Test n=78
    const lowSampleSummary = buildReliabilitySummary(
      0.45,
      REAL_VAL_RELIABILITY,
      REAL_TEST_RELIABILITY,
      "tr",
    );
    expect(lowSampleSummary).not.toBeNull();
    expect(lowSampleSummary!.valPart?.lowSample).toBe(true);
    expect(lowSampleSummary!.headlineText).toContain("Val: %100.0 (n=4)");
    expect(lowSampleSummary!.headlineText).toContain("Test: %12.8 (n=78)");

    // c) Yalnız tek dosya (ör. yalnız reliability_val.json) varsa hangi bölmeden olduğu açıkça yazılır
    const singleValSummary = buildReliabilitySummary(
      0.85,
      REAL_VAL_RELIABILITY,
      null,
      "tr",
    );
    expect(singleValSummary).not.toBeNull();
    expect(singleValSummary!.singleSplit).toBe("val");
    expect(singleValSummary!.headlineText).toContain("Val setinde");
    expect(singleValSummary!.headlineText).toContain("(n=240)");

    // d) ExplainView üzerinde n=0 iken reliability-sentence gizlenir, n<20 iken 'az örnek' rozeti ve energy_in_box çubuğu görünür
    const detHigh: Detection = {
      id: 1,
      cls: "short",
      cls_id: 3,
      conf: 0.95,
      box: [10, 20, 80, 90],
    };
    const expWithEnergy: ExplainResult = {
      overlay_png: "AAAA",
      zoom_png: "BBBB",
      method: "layercam",
      layer: 4,
      timing: { pre_ms: 1, model_ms: 10, post_ms: 1, total_ms: 12 },
      energy_in_box: 0.466,
      peak_in_box: true,
      peak_xy: [45, 55],
    };
    const dummyFile = new File(["abc"], "05_short.jpg", { type: "image/jpeg" });

    useAppStore.setState({
      selectedImage: { file: dummyFile, url: "blob:test", name: "05_short.jpg" },
      detectResult: {
        width: 1024,
        height: 1024,
        detections: [detHigh],
        timing: { pre_ms: 1, model_ms: 10, post_ms: 1, total_ms: 12 },
        backend: "ort-webgpu",
      },
      selectedDetectionId: 1,
      explainResult: expWithEnergy,
      explainSource: "api",
      explainDetectionId: 1,
      explainStatus: "success",
      explainLoading: false,
      explainError: null,
    });

    const { rerender } = render(
      <ExplainView
        onRunExplain={async () => {}}
        initialValReliability={REAL_VAL_RELIABILITY}
        initialTestReliability={REAL_TEST_RELIABILITY}
      />,
    );

    // conf=0.95 (bin 0.9-1.0, n=0) -> reliability cümlesi gizli
    expect(screen.queryByTestId("reliability-sentence")).not.toBeInTheDocument();
    // energy_in_box=0.466 -> yatay çubuk görünür
    expect(screen.getByTestId("energy-in-box-row")).toHaveTextContent(
      "Isı haritası enerjisinin %46.6'i kutunun içinde",
    );

    // conf=0.45 (bin 0.395-0.5, Val n=4 < 20) -> reliability cümlesi ve 'az örnek' rozeti görünür
    const detLowSample: Detection = { ...detHigh, conf: 0.45 };
    useAppStore.setState({
      detectResult: {
        width: 1024,
        height: 1024,
        detections: [detLowSample],
        timing: { pre_ms: 1, model_ms: 10, post_ms: 1, total_ms: 12 },
        backend: "ort-webgpu",
      },
    });
    rerender(
      <ExplainView
        onRunExplain={async () => {}}
        initialValReliability={REAL_VAL_RELIABILITY}
        initialTestReliability={REAL_TEST_RELIABILITY}
      />,
    );
    expect(screen.getByTestId("reliability-sentence")).toBeInTheDocument();
    expect(screen.getByTestId("reliability-low-sample-badge")).toHaveTextContent(
      "az örnek",
    );
  });

  it("3) Timeline status testi: yalnız status === 'approved' olduğunda gösterilir, aksi halde gizlenir", () => {
    expect(getApprovedTimelineItems(REAL_TIMELINE)).toHaveLength(11);
    expect(
      getApprovedTimelineItems({ ...REAL_TIMELINE, status: "draft" }),
    ).toEqual([]);
    expect(getApprovedTimelineItems(null)).toEqual([]);

    const { rerender } = render(<AboutView initialTimeline={REAL_TIMELINE} />);
    expect(screen.getByTestId("developer-avatar")).toHaveTextContent("EB");
    expect(screen.getByTestId("developer-email-link")).toHaveAttribute(
      "href",
      "mailto:eminburak1410@gmail.com",
    );
    expect(screen.getByTestId("developer-linkedin-link")).toHaveAttribute(
      "href",
      "https://www.linkedin.com/in/emin-burak-%C3%B6zt%C3%BCrk-85a674395/",
    );
    expect(screen.getByTestId("developer-timeline-card")).toBeInTheDocument();
    expect(screen.getByTestId("timeline-item-1")).toHaveTextContent(
      REAL_TIMELINE.items[0].title,
    );

    rerender(
      <AboutView initialTimeline={{ ...REAL_TIMELINE, status: "pending" }} />,
    );
    expect(
      screen.queryByTestId("developer-timeline-card"),
    ).not.toBeInTheDocument();
  });

  it("4) Toplu muayene (3 görsel) testi: sıralı kuyruk (eşzamanlılık=1) ile 3 görüntüyü işler", async () => {
    let activeWorkers = 0;
    let maxConcurrentWorkers = 0;

    const files = [
      new File(["1"], "card_05_a.jpg", { type: "image/jpeg" }),
      new File(["2"], "card_05_b.jpg", { type: "image/jpeg" }),
      new File(["3"], "card_11_c.jpg", { type: "image/jpeg" }),
    ];

    const queueItems: BatchQueueItem[] = files.map((file, i) => ({
      id: `q_${i}`,
      file,
      filename: file.name,
      status: "queued",
    }));

    const detectFn = vi.fn(async (file: File): Promise<DetectResult> => {
      activeWorkers++;
      maxConcurrentWorkers = Math.max(maxConcurrentWorkers, activeWorkers);
      await new Promise((r) => setTimeout(r, 10));
      activeWorkers--;

      const hasDefect = file.name !== "card_05_b.jpg";
      return {
        width: 1024,
        height: 1024,
        detections: hasDefect
          ? [
              {
                id: 0,
                cls: "short",
                cls_id: 3,
                conf: 0.88,
                box: [10, 10, 50, 50],
              },
            ]
          : [],
        timing: { pre_ms: 2, model_ms: 25, post_ms: 1, total_ms: 28 },
        backend: "mock",
      };
    });

    const updates: Array<{ id: string; patch: Partial<BatchQueueItem> }> = [];
    const results = await runBatchQueueSequentially({
      items: queueItems,
      confThreshold: 0.395,
      detectFn,
      onUpdateItem: (id, patch) => updates.push({ id, patch }),
    });

    expect(detectFn).toHaveBeenCalledTimes(3);
    expect(maxConcurrentWorkers).toBe(1); // Eşzamanlılık = 1
    expect(results.map((r) => r.status)).toEqual(["done", "done", "done"]);
    expect(results.map((r) => r.verdict)).toEqual(["FAIL", "PASS", "FAIL"]);
    expect(updates.length).toBeGreaterThanOrEqual(6);
  });

  it("5) Geçmişe kaydetme, listeleme, yan yana karşılaştırma ve geri açma testi (IndexedDB)", async () => {
    const file1 = new File(["img1"], "05_short_01.jpg", { type: "image/jpeg" });
    const file2 = new File(["img2"], "11_clean_02.jpg", { type: "image/jpeg" });

    const res1: DetectResult = {
      width: 1024,
      height: 1024,
      detections: [
        {
          id: 0,
          cls: "short",
          cls_id: 3,
          conf: 0.89,
          box: [100, 100, 180, 180],
        },
      ],
      timing: { pre_ms: 1, model_ms: 30, post_ms: 1, total_ms: 32 },
      backend: "ort-webgpu",
    };
    const res2: DetectResult = {
      width: 1024,
      height: 1024,
      detections: [],
      timing: { pre_ms: 1, model_ms: 28, post_ms: 1, total_ms: 30 },
      backend: "ort-webgpu",
    };

    const rec1 = await saveInspectionToHistory({
      idOverride: "rec_1",
      filename: file1.name,
      fileOrUrl: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=",
      confThreshold: 0.395,
      result: res1,
    });
    const rec2 = await saveInspectionToHistory({
      idOverride: "rec_2",
      filename: file2.name,
      fileOrUrl: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=",
      confThreshold: 0.395,
      result: res2,
    });

    const stored = await listHistoryRecords();
    expect(stored).toHaveLength(2);

    const onOpenRecord = vi.fn();
    render(<HistoryView onOpenRecord={onOpenRecord} />);

    await waitFor(() => {
      expect(screen.getByTestId(`history-card-${rec1.id}`)).toBeInTheDocument();
      expect(screen.getByTestId(`history-card-${rec2.id}`)).toBeInTheDocument();
    });

    // 2 kaydı yan yana karşılaştır
    fireEvent.click(screen.getByTestId(`history-compare-${rec1.id}`));
    fireEvent.click(screen.getByTestId(`history-compare-${rec2.id}`));
    expect(screen.getByTestId("history-compare-panel")).toBeInTheDocument();

    // Muayenede geri aç
    fireEvent.click(screen.getByTestId(`history-open-${rec1.id}`));
    expect(onOpenRecord).toHaveBeenCalledWith(
      expect.objectContaining({ id: rec1.id, filename: "05_short_01.jpg" }),
    );
  });

  it("6) PDF üretimi testi: boyut > 0, 'Kısa devre', 'KALDI' ve Türkçe karakterler (ğ ş ı İ ç ö ü) bozulmadan yer alır", async () => {
    const result: DetectResult = {
      width: 3056,
      height: 2464,
      detections: [
        {
          id: 0,
          cls: "short",
          cls_id: 3,
          conf: 0.912,
          box: [120, 140, 220, 230],
        },
        {
          id: 1,
          cls: "spur",
          cls_id: 4,
          conf: 0.745,
          box: [400, 500, 460, 560],
        },
      ],
      timing: { pre_ms: 4.1, model_ms: 37.3, post_ms: 1.5, total_ms: 42.9 },
      backend: "openvino-int8",
    };

    const pdfBytes = await generateInspectionPdfBytes({
      filename: "05_short_ğşıİçöü.jpg",
      meta: FALLBACK_META,
      confThreshold: 0.395,
      result,
      lang: "tr",
    });

    expect(pdfBytes.byteLength).toBeGreaterThan(10_000);

    const decoded = new TextDecoder("utf-8", { fatal: false }).decode(pdfBytes);
    expect(decoded.startsWith("%PDF-1.7")).toBe(true);
    expect(decoded).toContain("Kısa devre");
    expect(decoded).toContain("KALDI");
    expect(decoded).toContain("ğ ş ı İ ç ö ü");
    expect(decoded).toContain(
      "Bu rapor Tracera yazılımının otomatik çıktısıdır; herhangi bir standarda uygunluk belgesi değildir.",
    );
  });

  it("7) Ana Sayfa, Kusur Rehberi ve İlk Açılış Tanıtımı bileşenlerini hatasız render eder", () => {
    render(
      <HomeView
        onSelectExample={async () => {}}
        onSelectFile={async () => {}}
      />,
    );
    expect(screen.getByTestId("home-metric-cv")).toHaveTextContent(
      "0.858 ± 0.152",
    );
    expect(screen.getByTestId("home-metric-test")).toHaveTextContent("0.8101");

    render(<GuideView initialGuideIndex={null} initialClassMetrics={null} />);
    expect(screen.getByTestId("guide-dataset-note")).toHaveTextContent(
      "PKU-Market-PCB",
    );

    localStorage.removeItem(ONBOARDING_STORAGE_KEY);
    const onClose = vi.fn();
    render(<OnboardingModal open={true} onClose={onClose} />);
    expect(screen.getByTestId("onboarding-modal")).toBeInTheDocument();
    fireEvent.click(screen.getByTestId("onboarding-skip-btn"));
    expect(localStorage.getItem(ONBOARDING_STORAGE_KEY)).toBe("1");
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("8) Düşük güven: gözden geçirin etiketi (0.394 → etiket yok ve tespit gösterilmez; 0.45 → etiket var ve KALDI; 0.6 → etiket yok)", () => {
    expect(LOW_CONF_REVIEW_UPPER_BOUND).toBe(0.6);
    expect(isLowConfidenceReview(0.394, 0.395)).toBe(false);
    expect(isLowConfidenceReview(0.395, 0.395)).toBe(true);
    expect(isLowConfidenceReview(0.45, 0.395)).toBe(true);
    expect(isLowConfidenceReview(0.599, 0.395)).toBe(true);
    expect(isLowConfidenceReview(0.6, 0.395)).toBe(false);

    const dummyFile = new File(["pcb"], "06_spur.jpg", { type: "image/jpeg" });
    useAppStore.setState({
      confThreshold: 0.395,
      selectedImage: {
        file: dummyFile,
        url: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=",
        name: "06_spur.jpg",
      },
      detectResult: {
        width: 1024,
        height: 1024,
        detections: [
          {
            id: 1,
            cls: "spur",
            cls_id: 4,
            conf: 0.394,
            box: [10, 10, 40, 40],
          },
          {
            id: 2,
            cls: "spur",
            cls_id: 4,
            conf: 0.45,
            box: [50, 50, 90, 90],
          },
          {
            id: 3,
            cls: "short",
            cls_id: 3,
            conf: 0.6,
            box: [120, 120, 180, 180],
          },
        ],
        timing: { pre_ms: 1, model_ms: 15, post_ms: 1, total_ms: 17 },
        backend: "ort-webgpu",
      },
      selectedDetectionId: 2,
    });

    const { rerender } = render(
      <DetectView
        onSelectExample={async () => {}}
        onSelectFile={async () => {}}
        onRunDetect={async () => {}}
        onInspectDetection={() => {}}
      />,
    );

    // 0.394 -> tespit gösterilmez ve etiket yok
    expect(screen.queryByTestId("det-card-1")).not.toBeInTheDocument();
    expect(screen.queryByTestId("low-conf-badge-1")).not.toBeInTheDocument();

    // 0.45 -> tespit gösterilir, kusur kartında ve listede "Düşük güven: gözden geçirin" etiketi ve ipucu var
    expect(screen.getByTestId("det-card-2")).toBeInTheDocument();
    const listBadge = screen.getByTestId("low-conf-badge-2");
    expect(listBadge).toHaveTextContent(LOW_CONF_REVIEW_LABEL_TR);
    expect(listBadge).toHaveAttribute("title", LOW_CONF_REVIEW_TOOLTIP_TR);
    const selectedBadge = screen.getByTestId("selected-low-conf-badge");
    expect(selectedBadge).toHaveTextContent(LOW_CONF_REVIEW_LABEL_TR);
    expect(selectedBadge).toHaveAttribute("title", LOW_CONF_REVIEW_TOOLTIP_TR);

    // 0.6 -> tespit gösterilir ama düşük güven etiketi YOK
    expect(screen.getByTestId("det-card-3")).toBeInTheDocument();
    expect(screen.queryByTestId("low-conf-badge-3")).not.toBeInTheDocument();

    // Yalnız 0.45 tespiti varken de GEÇTİ/KALDI kuralı değişmez -> KALDI
    useAppStore.setState({
      detectResult: {
        width: 1024,
        height: 1024,
        detections: [
          {
            id: 2,
            cls: "spur",
            cls_id: 4,
            conf: 0.45,
            box: [50, 50, 90, 90],
          },
        ],
        timing: { pre_ms: 1, model_ms: 15, post_ms: 1, total_ms: 17 },
        backend: "ort-webgpu",
      },
    });
    rerender(
      <DetectView
        onSelectExample={async () => {}}
        onSelectFile={async () => {}}
        onRunDetect={async () => {}}
        onInspectDetection={() => {}}
      />,
    );
    expect(screen.getByTestId("verdict-badge")).toHaveTextContent("KALDI");
  });
});
