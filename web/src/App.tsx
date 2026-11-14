import * as React from "react";
import { AlertTriangle, Loader2 } from "lucide-react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { useTranslation } from "react-i18next";
import { AboutView } from "./components/AboutView";
import { BatchView, type BatchQueueItem } from "./components/BatchView";
import { DetectView } from "./components/DetectView";
import { ExplainView } from "./components/ExplainView";
import { HomeView } from "./components/HomeView";
import { LiveView } from "./components/LiveView";
import { MobileNav } from "./components/MobileNav";
import { MoreView } from "./components/MoreView";
import { ResultsView } from "./components/ResultsView";
import { Sidebar } from "./components/Sidebar";
import { TopBar } from "./components/TopBar";
import { VAL_EXAMPLES, type ValExampleItem } from "./fixtures";
import {
  ApiClientError,
  ApiDetector,
  checkApiHealth,
} from "./inference/api/ApiDetector";
import { CLOUD_UNAVAILABLE_MESSAGE } from "./inference/cloud/CloudClient";
import { CloudDetector } from "./inference/cloud/CloudDetector";
import { resolveExplain } from "./inference/explainResolver";
import { MockDetector } from "./inference/mock/MockDetector";
import {
  isOrtModelCached,
  OrtDetector,
  type OrtEpPreference,
} from "./inference/ort/OrtDetector";
import type { Detector } from "./inference/types";
import { loadAnalysis } from "./lib/analysis";
import { saveInspectionToHistory } from "./lib/historyDb";
import { loadModelMeta } from "./lib/meta";
import { loadMetrics } from "./lib/metrics";
import { generateInspectionPdfBytes, triggerPdfDownload } from "./lib/pdfReport";
import { connectTauriSidecar, isTauriApp } from "./platform/tauri";
import { useAppStore } from "./store/useAppStore";
import type { HistoryRecord } from "./types";

export const ORT_ENABLED = true;

export function App() {
  const { t, i18n } = useTranslation();
  const prefersReducedMotion = useReducedMotion();

  const activeTab = useAppStore((s) => s.activeTab);
  const setActiveTab = useAppStore((s) => s.setActiveTab);
  const theme = useAppStore((s) => s.theme);
  const lang = useAppStore((s) => s.lang);
  const backendChoice = useAppStore((s) => s.backendChoice);
  const setBackendChoice = useAppStore((s) => s.setBackendChoice);
  const apiBaseUrl = useAppStore((s) => s.apiBaseUrl);
  const setApiBaseUrl = useAppStore((s) => s.setApiBaseUrl);
  const cloudSpaceId = useAppStore((s) => s.cloudSpaceId);
  const tauriMode = useAppStore((s) => s.tauriMode);
  const tauriSidecarStatus = useAppStore((s) => s.tauriSidecarStatus);
  const tauriSidecarError = useAppStore((s) => s.tauriSidecarError);
  const setTauriState = useAppStore((s) => s.setTauriState);
  const setApiHealth = useAppStore((s) => s.setApiHealth);
  const setMeta = useAppStore((s) => s.setMeta);
  const setConfThreshold = useAppStore((s) => s.setConfThreshold);
  const setMetrics = useAppStore((s) => s.setMetrics);
  const setAnalysis = useAppStore((s) => s.setAnalysis);
  const selectedImage = useAppStore((s) => s.selectedImage);
  const setSelectedImage = useAppStore((s) => s.setSelectedImage);
  const confThreshold = useAppStore((s) => s.confThreshold);
  const detectResult = useAppStore((s) => s.detectResult);
  const selectedDetectionId = useAppStore((s) => s.selectedDetectionId);
  const setSelectedDetectionId = useAppStore((s) => s.setSelectedDetectionId);
  const setDetectLoading = useAppStore((s) => s.setDetectLoading);
  const setDetectResult = useAppStore((s) => s.setDetectResult);
  const setDetectError = useAppStore((s) => s.setDetectError);
  const startExplainRequest = useAppStore((s) => s.startExplainRequest);
  const setExplainWaking = useAppStore((s) => s.setExplainWaking);
  const setExplainResult = useAppStore((s) => s.setExplainResult);
  const setExplainError = useAppStore((s) => s.setExplainError);
  const setOrtProgress = useAppStore((s) => s.setOrtProgress);
  const setOrtModelCached = useAppStore((s) => s.setOrtModelCached);
  const bumpHistoryVersion = useAppStore((s) => s.bumpHistoryVersion);

  React.useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    document.documentElement.style.colorScheme = theme;
  }, [theme]);

  React.useEffect(() => {
    void i18n.changeLanguage(lang);
  }, [lang, i18n]);

  const recheckHealth = React.useCallback(async () => {
    try {
      const info = await checkApiHealth(apiBaseUrl);
      setApiHealth({ status: "online", info });
      return true;
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Offline";
      setApiHealth({ status: "offline", error: msg });
      return false;
    }
  }, [apiBaseUrl, setApiHealth]);

  const getOrtEpPreference = React.useCallback((): OrtEpPreference => {
    if (typeof window === "undefined") return "auto";
    const param = new URLSearchParams(window.location.search).get("ep");
    if (param === "wasm" || param === "webgpu") return param;
    return "auto";
  }, []);

  const createDetector = React.useCallback((): Detector => {
    if (backendChoice === "api") {
      return new ApiDetector(apiBaseUrl);
    }
    if (backendChoice === "cloud") {
      return new CloudDetector(cloudSpaceId);
    }
    if (backendChoice === "ort") {
      return new OrtDetector(
        useAppStore.getState().meta,
        (p) => setOrtProgress(p),
        getOrtEpPreference(),
      );
    }
    return new MockDetector();
  }, [
    apiBaseUrl,
    backendChoice,
    cloudSpaceId,
    getOrtEpPreference,
    setOrtProgress,
  ]);

  const detectReqSeqRef = React.useRef(0);

  const runDetectWithFile = React.useCallback(
    async (file: File, conf: number, saveToHistory = true) => {
      const reqId = ++detectReqSeqRef.current;
      setDetectLoading(true);
      setDetectError(null);
      try {
        const detector = createDetector();
        const result = await detector.detect(file, conf);
        if (reqId !== detectReqSeqRef.current) return;
        setDetectResult(result);
        if (backendChoice === "api") {
          setApiHealth({ status: "online" });
        } else if (backendChoice === "ort") {
          setOrtModelCached(true);
        }
        if (saveToHistory) {
          void saveInspectionToHistory({
            filename: file.name,
            fileOrUrl: file,
            confThreshold: conf,
            result,
          }).then(() => bumpHistoryVersion());
        }
      } catch (err) {
        if (reqId !== detectReqSeqRef.current) return;
        if (err instanceof ApiClientError && err.isOffline) {
          setApiHealth({ status: "offline", error: err.message });
          setDetectError(t("detect.apiOfflineError", { url: apiBaseUrl }));
        } else {
          setDetectError(err instanceof Error ? err.message : String(err));
        }
      } finally {
        if (reqId === detectReqSeqRef.current) {
          setDetectLoading(false);
          setOrtProgress({ stage: "idle", percent: 0, message: "" });
        }
      }
    },
    [
      apiBaseUrl,
      backendChoice,
      bumpHistoryVersion,
      createDetector,
      setApiHealth,
      setDetectError,
      setDetectLoading,
      setDetectResult,
      setOrtModelCached,
      setOrtProgress,
      t,
    ],
  );

  const prevBackendRef = React.useRef(backendChoice);
  React.useEffect(() => {
    if (prevBackendRef.current !== backendChoice) {
      prevBackendRef.current = backendChoice;
      explainReqSeqRef.current++;
      setDetectResult(null);
      setExplainResult(null);
      if (backendChoice === "api") {
        void recheckHealth();
      }
      const img = useAppStore.getState().selectedImage;
      if (img && (backendChoice !== "ort" || ORT_ENABLED)) {
        void runDetectWithFile(img.file, useAppStore.getState().confThreshold);
      }
      return;
    }
    if (backendChoice === "api") {
      void recheckHealth();
    }
  }, [
    backendChoice,
    recheckHealth,
    runDetectWithFile,
    setDetectResult,
    setExplainResult,
  ]);

  const handleSelectFile = React.useCallback(
    async (file: File) => {
      const url = URL.createObjectURL(file);
      setSelectedImage({ file, url, name: file.name });
      const conf = useAppStore.getState().confThreshold;
      await runDetectWithFile(file, conf);
    },
    [runDetectWithFile, setSelectedImage],
  );

  const handleSelectExample = React.useCallback(
    async (ex: ValExampleItem, autoDetect = true) => {
      const resp = await fetch(ex.url);
      if (!resp.ok) {
        setDetectError(`Örnek görüntü yüklenemedi: ${ex.filename}`);
        return;
      }
      if (typeof caches !== "undefined") {
        try {
          const cache = await caches.open("tracera-examples-v1");
          await cache.put(ex.url, resp.clone());
        } catch {
          // Gizli sekme veya test ortamında önbellek hatasını yoksay
        }
      }
      const blob = await resp.blob();
      const file = new File([blob], ex.filename, {
        type: blob.type || "image/jpeg",
      });
      if (autoDetect) {
        await handleSelectFile(file);
      } else {
        const url = URL.createObjectURL(file);
        setSelectedImage({ file, url, name: file.name });
      }
    },
    [handleSelectFile, setDetectError, setSelectedImage],
  );

  const handleOpenHistoryRecord = React.useCallback(
    async (rec: HistoryRecord) => {
      try {
        const resp = await fetch(rec.thumbnailDataUrl);
        const blob = await resp.blob();
        const file = new File([blob], rec.filename, {
          type: blob.type || "image/jpeg",
        });
        setSelectedImage({
          file,
          url: rec.thumbnailDataUrl,
          name: rec.filename,
        });
        setConfThreshold(rec.confThreshold);
        setDetectResult(rec.result);
        setActiveTab("detect");
      } catch {
        setActiveTab("detect");
      }
    },
    [setActiveTab, setConfThreshold, setDetectResult, setSelectedImage],
  );

  const handleRunDetect = React.useCallback(
    async (confOverride?: number) => {
      const img = useAppStore.getState().selectedImage;
      if (!img) return;
      const conf = confOverride ?? useAppStore.getState().confThreshold;
      await runDetectWithFile(img.file, conf);
    },
    [runDetectWithFile],
  );

  const explainReqSeqRef = React.useRef(0);

  const handleRunExplain = React.useCallback(async () => {
    const state = useAppStore.getState();
    const img = state.selectedImage;
    const detections = state.detectResult?.detections ?? [];
    const det =
      detections.find((d) => d.id === state.selectedDetectionId) ??
      detections[0];
    if (!img || !det) return;

    const reqId = ++explainReqSeqRef.current;
    startExplainRequest(det.id);
    try {
      const { result, source } = await resolveExplain({
        backendChoice: state.backendChoice,
        apiBaseUrl: state.apiBaseUrl,
        cloudSpaceId: state.cloudSpaceId,
        file: img.file,
        detection: det,
        cloudOnly: !state.localApiAvailable,
        onApiStatusChange: (status, error) => setApiHealth({ status, error }),
        onCloudWaking: () => {
          const cur = useAppStore.getState();
          if (
            reqId === explainReqSeqRef.current &&
            cur.selectedDetectionId === det.id &&
            cur.selectedImage?.file === img.file
          ) {
            setExplainWaking();
          }
        },
      });
      const cur = useAppStore.getState();
      if (
        reqId !== explainReqSeqRef.current ||
        cur.selectedDetectionId !== det.id ||
        cur.selectedImage?.file !== img.file
      ) {
        return;
      }
      setExplainResult(result, source, det.id);
    } catch (err) {
      const cur = useAppStore.getState();
      if (
        reqId !== explainReqSeqRef.current ||
        cur.selectedDetectionId !== det.id ||
        cur.selectedImage?.file !== img.file
      ) {
        return;
      }
      if (err instanceof ApiClientError && err.isOffline) {
        setApiHealth({ status: "offline", error: err.message });
        setExplainError(t("detect.apiOfflineError", { url: apiBaseUrl }));
      } else {
        setExplainError(
          err instanceof Error ? err.message : CLOUD_UNAVAILABLE_MESSAGE,
        );
      }
    }
  }, [
    apiBaseUrl,
    setApiHealth,
    setExplainError,
    setExplainResult,
    setExplainWaking,
    startExplainRequest,
    t,
  ]);

  const handleInspectDetection = React.useCallback(
    (detId: number) => {
      setSelectedDetectionId(detId);
      setActiveTab("explain");
      void handleRunExplain();
    },
    [handleRunExplain, setActiveTab, setSelectedDetectionId],
  );

  const handleExportPdf = React.useCallback(async () => {
    const state = useAppStore.getState();
    if (!state.selectedImage || !state.detectResult) return;
    const pdfBytes = await generateInspectionPdfBytes({
      filename: state.selectedImage.name,
      meta: state.meta,
      confThreshold: state.confThreshold,
      result: state.detectResult,
      explainResult: state.explainResult,
      lang: state.lang,
    });
    const baseName = state.selectedImage.name.replace(/\.[^.]+$/, "");
    triggerPdfDownload(pdfBytes, `${baseName}_tracera_rapor.pdf`);
  }, []);

  const handleSelectExampleRef = React.useRef(handleSelectExample);
  handleSelectExampleRef.current = handleSelectExample;

  React.useEffect(() => {
    let active = true;
    let cleanupSidecar: (() => void) | null = null;

    const finishBootstrap = async (
      resolvedApiUrl: string | null,
      cachedOrt: boolean,
    ) => {
      const { data: loadedMetrics, source } = await loadMetrics(resolvedApiUrl);
      if (!active) return;
      setMetrics(loadedMetrics, source);

      const loadedAnalysis = await loadAnalysis();
      if (!active) return;
      if (loadedAnalysis) {
        setAnalysis(loadedAnalysis);
      }

      if (!useAppStore.getState().selectedImage && VAL_EXAMPLES.length > 0) {
        try {
          const shouldAutoDetect =
            useAppStore.getState().backendChoice !== "ort" || cachedOrt;
          await handleSelectExampleRef.current(
            VAL_EXAMPLES[0],
            shouldAutoDetect,
          );
        } catch {
          // ignore offline bootstrap failure in unit tests
        }
      }
    };

    void (async () => {
      const loadedMeta = await loadModelMeta();
      if (!active) return;
      setMeta(loadedMeta);
      setConfThreshold(loadedMeta.conf);

      const cached = await isOrtModelCached(loadedMeta.onnx);
      if (!active) return;
      setOrtModelCached(cached);

      const inTauri = await isTauriApp();
      if (!active) return;

      if (inTauri) {
        setTauriState({ tauriMode: true });
        cleanupSidecar = await connectTauriSidecar({
          onStarting: () => {
            if (!active) return;
            setBackendChoice("api");
            setTauriState({
              tauriPlatform: "windows",
              localApiAvailable: true,
              tauriSidecarStatus: "starting",
              tauriSidecarError: null,
            });
          },
          onReady: (baseUrl) => {
            if (!active) return;
            setApiBaseUrl(baseUrl);
            setBackendChoice("api");
            setTauriState({
              tauriPlatform: "windows",
              localApiAvailable: true,
              tauriSidecarStatus: "ready",
              tauriSidecarError: null,
            });
            void checkApiHealth(baseUrl)
              .then((info) => {
                if (active) setApiHealth({ status: "online", info });
              })
              .catch(() => {});
            void finishBootstrap(baseUrl, cached);
          },
          onUnavailable: (platform) => {
            if (!active) return;
            setBackendChoice("ort");
            setTauriState({
              tauriPlatform: platform,
              localApiAvailable: false,
              tauriSidecarStatus: "idle",
              tauriSidecarError: null,
            });
            void finishBootstrap(null, cached);
          },
          onError: (errMsg) => {
            if (!active) return;
            setTauriState({
              tauriSidecarStatus: "error",
              tauriSidecarError: errMsg,
            });
          },
        });
        return;
      }

      await finishBootstrap(useAppStore.getState().apiBaseUrl, cached);
    })();

    return () => {
      active = false;
      cleanupSidecar?.();
    };
  }, [
    setAnalysis,
    setApiBaseUrl,
    setApiHealth,
    setBackendChoice,
    setConfThreshold,
    setMeta,
    setMetrics,
    setOrtModelCached,
    setTauriState,
  ]);

  void confThreshold;
  void detectResult;
  void selectedDetectionId;
  void selectedImage;

  if (tauriMode && tauriSidecarStatus === "starting") {
    return (
      <div
        data-testid="tauri-starting-screen"
        className="flex min-h-screen flex-col items-center justify-center bg-[var(--bg-canvas)] px-6 text-center text-[var(--text-primary)]"
      >
        <div className="max-w-md space-y-3 rounded-xl border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-8 shadow-lg">
          <Loader2 className="mx-auto h-10 w-10 animate-spin text-[var(--accent)]" />
          <h1 className="text-lg font-semibold">{t("tauri.startingTitle")}</h1>
          <p className="text-xs text-[var(--text-secondary)]">
            {t("tauri.startingSubtitle")}
          </p>
        </div>
      </div>
    );
  }

  if (tauriMode && tauriSidecarStatus === "error") {
    return (
      <div
        role="alert"
        data-testid="tauri-error-screen"
        className="flex min-h-screen flex-col items-center justify-center bg-[var(--bg-canvas)] px-6 text-center text-[var(--text-primary)]"
      >
        <div className="max-w-lg space-y-3 rounded-xl border border-red-500/40 bg-red-500/10 p-8 shadow-lg">
          <AlertTriangle className="mx-auto h-10 w-10 text-red-500" />
          <h1 className="text-lg font-semibold text-red-700 dark:text-red-300">
            {t("tauri.errorTitle")}
          </h1>
          <p className="break-words font-mono text-xs text-red-800 dark:text-red-200">
            {tauriSidecarError}
          </p>
        </div>
      </div>
    );
  }

  const pageTransition = prefersReducedMotion
    ? { duration: 0 }
    : { type: "spring" as const, stiffness: 360, damping: 32 };

  return (
    <div className="flex min-h-screen bg-[var(--bg-canvas)] text-[var(--text-primary)]">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:rounded-md focus:bg-[var(--accent)] focus:px-3 focus:py-2 focus:text-xs focus:font-semibold focus:text-[var(--accent-contrast)]"
      >
        {t("app.skipToContent")}
      </a>

      <Sidebar />

      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar
          onRecheckHealth={() => void recheckHealth()}
          ortEnabled={ORT_ENABLED}
        />

        <main
          id="main-content"
          className="mx-auto w-full max-w-7xl flex-1 px-4 py-5"
        >
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={activeTab}
              initial={prefersReducedMotion ? false : { opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={prefersReducedMotion ? undefined : { opacity: 0, y: -4 }}
              transition={pageTransition}
            >
              {activeTab === "home" && (
                <HomeView
                  onSelectExample={handleSelectExample}
                  onSelectFile={handleSelectFile}
                  onOpenHistoryRecord={handleOpenHistoryRecord}
                />
              )}
              {activeTab === "detect" && (
                <DetectView
                  onSelectExample={handleSelectExample}
                  onSelectFile={handleSelectFile}
                  onRunDetect={handleRunDetect}
                  onInspectDetection={handleInspectDetection}
                  onExportPdf={handleExportPdf}
                />
              )}
              {activeTab === "live" && <LiveView />}
              {activeTab === "explain" && (
                <ExplainView onRunExplain={handleRunExplain} />
              )}
              {activeTab === "batch" && (
                <BatchView
                  onDetectFile={async (file, conf) => {
                    const detector = createDetector();
                    return await detector.detect(file, conf);
                  }}
                  onOpenInInspect={(item: BatchQueueItem) => {
                    if (!item.result) return;
                    const url = URL.createObjectURL(item.file);
                    setSelectedImage({
                      file: item.file,
                      url,
                      name: item.filename,
                    });
                    setDetectResult(item.result);
                    setActiveTab("detect");
                  }}
                />
              )}
              {activeTab === "results" && <ResultsView />}
              {(activeTab === "about" || activeTab === "developer") && (
                <AboutView />
              )}
              {activeTab === "more" && <MoreView />}
            </motion.div>
          </AnimatePresence>
        </main>

        <MobileNav />
      </div>
    </div>
  );
}

export default App;
