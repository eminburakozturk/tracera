import * as React from "react";
import { useTranslation } from "react-i18next";
import { AboutView } from "./components/AboutView";
import { DetectView } from "./components/DetectView";
import { ExplainView } from "./components/ExplainView";
import { MobileNav } from "./components/MobileNav";
import { ResultsView } from "./components/ResultsView";
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
import { loadModelMeta } from "./lib/meta";
import { loadMetrics } from "./lib/metrics";
import { useAppStore } from "./store/useAppStore";

export const ORT_ENABLED = true;

export function App() {
  const { t, i18n } = useTranslation();

  const activeTab = useAppStore((s) => s.activeTab);
  const setActiveTab = useAppStore((s) => s.setActiveTab);
  const theme = useAppStore((s) => s.theme);
  const lang = useAppStore((s) => s.lang);
  const backendChoice = useAppStore((s) => s.backendChoice);
  const apiBaseUrl = useAppStore((s) => s.apiBaseUrl);
  const cloudSpaceId = useAppStore((s) => s.cloudSpaceId);
  const setApiHealth = useAppStore((s) => s.setApiHealth);
  const setMeta = useAppStore((s) => s.setMeta);
  const setConfThreshold = useAppStore((s) => s.setConfThreshold);
  const setMetrics = useAppStore((s) => s.setMetrics);
  const selectedImage = useAppStore((s) => s.selectedImage);
  const setSelectedImage = useAppStore((s) => s.setSelectedImage);
  const confThreshold = useAppStore((s) => s.confThreshold);
  const detectResult = useAppStore((s) => s.detectResult);
  const selectedDetectionId = useAppStore((s) => s.selectedDetectionId);
  const setSelectedDetectionId = useAppStore((s) => s.setSelectedDetectionId);
  const setDetectLoading = useAppStore((s) => s.setDetectLoading);
  const setDetectResult = useAppStore((s) => s.setDetectResult);
  const setDetectError = useAppStore((s) => s.setDetectError);
  const setExplainLoading = useAppStore((s) => s.setExplainLoading);
  const setExplainResult = useAppStore((s) => s.setExplainResult);
  const setExplainError = useAppStore((s) => s.setExplainError);
  const setOrtProgress = useAppStore((s) => s.setOrtProgress);
  const setOrtModelCached = useAppStore((s) => s.setOrtModelCached);

  React.useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
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
  }, [apiBaseUrl, backendChoice, cloudSpaceId, getOrtEpPreference, setOrtProgress]);

  const detectReqSeqRef = React.useRef(0);

  const runDetectWithFile = React.useCallback(
    async (file: File, conf: number) => {
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

  const handleRunDetect = React.useCallback(
    async (confOverride?: number) => {
      const img = useAppStore.getState().selectedImage;
      if (!img) return;
      const conf = confOverride ?? useAppStore.getState().confThreshold;
      await runDetectWithFile(img.file, conf);
    },
    [runDetectWithFile],
  );

  const handleRunExplain = React.useCallback(async () => {
    const state = useAppStore.getState();
    const img = state.selectedImage;
    const detections = state.detectResult?.detections ?? [];
    const det =
      detections.find((d) => d.id === state.selectedDetectionId) ?? detections[0];
    if (!img || !det) return;

    setExplainLoading(true);
    setExplainError(null);
    try {
      const { result, source } = await resolveExplain({
        backendChoice: state.backendChoice,
        apiBaseUrl: state.apiBaseUrl,
        cloudSpaceId: state.cloudSpaceId,
        file: img.file,
        detection: det,
        onApiStatusChange: (status, error) => setApiHealth({ status, error }),
      });
      setExplainResult(result, source);
    } catch (err) {
      if (err instanceof ApiClientError && err.isOffline) {
        setApiHealth({ status: "offline", error: err.message });
        setExplainError(t("detect.apiOfflineError", { url: apiBaseUrl }));
      } else {
        setExplainError(
          err instanceof Error ? err.message : CLOUD_UNAVAILABLE_MESSAGE,
        );
      }
    } finally {
      setExplainLoading(false);
    }
  }, [
    apiBaseUrl,
    setApiHealth,
    setExplainError,
    setExplainLoading,
    setExplainResult,
    t,
  ]);

  const handleInspectDetection = React.useCallback(
    (detId: number) => {
      setSelectedDetectionId(detId);
      setActiveTab("explain");
      setTimeout(() => {
        void handleRunExplain();
      }, 0);
    },
    [handleRunExplain, setActiveTab, setSelectedDetectionId],
  );

  const bootstrappedRef = React.useRef(false);
  React.useEffect(() => {
    if (bootstrappedRef.current) return;
    bootstrappedRef.current = true;

    void (async () => {
      const loadedMeta = await loadModelMeta();
      setMeta(loadedMeta);
      setConfThreshold(loadedMeta.conf);

      const { data: loadedMetrics, source } = await loadMetrics(
        useAppStore.getState().apiBaseUrl,
      );
      setMetrics(loadedMetrics, source);

      const cached = await isOrtModelCached(loadedMeta.onnx);
      setOrtModelCached(cached);

      if (!useAppStore.getState().selectedImage && VAL_EXAMPLES.length > 0) {
        try {
          const shouldAutoDetect =
            useAppStore.getState().backendChoice !== "ort" || cached;
          await handleSelectExample(VAL_EXAMPLES[0], shouldAutoDetect);
        } catch {
          // ignore offline bootstrap failure in unit tests
        }
      }
    })();
  }, [handleSelectExample, setConfThreshold, setMeta, setMetrics, setOrtModelCached]);

  // Suppress unused variable warnings on confThreshold/detectResult/selectedDetectionId if needed
  void confThreshold;
  void detectResult;
  void selectedDetectionId;
  void selectedImage;

  return (
    <div className="flex min-h-screen flex-col bg-[var(--bg-canvas)] text-[var(--text-primary)]">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:rounded-md focus:bg-[var(--accent)] focus:px-3 focus:py-2 focus:text-xs focus:font-semibold focus:text-[var(--accent-contrast)]"
      >
        {t("app.skipToContent")}
      </a>

      <TopBar onRecheckHealth={() => void recheckHealth()} ortEnabled={ORT_ENABLED} />

      <main id="main-content" className="mx-auto w-full max-w-7xl flex-1 px-4 py-5">
        {activeTab === "detect" && (
          <DetectView
            onSelectExample={handleSelectExample}
            onSelectFile={handleSelectFile}
            onRunDetect={handleRunDetect}
            onInspectDetection={handleInspectDetection}
          />
        )}
        {activeTab === "explain" && (
          <ExplainView onRunExplain={handleRunExplain} />
        )}
        {activeTab === "results" && <ResultsView />}
        {activeTab === "about" && <AboutView />}
      </main>

      <MobileNav />
    </div>
  );
}

export default App;
