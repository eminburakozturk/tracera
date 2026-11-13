import { create } from "zustand";
import { FALLBACK_META } from "../lib/meta";
import { DEFAULT_API_BASE_URL } from "../inference/api/ApiDetector";
import { DEFAULT_CLOUD_SPACE } from "../inference/cloud/CloudClient";
import type { BackendChoice, ExplainSource, OrtProgress } from "../inference/types";
import type { TauriPlatform } from "../platform/tauri";
import type {
  DetectResult,
  ErrorAnalysisPayload,
  ExplainResult,
  HealthResponse,
  MetricsPayload,
  ModelMeta,
} from "../types";

export type AppTab = "detect" | "live" | "explain" | "results" | "about";
export type ThemeMode = "dark" | "light";
export type LangCode = "tr" | "en";
export type ExplainStatus = "idle" | "loading" | "waking" | "success" | "error";

export function getInitialTheme(): ThemeMode {
  if (typeof window !== "undefined" && typeof window.matchMedia === "function") {
    if (window.matchMedia("(prefers-color-scheme: light)").matches) {
      return "light";
    }
  }
  return "dark";
}

export interface SelectedImageState {
  file: File;
  url: string;
  name: string;
}

export interface ApiHealthState {
  status: "unknown" | "online" | "offline";
  info?: HealthResponse;
  error?: string;
}

interface AppState {
  activeTab: AppTab;
  theme: ThemeMode;
  lang: LangCode;
  backendChoice: BackendChoice;
  apiBaseUrl: string;
  apiHealth: ApiHealthState;
  cloudSpaceId: string;
  meta: ModelMeta;
  metrics: MetricsPayload | null;
  metricsSource: "api" | "static" | "fallback" | null;
  analysis: ErrorAnalysisPayload | null;

  selectedImage: SelectedImageState | null;
  confThreshold: number;
  classFilter: string;
  detectResult: DetectResult | null;
  detectLoading: boolean;
  detectError: string | null;
  selectedDetectionId: number | null;

  explainResult: ExplainResult | null;
  explainSource: ExplainSource | null;
  explainDetectionId: number | null;
  explainStatus: ExplainStatus;
  explainLoading: boolean;
  explainError: string | null;
  heatmapOpacity: number;

  ortProgress: OrtProgress;
  ortModelCached: boolean | null;

  tauriMode: boolean;
  tauriPlatform: TauriPlatform | null;
  localApiAvailable: boolean;
  tauriSidecarStatus: "idle" | "starting" | "ready" | "error";
  tauriSidecarError: string | null;

  setActiveTab: (tab: AppTab) => void;
  setTheme: (theme: ThemeMode) => void;
  setLang: (lang: LangCode) => void;
  setBackendChoice: (backend: BackendChoice) => void;
  setApiBaseUrl: (url: string) => void;
  setApiHealth: (health: ApiHealthState) => void;
  setCloudSpaceId: (spaceId: string) => void;
  setMeta: (meta: ModelMeta) => void;
  setMetrics: (metrics: MetricsPayload, source: "api" | "static" | "fallback") => void;
  setAnalysis: (analysis: ErrorAnalysisPayload | null) => void;

  setSelectedImage: (img: SelectedImageState | null) => void;
  setConfThreshold: (conf: number) => void;
  setClassFilter: (cls: string) => void;
  setDetectLoading: (loading: boolean) => void;
  setDetectResult: (res: DetectResult | null) => void;
  setDetectError: (err: string | null) => void;
  setSelectedDetectionId: (id: number | null) => void;

  setExplainLoading: (loading: boolean) => void;
  startExplainRequest: (detectionId: number) => void;
  setExplainWaking: () => void;
  setExplainResult: (
    res: ExplainResult | null,
    source?: ExplainSource | null,
    detectionId?: number | null,
  ) => void;
  setExplainError: (err: string | null) => void;
  setHeatmapOpacity: (opacity: number) => void;

  setOrtProgress: (progress: OrtProgress) => void;
  setOrtModelCached: (cached: boolean | null) => void;
  setTauriState: (patch: {
    tauriMode?: boolean;
    tauriPlatform?: TauriPlatform | null;
    localApiAvailable?: boolean;
    tauriSidecarStatus?: "idle" | "starting" | "ready" | "error";
    tauriSidecarError?: string | null;
  }) => void;
}

export const useAppStore = create<AppState>((set) => ({
  activeTab: "detect",
  theme: getInitialTheme(),
  lang: "tr",
  backendChoice: "ort",
  apiBaseUrl: DEFAULT_API_BASE_URL,
  apiHealth: { status: "unknown" },
  cloudSpaceId: DEFAULT_CLOUD_SPACE,
  meta: FALLBACK_META,
  metrics: null,
  metricsSource: null,
  analysis: null,

  selectedImage: null,
  confThreshold: FALLBACK_META.conf,
  classFilter: "all",
  detectResult: null,
  detectLoading: false,
  detectError: null,
  selectedDetectionId: null,

  explainResult: null,
  explainSource: null,
  explainDetectionId: null,
  explainStatus: "idle",
  explainLoading: false,
  explainError: null,
  heatmapOpacity: 0.72,

  ortProgress: { stage: "idle", percent: 0, message: "" },
  ortModelCached: null,

  tauriMode: false,
  tauriPlatform: null,
  localApiAvailable: true,
  tauriSidecarStatus: "idle",
  tauriSidecarError: null,

  setActiveTab: (activeTab) => set({ activeTab }),
  setTheme: (theme) => set({ theme }),
  setLang: (lang) => set({ lang }),
  setBackendChoice: (backendChoice) => set({ backendChoice, detectError: null }),
  setApiBaseUrl: (apiBaseUrl) => set({ apiBaseUrl }),
  setApiHealth: (apiHealth) => set({ apiHealth }),
  setCloudSpaceId: (cloudSpaceId) => set({ cloudSpaceId }),
  setMeta: (meta) => set({ meta }),
  setMetrics: (metrics, metricsSource) => set({ metrics, metricsSource }),
  setAnalysis: (analysis) => set({ analysis }),

  setSelectedImage: (img) =>
    set((prev) => {
      if (prev.selectedImage && prev.selectedImage.url !== img?.url) {
        URL.revokeObjectURL(prev.selectedImage.url);
      }
      return {
        selectedImage: img,
        detectResult: null,
        detectError: null,
        selectedDetectionId: null,
        explainResult: null,
        explainSource: null,
        explainDetectionId: null,
        explainStatus: "idle",
        explainLoading: false,
        explainError: null,
      };
    }),
  setConfThreshold: (confThreshold) => set({ confThreshold }),
  setClassFilter: (classFilter) => set({ classFilter }),
  setDetectLoading: (detectLoading) => set({ detectLoading }),
  setDetectResult: (detectResult) =>
    set({
      detectResult,
      detectError: null,
      selectedDetectionId:
        detectResult && detectResult.detections.length > 0 ? detectResult.detections[0].id : null,
      explainResult: null,
      explainSource: null,
      explainDetectionId: null,
      explainStatus: "idle",
      explainLoading: false,
      explainError: null,
    }),
  setDetectError: (detectError) => set({ detectError }),
  setSelectedDetectionId: (selectedDetectionId) =>
    set({
      selectedDetectionId,
      explainResult: null,
      explainSource: null,
      explainDetectionId: null,
      explainStatus: "idle",
      explainLoading: false,
      explainError: null,
    }),

  setExplainLoading: (explainLoading) =>
    set((prev) => ({
      explainLoading,
      explainStatus: explainLoading ? prev.explainStatus === "waking" ? "waking" : "loading" : prev.explainStatus,
    })),
  startExplainRequest: (detectionId) =>
    set({
      explainLoading: true,
      explainStatus: "loading",
      explainError: null,
      explainResult: null,
      explainSource: null,
      explainDetectionId: detectionId,
    }),
  setExplainWaking: () =>
    set((prev) => (prev.explainLoading ? { explainStatus: "waking" } : prev)),
  setExplainResult: (explainResult, explainSource = null, detectionId = null) =>
    set({
      explainResult,
      explainSource,
      explainDetectionId: explainResult ? detectionId : null,
      explainLoading: false,
      explainError: null,
      explainStatus: explainResult ? "success" : "idle",
    }),
  setExplainError: (explainError) =>
    set(
      explainError
        ? {
            explainError,
            explainLoading: false,
            explainStatus: "error",
            explainResult: null,
            explainSource: null,
            explainDetectionId: null,
          }
        : {
            explainError: null,
          },
    ),
  setHeatmapOpacity: (heatmapOpacity) => set({ heatmapOpacity }),

  setOrtProgress: (ortProgress) => set({ ortProgress }),
  setOrtModelCached: (ortModelCached) => set({ ortModelCached }),
  setTauriState: (patch) => set((prev) => ({ ...prev, ...patch })),
}));
