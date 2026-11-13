import { create } from "zustand";
import { FALLBACK_META } from "../lib/meta";
import { DEFAULT_API_BASE_URL } from "../inference/api/ApiDetector";
import type { BackendChoice, OrtProgress } from "../inference/types";
import type {
  DetectResult,
  ExplainResult,
  HealthResponse,
  MetricsPayload,
  ModelMeta,
} from "../types";

export type AppTab = "detect" | "explain" | "results" | "about";
export type ThemeMode = "dark" | "light";
export type LangCode = "tr" | "en";

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
  meta: ModelMeta;
  metrics: MetricsPayload | null;
  metricsSource: "api" | "static" | "fallback" | null;

  selectedImage: SelectedImageState | null;
  confThreshold: number;
  classFilter: string;
  detectResult: DetectResult | null;
  detectLoading: boolean;
  detectError: string | null;
  selectedDetectionId: number | null;

  explainResult: ExplainResult | null;
  explainLoading: boolean;
  explainError: string | null;
  heatmapOpacity: number;

  ortProgress: OrtProgress;

  setActiveTab: (tab: AppTab) => void;
  setTheme: (theme: ThemeMode) => void;
  setLang: (lang: LangCode) => void;
  setBackendChoice: (backend: BackendChoice) => void;
  setApiBaseUrl: (url: string) => void;
  setApiHealth: (health: ApiHealthState) => void;
  setMeta: (meta: ModelMeta) => void;
  setMetrics: (metrics: MetricsPayload, source: "api" | "static" | "fallback") => void;

  setSelectedImage: (img: SelectedImageState | null) => void;
  setConfThreshold: (conf: number) => void;
  setClassFilter: (cls: string) => void;
  setDetectLoading: (loading: boolean) => void;
  setDetectResult: (res: DetectResult | null) => void;
  setDetectError: (err: string | null) => void;
  setSelectedDetectionId: (id: number | null) => void;

  setExplainLoading: (loading: boolean) => void;
  setExplainResult: (res: ExplainResult | null) => void;
  setExplainError: (err: string | null) => void;
  setHeatmapOpacity: (opacity: number) => void;

  setOrtProgress: (progress: OrtProgress) => void;
}

export const useAppStore = create<AppState>((set) => ({
  activeTab: "detect",
  theme: "dark",
  lang: "tr",
  backendChoice: "ort",
  apiBaseUrl: DEFAULT_API_BASE_URL,
  apiHealth: { status: "unknown" },
  meta: FALLBACK_META,
  metrics: null,
  metricsSource: null,

  selectedImage: null,
  confThreshold: FALLBACK_META.conf,
  classFilter: "all",
  detectResult: null,
  detectLoading: false,
  detectError: null,
  selectedDetectionId: null,

  explainResult: null,
  explainLoading: false,
  explainError: null,
  heatmapOpacity: 0.72,

  ortProgress: { stage: "idle", percent: 0, message: "" },

  setActiveTab: (activeTab) => set({ activeTab }),
  setTheme: (theme) => set({ theme }),
  setLang: (lang) => set({ lang }),
  setBackendChoice: (backendChoice) => set({ backendChoice, detectError: null }),
  setApiBaseUrl: (apiBaseUrl) => set({ apiBaseUrl }),
  setApiHealth: (apiHealth) => set({ apiHealth }),
  setMeta: (meta) => set({ meta }),
  setMetrics: (metrics, metricsSource) => set({ metrics, metricsSource }),

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
    }),
  setDetectError: (detectError) => set({ detectError }),
  setSelectedDetectionId: (selectedDetectionId) =>
    set({ selectedDetectionId, explainResult: null, explainError: null }),

  setExplainLoading: (explainLoading) => set({ explainLoading }),
  setExplainResult: (explainResult) => set({ explainResult, explainError: null }),
  setExplainError: (explainError) => set({ explainError }),
  setHeatmapOpacity: (heatmapOpacity) => set({ heatmapOpacity }),

  setOrtProgress: (ortProgress) => set({ ortProgress }),
}));
