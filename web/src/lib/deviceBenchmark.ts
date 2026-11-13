import { VAL_EXAMPLES, type ValExampleItem } from "../fixtures";
import {
  OrtDetector,
  type OrtEpPreference,
} from "../inference/ort/OrtDetector";
import type { DetectResult, ModelMeta } from "../types";

export interface GpuAdapterSummary {
  available: boolean;
  vendor?: string;
  architecture?: string;
  device?: string;
  description?: string;
}

export interface DeviceBenchmarkReport {
  timestamp: string;
  backend: string;
  images: number;
  repeats: number;
  warmup_runs: number;
  timed_runs: number;
  median_model_ms: number;
  median_total_ms: number;
  model_ms_samples: number[];
  total_ms_samples: number[];
  user_agent: string;
  gpu_adapter: GpuAdapterSummary;
}

export interface DeviceBenchmarkDetector {
  detect(file: File, conf: number): Promise<DetectResult>;
}

export function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 1) {
    return Number(sorted[mid].toFixed(2));
  }
  return Number(((sorted[mid - 1] + sorted[mid]) / 2).toFixed(2));
}

export async function getGpuAdapterSummary(): Promise<GpuAdapterSummary> {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) {
    return { available: false };
  }
  try {
    const gpu = (
      navigator as Navigator & {
        gpu?: {
          requestAdapter?: () => Promise<{
            info?: {
              vendor?: string;
              architecture?: string;
              device?: string;
              description?: string;
            };
          } | null>;
        };
      }
    ).gpu;
    const adapter = await gpu?.requestAdapter?.();
    if (!adapter) {
      return { available: false };
    }
    const info = adapter.info ?? {};
    return {
      available: true,
      vendor: info.vendor || undefined,
      architecture: info.architecture || undefined,
      device: info.device || undefined,
      description: info.description || undefined,
    };
  } catch {
    return { available: false };
  }
}

async function fetchValImageFile(item: ValExampleItem): Promise<File> {
  const res = await fetch(item.url);
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}: ${item.url}`);
  }
  const blob = await res.blob();
  return new File([blob], item.filename, {
    type: blob.type || "image/jpeg",
  });
}

function getEpPrefFromUrl(): OrtEpPreference {
  if (typeof window === "undefined") return "auto";
  const param = new URLSearchParams(window.location.search).get("ep");
  if (param === "wasm" || param === "webgpu") return param;
  return "auto";
}

export interface RunDeviceBenchmarkOptions {
  meta: ModelMeta;
  repeats?: number;
  examples?: ValExampleItem[];
  onProgress?: (done: number, total: number) => void;
  detector?: DeviceBenchmarkDetector;
  loadExampleFile?: (item: ValExampleItem) => Promise<File>;
  shouldAbort?: () => boolean;
}

export async function runDeviceBenchmark(
  opts: RunDeviceBenchmarkOptions,
): Promise<DeviceBenchmarkReport> {
  const repeats = opts.repeats ?? 5;
  const examples = opts.examples ?? VAL_EXAMPLES;
  const detector =
    opts.detector ??
    new OrtDetector(opts.meta, undefined, getEpPrefFromUrl());
  const loadFile = opts.loadExampleFile ?? fetchValImageFile;

  const files = await Promise.all(examples.map((ex) => loadFile(ex)));
  if (files.length === 0) {
    throw new Error("Ölçüm için örnek görüntü bulunamadı.");
  }

  const totalSteps = 1 + files.length * repeats;
  let doneSteps = 0;

  // 1 warm-up run (model load + WebGPU shader / WASM compilation excluded from median)
  const warmResult = await detector.detect(files[0], opts.meta.conf);
  doneSteps += 1;
  opts.onProgress?.(doneSteps, totalSteps);

  let lastBackend = warmResult.backend;
  const modelSamples: number[] = [];
  const totalSamples: number[] = [];

  for (let rep = 0; rep < repeats; rep++) {
    for (const file of files) {
      if (opts.shouldAbort?.()) {
        throw new Error("Ölçüm iptal edildi.");
      }
      const res = await detector.detect(file, opts.meta.conf);
      lastBackend = res.backend;
      modelSamples.push(res.timing.model_ms);
      totalSamples.push(res.timing.total_ms);
      doneSteps += 1;
      opts.onProgress?.(doneSteps, totalSteps);
    }
  }

  const gpuAdapter = await getGpuAdapterSummary();

  return {
    timestamp: new Date().toISOString(),
    backend: lastBackend,
    images: files.length,
    repeats,
    warmup_runs: 1,
    timed_runs: modelSamples.length,
    median_model_ms: median(modelSamples),
    median_total_ms: median(totalSamples),
    model_ms_samples: modelSamples,
    total_ms_samples: totalSamples,
    user_agent:
      typeof navigator !== "undefined" ? navigator.userAgent : "unknown",
    gpu_adapter: gpuAdapter,
  };
}
