import type { ModelMeta } from "../types";

export const FALLBACK_META: ModelMeta = {
  version: "1.1.0",
  model_version: "1.0.0",
  model: "yolo11s",
  imgsz: 1024,
  conf: 0.395,
  iou_nms: 0.7,
  letterbox_pad: 114,
  classes: [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
  ],
  colors: ["#38bdf8", "#f472b6", "#facc15", "#f87171", "#a78bfa", "#34d399"],
  onnx: "tracera_1024.onnx",
  onnx_sha256:
    "aa8e1f42702c823426df778c9ff298293e8661010dce1d410ea89b6caccbb58c",
};

export function getModelVersion(meta?: Partial<ModelMeta> | null): string {
  const mv = meta?.model_version?.trim();
  return mv && mv.length > 0 ? mv : "bilinmiyor";
}

export function getAppVersion(meta?: Partial<ModelMeta> | null): string {
  const av = meta?.version?.trim();
  return av && av.length > 0 ? av : "bilinmiyor";
}

let metaCache: ModelMeta | null = null;

export async function loadModelMeta(): Promise<ModelMeta> {
  if (metaCache) return metaCache;
  try {
    const resp = await fetch("/models/meta.json");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    metaCache = (await resp.json()) as ModelMeta;
    return metaCache;
  } catch {
    return FALLBACK_META;
  }
}

export function resetMetaCache(): void {
  metaCache = null;
}
