import { METRICS_FALLBACK } from "../fixtures";
import type { EdgeBenchmarkRow, MetricsPayload } from "../types";

export async function loadMetrics(apiBaseUrl?: string | null): Promise<{
  data: MetricsPayload;
  source: "api" | "static" | "fallback";
}> {
  if (apiBaseUrl) {
    try {
      const cleanBase = apiBaseUrl.replace(/\/+$/, "");
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 4000);
      const resp = await fetch(`${cleanBase}/metrics`, { signal: ctrl.signal });
      clearTimeout(timer);
      if (resp.ok) {
        const json = (await resp.json()) as MetricsPayload;
        return { data: json, source: "api" };
      }
    } catch {
      // Static metrics.json'a düş (metrik görüntüleme için)
    }
  }

  try {
    const resp = await fetch("/metrics.json");
    if (resp.ok) {
      const json = (await resp.json()) as MetricsPayload;
      return { data: json, source: "static" };
    }
  } catch {
    // Fallback
  }

  return { data: METRICS_FALLBACK, source: "fallback" };
}

/**
 * Ana Sayfa hız kartının kaynağı: YALNIZ 1024 modelin OpenVINO INT8 (CPU) satırı.
 * (Önceki sürüm ilk "int8" satırını alıyordu → 640 modelin 17.4 ms'i gösteriliyordu.)
 */
export function selectOpenVinoInt8Row(
  rows: EdgeBenchmarkRow[] | undefined,
  imgsz = 1024,
): EdgeBenchmarkRow | undefined {
  return rows?.find(
    (r) => r.imgsz === imgsz && r.format === "OpenVINO INT8 (CPU)",
  );
}
