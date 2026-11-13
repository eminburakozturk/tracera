import type { ErrorAnalysisPayload } from "../types";

export async function loadAnalysis(): Promise<ErrorAnalysisPayload | null> {
  try {
    const res = await fetch("/analysis.json", { cache: "no-store" });
    if (!res.ok) return null;
    const payload = (await res.json()) as ErrorAnalysisPayload;
    if (!payload || !Array.isArray(payload.board_class_matrix)) {
      return null;
    }
    return payload;
  } catch {
    return null;
  }
}
