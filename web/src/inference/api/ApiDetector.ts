import type { DetectResult, HealthResponse } from "../../types";
import type { Detector } from "../types";

export const DEFAULT_API_BASE_URL = "http://127.0.0.1:8765";

export class ApiClientError extends Error {
  status?: number;
  isOffline: boolean;

  constructor(message: string, opts: { status?: number; isOffline?: boolean } = {}) {
    super(message);
    this.name = "ApiClientError";
    this.status = opts.status;
    this.isOffline = Boolean(opts.isOffline);
  }
}

export async function parseErrorResponse(resp: Response): Promise<string> {
  try {
    const data = (await resp.json()) as { error?: string; detail?: string };
    if (typeof data.error === "string" && data.error.trim()) return data.error;
    if (typeof data.detail === "string" && data.detail.trim()) return data.detail;
  } catch {
    // ignore JSON parse failure
  }
  return `HTTP ${resp.status} ${resp.statusText}`;
}

export async function fetchWithTimeout(
  url: string,
  init: RequestInit,
  timeoutMs: number,
): Promise<Response> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: ctrl.signal });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new ApiClientError(`İstek zaman aşımına uğradı (${Math.round(timeoutMs / 1000)} sn).`, {
        isOffline: false,
      });
    }
    throw new ApiClientError("Yerel sunucu kapalı veya erişilemiyor.", {
      isOffline: true,
    });
  } finally {
    clearTimeout(timer);
  }
}

export async function checkApiHealth(
  baseUrl: string = DEFAULT_API_BASE_URL,
  timeoutMs: number = 3500,
): Promise<HealthResponse> {
  const cleanBase = baseUrl.replace(/\/+$/, "");
  const resp = await fetchWithTimeout(`${cleanBase}/health`, { method: "GET" }, timeoutMs);
  if (!resp.ok) {
    throw new ApiClientError(await parseErrorResponse(resp), { status: resp.status });
  }
  return (await resp.json()) as HealthResponse;
}

export class ApiDetector implements Detector {
  private baseUrl: string;
  private timeoutMs: number;

  constructor(baseUrl: string = DEFAULT_API_BASE_URL, timeoutMs: number = 30_000) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.timeoutMs = timeoutMs;
  }

  async detect(file: File, conf: number): Promise<DetectResult> {
    const form = new FormData();
    form.append("image", file, file.name);
    form.append("conf", String(conf));

    const resp = await fetchWithTimeout(
      `${this.baseUrl}/detect`,
      { method: "POST", body: form },
      this.timeoutMs,
    );
    if (!resp.ok) {
      throw new ApiClientError(await parseErrorResponse(resp), { status: resp.status });
    }
    return (await resp.json()) as DetectResult;
  }
}
