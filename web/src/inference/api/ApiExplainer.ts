import type { Detection, ExplainResult } from "../../types";
import type { Explainer } from "../types";
import {
  ApiClientError,
  DEFAULT_API_BASE_URL,
  fetchWithTimeout,
  parseErrorResponse,
} from "./ApiDetector";

export class ApiExplainer implements Explainer {
  private baseUrl: string;
  private timeoutMs: number;

  constructor(baseUrl: string = DEFAULT_API_BASE_URL, timeoutMs: number = 60_000) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.timeoutMs = timeoutMs;
  }

  async explain(file: File, det: Detection): Promise<ExplainResult> {
    const form = new FormData();
    form.append("image", file, file.name);
    form.append("box", det.box.map((v) => v.toFixed(3)).join(","));
    form.append("cls_id", String(det.cls_id));

    const resp = await fetchWithTimeout(
      `${this.baseUrl}/explain`,
      { method: "POST", body: form },
      this.timeoutMs,
    );
    if (!resp.ok) {
      throw new ApiClientError(await parseErrorResponse(resp), { status: resp.status });
    }
    return (await resp.json()) as ExplainResult;
  }
}
