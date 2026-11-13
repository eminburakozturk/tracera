import type { Detection, ExplainResult } from "../types";
import { ApiClientError } from "./api/ApiDetector";
import { ApiExplainer } from "./api/ApiExplainer";
import { CloudExplainer } from "./cloud/CloudExplainer";
import { MockExplainer } from "./mock/MockExplainer";
import type { BackendChoice, Explainer, ExplainSource } from "./types";

export interface ExplainResolveOptions {
  backendChoice: BackendChoice;
  apiBaseUrl: string;
  cloudSpaceId: string;
  file: File;
  detection: Detection;
  cloudOnly?: boolean;
  onApiStatusChange?: (status: "online" | "offline", error?: string) => void;
  factories?: {
    mock?: () => Explainer;
    api?: (baseUrl: string) => Explainer;
    cloud?: (spaceId: string) => Explainer;
  };
}

export interface ExplainResolved {
  result: ExplainResult;
  source: ExplainSource;
}

export async function resolveExplain(
  opts: ExplainResolveOptions,
): Promise<ExplainResolved> {
  const mkMock = opts.factories?.mock ?? (() => new MockExplainer());
  const mkApi = opts.factories?.api ?? ((url) => new ApiExplainer(url));
  const mkCloud = opts.factories?.cloud ?? ((id) => new CloudExplainer(id));

  if (opts.backendChoice === "mock") {
    return {
      result: await mkMock().explain(opts.file, opts.detection),
      source: "mock",
    };
  }

  if (opts.backendChoice === "cloud") {
    return {
      result: await mkCloud(opts.cloudSpaceId).explain(opts.file, opts.detection),
      source: "cloud",
    };
  }

  if (opts.backendChoice === "api") {
    const result = await mkApi(opts.apiBaseUrl).explain(opts.file, opts.detection);
    opts.onApiStatusChange?.("online");
    return { result, source: "api" };
  }

  // backendChoice === "ort": Android/no-sidecar modunda yalnız Bulut; masaüstü/webde önce Yerel sunucu, çevrimdışıysa Bulut.
  if (opts.cloudOnly) {
    const cloudResult = await mkCloud(opts.cloudSpaceId).explain(
      opts.file,
      opts.detection,
    );
    return { result: cloudResult, source: "cloud" };
  }

  try {
    const result = await mkApi(opts.apiBaseUrl).explain(opts.file, opts.detection);
    opts.onApiStatusChange?.("online");
    return { result, source: "api" };
  } catch (err) {
    if (!(err instanceof ApiClientError && err.isOffline)) {
      throw err;
    }
    opts.onApiStatusChange?.("offline", err.message);
    const cloudResult = await mkCloud(opts.cloudSpaceId).explain(
      opts.file,
      opts.detection,
    );
    return { result: cloudResult, source: "cloud" };
  }
}
