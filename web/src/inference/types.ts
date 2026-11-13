import type { DetectResult, Detection, ExplainResult } from "../types";

export type BackendChoice = "mock" | "api" | "cloud" | "ort";
export type ExplainSource = "api" | "cloud" | "mock";

export interface OrtProgress {
  stage: "idle" | "downloading" | "initializing" | "ready" | "error";
  percent: number;
  message: string;
}

export interface Detector {
  detect(file: File, conf: number): Promise<DetectResult>;
}

export interface Explainer {
  explain(file: File, det: Detection): Promise<ExplainResult>;
}
