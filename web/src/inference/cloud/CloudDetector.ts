import type { DetectResult } from "../../types";
import type { Detector } from "../types";
import {
  CLOUD_UNAVAILABLE_MESSAGE,
  CloudUnavailableError,
  DEFAULT_CLOUD_SPACE,
  callCloudEndpoint,
  prepareGradioFile,
  type CloudConnector,
} from "./CloudClient";

export class CloudDetector implements Detector {
  private readonly spaceId: string;
  private readonly connector?: CloudConnector;
  private readonly timeoutMs?: number;

  constructor(
    spaceId = DEFAULT_CLOUD_SPACE,
    options?: { connector?: CloudConnector; timeoutMs?: number },
  ) {
    this.spaceId = spaceId.trim() || DEFAULT_CLOUD_SPACE;
    this.connector = options?.connector;
    this.timeoutMs = options?.timeoutMs;
  }

  async detect(file: File, conf: number): Promise<DetectResult> {
    const payload = await callCloudEndpoint<DetectResult>(
      this.spaceId,
      "/detect_api",
      {
        image: await prepareGradioFile(file),
        conf,
      },
      { connector: this.connector, timeoutMs: this.timeoutMs },
    );

    if (
      !Array.isArray(payload.detections) ||
      typeof payload.width !== "number" ||
      typeof payload.height !== "number"
    ) {
      throw new CloudUnavailableError(CLOUD_UNAVAILABLE_MESSAGE);
    }

    return {
      ...payload,
      backend: payload.backend || `cloud:${this.spaceId}`,
    };
  }
}
