import type { Detection, ExplainResult } from "../../types";
import type { Explainer } from "../types";
import {
  CLOUD_UNAVAILABLE_MESSAGE,
  CloudUnavailableError,
  DEFAULT_CLOUD_SPACE,
  callCloudEndpoint,
  prepareGradioFile,
  type CloudConnector,
} from "./CloudClient";

export class CloudExplainer implements Explainer {
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

  async explain(file: File, det: Detection): Promise<ExplainResult> {
    const boxParam = det.box.map((v) => v.toFixed(2)).join(",");
    const payload = await callCloudEndpoint<ExplainResult>(
      this.spaceId,
      "/explain_api",
      {
        image: await prepareGradioFile(file),
        box: boxParam,
        cls_id: det.cls_id,
      },
      { connector: this.connector, timeoutMs: this.timeoutMs },
    );

    if (
      typeof payload.overlay_png !== "string" ||
      typeof payload.zoom_png !== "string"
    ) {
      throw new CloudUnavailableError(CLOUD_UNAVAILABLE_MESSAGE);
    }

    return payload;
  }
}
