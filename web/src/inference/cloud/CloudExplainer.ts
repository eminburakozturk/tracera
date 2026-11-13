import type { Detection, ExplainResult } from "../../types";
import type { Explainer } from "../types";
import {
  CLOUD_UNAVAILABLE_MESSAGE,
  CloudUnavailableError,
  DEFAULT_CLOUD_SPACE,
  callCloudEndpoint,
  prepareGradioFile,
  type CloudCallOptions,
} from "./CloudClient";

export class CloudExplainer implements Explainer {
  private readonly spaceId: string;
  private readonly options?: CloudCallOptions;

  constructor(spaceId = DEFAULT_CLOUD_SPACE, options?: CloudCallOptions) {
    this.spaceId = spaceId.trim() || DEFAULT_CLOUD_SPACE;
    this.options = options;
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
      this.options,
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
