import { findDetectFixture } from "../../fixtures";
import type { DetectResult } from "../../types";
import type { Detector } from "../types";

export class MockDetector implements Detector {
  async detect(file: File, conf: number): Promise<DetectResult> {
    const t0 = performance.now();
    const fixture = findDetectFixture(file.name);
    // Küçük gerçekçi bekleme
    await new Promise((r) => setTimeout(r, 90));
    const filtered = fixture.detections
      .filter((d) => d.conf >= conf)
      .map((d, idx) => ({ ...d, id: idx }));

    const elapsed = Number((performance.now() - t0).toFixed(2));
    return {
      width: fixture.width,
      height: fixture.height,
      detections: filtered,
      timing: {
        pre_ms: fixture.timing.pre_ms,
        model_ms: fixture.timing.model_ms,
        post_ms: fixture.timing.post_ms,
        total_ms: Math.max(fixture.timing.total_ms, elapsed),
      },
      backend: fixture.backend,
    };
  }
}
