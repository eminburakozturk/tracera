// @vitest-environment node
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { afterAll, afterEach, beforeAll, describe, expect, it } from "vitest";
import {
  ApiClientError,
  ApiDetector,
  checkApiHealth,
} from "../../src/inference/api/ApiDetector";
import { ApiExplainer } from "../../src/inference/api/ApiExplainer";
import { findDetectFixture } from "../../src/fixtures";

const BASE_URL = "http://127.0.0.1:8765";

const server = setupServer(
  http.get(`${BASE_URL}/health`, () => {
    return HttpResponse.json({
      status: "ok",
      model: "yolo11s",
      version: "1.0.0",
      device: "cpu",
      backends: ["openvino-int8", "pytorch"],
    });
  }),
  http.post(`${BASE_URL}/detect`, () => {
    return HttpResponse.json(findDetectFixture("val_05_missing_hole_01.jpg"));
  }),
  http.post(`${BASE_URL}/explain`, () => {
    return HttpResponse.json({
      overlay_png: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
      zoom_png: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
      method: "layercam",
      layer: 4,
      timing: { pre_ms: 1.0, model_ms: 45.0, post_ms: 2.0, total_ms: 48.0 },
    });
  }),
);

beforeAll(() => server.listen({ onUnhandledRequest: "error" }));
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

describe("ApiDetector & ApiExplainer (MSW)", () => {
  it("checks /health, runs /detect and /explain via CONTRACT §3", async () => {
    const health = await checkApiHealth(BASE_URL);
    expect(health.status).toBe("ok");
    expect(health.model).toBe("yolo11s");

    const detector = new ApiDetector(BASE_URL);
    const file = new File([new Uint8Array([1, 2, 3])], "val_05_missing_hole_01.jpg", {
      type: "image/jpeg",
    });
    const detRes = await detector.detect(file, 0.395);
    expect(detRes.detections.length).toBeGreaterThan(0);

    const explainer = new ApiExplainer(BASE_URL);
    const expRes = await explainer.explain(file, detRes.detections[0]);
    expect(expRes.method).toBe("layercam");
    expect(expRes.layer).toBe(4);
    expect(expRes.overlay_png.length).toBeGreaterThan(10);
  });

  it("throws ApiClientError with isOffline=true on network error instead of falling back", async () => {
    server.use(
      http.post(`${BASE_URL}/detect`, () => {
        return HttpResponse.error();
      }),
    );

    const detector = new ApiDetector(BASE_URL);
    const file = new File([new Uint8Array([1, 2, 3])], "test.jpg", {
      type: "image/jpeg",
    });
    await expect(detector.detect(file, 0.395)).rejects.toMatchObject({
      name: "ApiClientError",
      isOffline: true,
    });
  });

  it("propagates API JSON error payload on HTTP 413/400", async () => {
    server.use(
      http.post(`${BASE_URL}/detect`, () => {
        return HttpResponse.json(
          { error: "Dosya boyutu 25 MB sınırını aşıyor." },
          { status: 413 },
        );
      }),
    );

    const detector = new ApiDetector(BASE_URL);
    const file = new File([new Uint8Array([1, 2, 3])], "huge.jpg", {
      type: "image/jpeg",
    });
    try {
      await detector.detect(file, 0.395);
      expect.unreachable("Should have thrown ApiClientError");
    } catch (err) {
      expect(err).toBeInstanceOf(ApiClientError);
      expect((err as ApiClientError).status).toBe(413);
      expect((err as ApiClientError).message).toContain("25 MB");
    }
  });
});
