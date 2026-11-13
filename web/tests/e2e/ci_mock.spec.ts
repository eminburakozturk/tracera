import { expect, test } from "@playwright/test";

// 1x1 transparent PNG base64
const DUMMY_PNG_B64 =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";

test.describe("CI Mock & Route E2E (WebGPU / Python API bağımsız)", () => {
  test("Mock motor ile tespit, LayerCAM açıklama, sekmeler ve TR/EN geçişi", async ({
    page,
  }) => {
    await page.goto("/");

    await page.getByTestId("backend-select").click();
    await page.getByRole("option", { name: /Mock/i }).click();

    await expect(page.getByTestId("backend-badge")).toContainText(/Mock/i);
    await page.waitForSelector('[data-testid^="bbox-"]', { timeout: 15_000 });

    await page.locator('[data-testid="bbox-0"] rect').first().click();
    await page.getByTestId("explain-det-0").click();

    await expect(page.getByTestId("explain-zoom-img")).toBeVisible({ timeout: 15_000 });
    await expect(page.getByTestId("explain-overlay-img")).toBeVisible();
    await expect(page.getByTestId("explain-source-badge")).toContainText(/Mock/i);

    await page.getByTestId("tab-results").click();
    await expect(page.getByRole("heading", { level: 2 }).first()).toBeVisible();

    await page.getByTestId("tab-live").click();
    await expect(page.getByTestId("live-sample-btn")).toBeVisible();

    await page.getByTestId("tab-about").click();
    await expect(page.getByText(/AGPL-3\.0/i).first()).toBeVisible();

    await page.getByTestId("lang-toggle").click();
    await expect(page.getByTestId("lang-toggle")).toHaveText(/EN/i);
  });

  test("Playwright route mock ile API modu (/health, /detect, /explain)", async ({
    page,
  }) => {
    await page.route("http://127.0.0.1:8765/**", async (route) => {
      const url = route.request().url();
      if (url.endsWith("/health")) {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            status: "ok",
            model: "yolo11s",
            version: "1",
            device: "CPU",
            backends: ["openvino-int8"],
          }),
        });
        return;
      }
      if (url.endsWith("/metrics")) {
        await route.fulfill({ status: 404, body: "{}" });
        return;
      }
      if (url.endsWith("/detect")) {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            width: 3056,
            height: 2464,
            detections: [
              {
                id: 0,
                cls: "short",
                cls_id: 3,
                conf: 0.91,
                box: [120, 140, 220, 230],
              },
            ],
            timing: { pre_ms: 4.1, model_ms: 37.3, post_ms: 1.5, total_ms: 42.9 },
            backend: "openvino-int8",
          }),
        });
        return;
      }
      if (url.endsWith("/explain")) {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            overlay_png: DUMMY_PNG_B64,
            zoom_png: DUMMY_PNG_B64,
            method: "layercam",
            layer: 4,
            timing: { pre_ms: 5.0, model_ms: 45.0, post_ms: 2.0, total_ms: 52.0 },
          }),
        });
        return;
      }
      await route.continue();
    });

    await page.goto("/");
    await page.getByTestId("backend-select").click();
    await page.getByRole("option", { name: /FastAPI/i }).click();

    await expect(page.getByTestId("api-health-badge")).toContainText(
      /Sunucu Açık|Server Online/i,
      { timeout: 15_000 },
    );
    await page.waitForSelector('[data-testid="bbox-0"]', { timeout: 15_000 });
    await page.getByTestId("explain-det-0").click();

    await expect(page.getByTestId("explain-overlay-img")).toBeVisible({ timeout: 15_000 });
    await expect(page.getByTestId("explain-source-badge")).toContainText(/FastAPI/i);
  });
});
