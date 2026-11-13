import path from "node:path";
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

  test("Mobil 390px: safe-area-inset simülasyonu, tek bulut notu ve ekran görüntüsü", async ({
    browser,
  }) => {
    const context = await browser.newContext({
      viewport: { width: 390, height: 844 },
    });
    const page = await context.newPage();
    await page.addInitScript(() => {
      (window as unknown as Record<string, unknown>).isTauri = true;
      (window as unknown as Record<string, unknown>).__TAURI_INTERNALS__ = {
        invoke: async (cmd: string) => {
          if (cmd === "get_platform") return "android";
          if (cmd === "get_sidecar_status") {
            return {
              port: 0,
              base_url: "",
              ready: false,
              startup_ms: null,
              error: null,
              available: false,
            };
          }
          if (cmd === "get_api_base_url") return "";
          return null;
        },
        transformCallback: () => 0,
      };
    });
    await page.goto("/");

    const viewportContent = await page
      .locator('meta[name="viewport"]')
      .getAttribute("content");
    expect(viewportContent).toContain("viewport-fit=cover");

    // Masaüstü/headless WebView varsayılan env() değeri (0px beklenir)
    const defaultTopPadding = await page
      .getByTestId("top-bar")
      .evaluate((el) => getComputedStyle(el).paddingTop);
    expect(defaultTopPadding).toBe("0px");

    // Android durum çubuğu & alt çubuk için CSS env override simülasyonu
    await page.addStyleTag({
      content: `
        :root {
          --safe-area-inset-top: 40px;
          --safe-area-inset-bottom: 24px;
        }
      `,
    });

    const topBar = page.getByTestId("top-bar");
    const mobileNav = page.getByTestId("mobile-nav");
    await expect(topBar).toBeVisible();
    await expect(mobileNav).toBeVisible();

    const simulatedTopPadding = await topBar.evaluate(
      (el) => getComputedStyle(el).paddingTop,
    );
    const simulatedBottomPadding = await mobileNav.evaluate(
      (el) => getComputedStyle(el).paddingBottom,
    );
    expect(simulatedTopPadding).toBe("40px");
    expect(simulatedBottomPadding).toBe("24px");

    // "Tracera" başlığı 40px durum çubuğu güvenli alanının altında kalmalı
    const titleBox = await topBar.getByText("Tracera", { exact: true }).boundingBox();
    expect(titleBox).not.toBeNull();
    expect(titleBox!.y).toBeGreaterThanOrEqual(40);

    // Alt sekme çubuğu butonları 24px alt güvenli alanın üstünde kalmalı
    const explainTabBtn = page.getByTestId("mobile-tab-explain");
    const explainBtnBox = await explainTabBtn.boundingBox();
    expect(explainBtnBox).not.toBeNull();
    expect(explainBtnBox!.y + explainBtnBox!.height).toBeLessThanOrEqual(844 - 24 + 1);

    // Açıkla sekmesinde sadeleştirilmiş tek bulut notu ve "seçilen tespitin"
    await explainTabBtn.click();
    const cloudNotes = page.getByText(
      "Bu cihazda ısı haritası bulutta hesaplanır (internet gerekir).",
    );
    await expect(cloudNotes).toHaveCount(1);
    await expect(page.getByText(/seçilen tespitin/i)).toBeVisible();
    await expect(page.getByText(/anchor/i)).toHaveCount(0);

    const hasOverflow = await page.evaluate(
      () => document.documentElement.scrollWidth > window.innerWidth + 2,
    );
    expect(hasOverflow).toBe(false);

    await page.screenshot({
      path: path.resolve(process.cwd(), "../results/ui_mobile_safe_area.png"),
      fullPage: false,
    });
    await context.close();
  });
});
