import { expect, test, type Page } from "@playwright/test";

async function switchBackendToApi(page: Page) {
  await page.getByTestId("backend-select").click();
  await page.getByRole("option", { name: /Yerel Sunucu/i }).click();
}

test.describe("Adım 2 — Real FastAPI End-to-End Workflow", () => {
  test("Desktop 1440px: example -> API detect -> click box -> LayerCAM explain -> download", async ({
    browser,
  }) => {
    const context = await browser.newContext({
      viewport: { width: 1440, height: 900 },
      acceptDownloads: true,
    });
    const page = await context.newPage();

    await page.goto("/");
    await switchBackendToApi(page);

    await expect(page.getByTestId("api-health-badge")).toContainText(/Sunucu Açık|Server Online/i, {
      timeout: 15_000,
    });
    await page.waitForSelector('[data-testid^="bbox-"]', { timeout: 20_000 });

    const detectReqPromise = page.waitForResponse(
      (res) => res.url().includes("/detect") && res.request().method() === "POST",
    );
    await page.getByTestId("example-val_05_short_01").click();
    const detectRes = await detectReqPromise;
    expect(detectRes.status()).toBe(200);

    await page.waitForSelector('[data-testid="bbox-0"]', { timeout: 20_000 });
    await page.locator('[data-testid="bbox-0"] rect').first().click();

    const jsonDownloadPromise = page.waitForEvent("download");
    await page.getByTestId("download-json-btn").click();
    const jsonDownload = await jsonDownloadPromise;
    expect(jsonDownload.suggestedFilename()).toContain(".json");

    const pngDownloadPromise = page.waitForEvent("download");
    await page.getByTestId("download-png-btn").click();
    const pngDownload = await pngDownloadPromise;
    expect(pngDownload.suggestedFilename()).toContain(".png");

    const explainReqPromise = page.waitForResponse(
      (res) => res.url().includes("/explain") && res.request().method() === "POST",
    );
    await page.getByTestId("explain-det-0").click();
    const explainRes = await explainReqPromise;
    expect(explainRes.status()).toBe(200);

    await expect(page.getByTestId("explain-zoom-img")).toBeVisible({ timeout: 25_000 });
    await expect(page.getByTestId("explain-overlay-img")).toBeVisible();

    const overlayDownloadPromise = page.waitForEvent("download");
    await page.getByTestId("download-overlay-btn").click();
    const overlayDownload = await overlayDownloadPromise;
    expect(overlayDownload.suggestedFilename()).toContain("layercam_overlay.png");

    await context.close();
  });

  test("Mobile 390px: example -> API detect -> explain -> download", async ({
    browser,
  }) => {
    const context = await browser.newContext({
      viewport: { width: 390, height: 844 },
      acceptDownloads: true,
    });
    const page = await context.newPage();

    await page.goto("/");
    await switchBackendToApi(page);
    await expect(page.getByTestId("api-health-badge")).toContainText(/Sunucu Açık|Server Online/i, {
      timeout: 15_000,
    });

    await page.waitForSelector('[data-testid^="bbox-"]', { timeout: 20_000 });
    const explainReqPromise = page.waitForResponse(
      (res) => res.url().includes("/explain") && res.request().method() === "POST",
    );
    await page.getByTestId("explain-det-0").click();
    const explainRes = await explainReqPromise;
    expect(explainRes.status()).toBe(200);

    await expect(page.getByTestId("explain-overlay-img")).toBeVisible({ timeout: 25_000 });
    const overlayDownloadPromise = page.waitForEvent("download");
    await page.getByTestId("download-overlay-btn").click();
    const overlayDownload = await overlayDownloadPromise;
    expect(overlayDownload.suggestedFilename()).toContain(".png");

    await context.close();
  });

  test("Shows explicit offline error when API endpoint is unreachable (no silent mock fallback)", async ({
    browser,
  }) => {
    const context = await browser.newContext({
      viewport: { width: 1280, height: 800 },
    });
    const page = await context.newPage();
    await page.route("http://127.0.0.1:8765/**", (route) => route.abort());

    await page.goto("/");
    await switchBackendToApi(page);

    await expect(page.getByTestId("detect-error-banner")).toBeVisible({ timeout: 15_000 });
    await expect(page.getByTestId("detect-error-banner")).toContainText(/Yerel sunucu kapalı|offline/i);
    await context.close();
  });
});
