import fs from "node:fs";
import path from "node:path";
import { expect, test, type Page } from "@playwright/test";

/**
 * README ekran görüntüleri: üretim görünümü (Mock yok). Gerçek yerel API (OpenVINO INT8 + LayerCAM)
 * ile tespit ve ısı haritası. API http://127.0.0.1:8765 adresinde çalışıyor olmalı
 * (claude_run/release_prep.bat başlatır). Çıktı: results/readme/*.png
 */
const OUT_DIR = path.resolve(process.cwd(), "../results/readme");
const EXAMPLE = "example-val_05_missing_hole_01";

async function useLocalApi(page: Page) {
  await page.getByTestId("backend-select").click();
  await page.getByRole("option", { name: /Yerel Sunucu/i }).click();
  await expect(page.getByTestId("api-health-badge")).toContainText(/Sunucu Açık|Server Online/i, {
    timeout: 20_000,
  });
}

async function detectAndExplain(page: Page) {
  await page.waitForSelector(`[data-testid="${EXAMPLE}"]`, { timeout: 20_000 });
  await page.getByTestId(EXAMPLE).click();
  await page.waitForSelector('[data-testid="bbox-0"]', { timeout: 30_000 });
  await page.getByTestId("explain-det-0").click();
  await expect(page.getByTestId("explain-overlay-img")).toBeVisible({ timeout: 40_000 });
}

test.describe("README ekran görüntüleri (gerçek yerel API)", () => {
  test.beforeAll(() => {
    fs.mkdirSync(OUT_DIR, { recursive: true });
  });

  test("Masaüstü: koyu ana ekranlar + açık Sonuçlar", async ({ browser }) => {
    test.setTimeout(180_000);
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, colorScheme: "dark" });
    const page = await ctx.newPage();
    await page.goto("/");
    await page.getByTestId("tab-detect").click();
    await useLocalApi(page);
    await detectAndExplain(page);

    await page.getByTestId("tab-home").click();
    await page.waitForTimeout(400);
    await page.screenshot({ path: path.join(OUT_DIR, "home_desktop_dark.png") });

    await page.getByTestId("tab-detect").click();
    await page.waitForTimeout(400);
    await page.screenshot({ path: path.join(OUT_DIR, "detect_desktop_dark.png") });

    await page.getByTestId("tab-explain").click();
    await page.waitForTimeout(600);
    await page.screenshot({ path: path.join(OUT_DIR, "explain_desktop_dark.png"), fullPage: true });
    await ctx.close();

    const light = await browser.newContext({ viewport: { width: 1440, height: 900 }, colorScheme: "light" });
    const lp = await light.newPage();
    await lp.goto("/");
    await lp.getByTestId("tab-results").click();
    await lp.waitForTimeout(800);
    await lp.screenshot({ path: path.join(OUT_DIR, "results_desktop_light.png") });
    await light.close();
  });

  test("Mobil 390px: Muayene (koyu), Açıklama (açık), Rehber (koyu)", async ({ browser }) => {
    test.setTimeout(180_000);
    const dark = await browser.newContext({ viewport: { width: 390, height: 844 }, colorScheme: "dark", hasTouch: true });
    const dp = await dark.newPage();
    await dp.goto("/");
    await dp.getByTestId("mobile-tab-detect").click();
    await useLocalApi(dp);
    await dp.waitForSelector(`[data-testid="${EXAMPLE}"]`, { timeout: 20_000 });
    await dp.getByTestId(EXAMPLE).click();
    await dp.waitForSelector('[data-testid="bbox-0"]', { timeout: 30_000 });
    await dp.screenshot({ path: path.join(OUT_DIR, "detect_mobile_dark.png") });
    await dp.getByTestId("mobile-tab-more").click();
    await dp.getByTestId("more-link-guide").click();
    await dp.waitForTimeout(600);
    await dp.screenshot({ path: path.join(OUT_DIR, "guide_mobile_dark.png") });
    await dark.close();

    const light = await browser.newContext({ viewport: { width: 390, height: 844 }, colorScheme: "light", hasTouch: true });
    const lp = await light.newPage();
    await lp.goto("/");
    await lp.getByTestId("mobile-tab-detect").click();
    await useLocalApi(lp);
    await detectAndExplain(lp);
    await lp.getByTestId("mobile-tab-explain").click();
    await lp.waitForTimeout(600);
    await lp.screenshot({ path: path.join(OUT_DIR, "explain_mobile_light.png") });
    await light.close();
  });
});
