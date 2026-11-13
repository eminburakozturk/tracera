import path from "node:path";
import { expect, test } from "@playwright/test";

const RESULTS_DIR = path.resolve(process.cwd(), "../results");

test("Adım 1 & Mobil WebView Uyumluluğu — Masaüstü (1440px) ve Mobil (390px) tüm sekmelerin ekran görüntüleri", async ({
  browser,
}) => {
  test.setTimeout(120_000);

  // 1. Desktop (1440x900)
  const desktopCtx = await browser.newContext({
    viewport: { width: 1440, height: 900 },
  });
  const desktopPage = await desktopCtx.newPage();
  await desktopPage.goto("/");
  await expect(desktopPage.getByTestId("run-detect-btn")).toBeEnabled({
    timeout: 15_000,
  });
  await desktopPage.getByTestId("run-detect-btn").click();
  await desktopPage.waitForSelector('[data-testid^="bbox-"]', { timeout: 30_000 });
  await desktopPage.waitForSelector('[data-testid="timing-badge"]', {
    timeout: 15_000,
  });
  await desktopPage.screenshot({
    path: path.join(RESULTS_DIR, "ui_web_desktop.png"),
    fullPage: true,
  });
  await desktopCtx.close();

  // 2. Mobile (390x844) — tüm sekmeler (detect, live, explain, results, about)
  const mobileCtx = await browser.newContext({
    viewport: { width: 390, height: 844 },
  });
  const mobilePage = await mobileCtx.newPage();
  await mobilePage.goto("/");
  await expect(mobilePage.getByTestId("run-detect-btn")).toBeEnabled({
    timeout: 15_000,
  });
  await mobilePage.getByTestId("run-detect-btn").click();
  await mobilePage.waitForSelector('[data-testid^="bbox-"]', { timeout: 30_000 });
  await expect(mobilePage.getByTestId("mobile-tab-detect")).toBeVisible();

  const checkNoHorizontalOverflow = async (label: string) => {
    const hasOverflow = await mobilePage.evaluate(() => {
      return document.documentElement.scrollWidth > window.innerWidth + 2;
    });
    expect(hasOverflow, `Horizontal overflow on mobile tab: ${label}`).toBe(false);
  };

  // Detect tab (390px)
  await checkNoHorizontalOverflow("detect");
  await mobilePage.screenshot({
    path: path.join(RESULTS_DIR, "ui_web_mobile.png"),
    fullPage: true,
  });
  await mobilePage.screenshot({
    path: path.join(RESULTS_DIR, "ui_mobile_detect.png"),
    fullPage: true,
  });

  // Live tab (390px)
  await mobilePage.getByTestId("mobile-tab-live").click();
  await expect(mobilePage.getByTestId("live-camera-note")).toBeVisible();
  await mobilePage.waitForFunction(
    () => Number(document.querySelector('[data-testid="live-det-count"]')?.getAttribute("data-frames") ?? "0") >= 1,
    undefined,
    { timeout: 30_000 },
  );
  await checkNoHorizontalOverflow("live");
  await mobilePage.screenshot({
    path: path.join(RESULTS_DIR, "ui_mobile_live.png"),
    fullPage: true,
  });

  // Explain tab (390px) — LayerCAM ısı haritası mümkünse üretilir
  await mobilePage.getByTestId("mobile-tab-explain").click();
  await expect(mobilePage.getByTestId("generate-heatmap-btn")).toBeVisible();
  await mobilePage.getByTestId("generate-heatmap-btn").click();
  await mobilePage
    .locator('[data-testid="explain-overlay-img"], [data-testid="explain-error"]')
    .first()
    .waitFor({ state: "visible", timeout: 25_000 });
  await checkNoHorizontalOverflow("explain");
  await mobilePage.screenshot({
    path: path.join(RESULTS_DIR, "ui_mobile_explain.png"),
    fullPage: true,
  });

  // Results tab (390px) — Hata analizi galerisi ve cihaz hız testi kartı dahil
  await mobilePage.getByTestId("mobile-tab-results").click();
  await expect(mobilePage.getByTestId("error-analysis-section")).toBeVisible();
  await expect(mobilePage.getByTestId("device-bench-section")).toBeVisible();
  await checkNoHorizontalOverflow("results");
  await mobilePage.screenshot({
    path: path.join(RESULTS_DIR, "ui_mobile_results.png"),
    fullPage: true,
  });

  // About tab (390px)
  await mobilePage.getByTestId("mobile-tab-about").click();
  await expect(
    mobilePage.getByText(/Tracera Hakkında|About Tracera/i),
  ).toBeVisible();
  await checkNoHorizontalOverflow("about");
  await mobilePage.screenshot({
    path: path.join(RESULTS_DIR, "ui_mobile_about.png"),
    fullPage: true,
  });

  await mobileCtx.close();
});
