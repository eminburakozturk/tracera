import path from "node:path";
import { expect, test } from "@playwright/test";

const RESULTS_DIR = path.resolve(process.cwd(), "../results");

test("Adım 1 — Capture desktop (1440px) and mobile (390px) screenshots", async ({
  browser,
}) => {
  const desktopCtx = await browser.newContext({
    viewport: { width: 1440, height: 900 },
  });
  const desktopPage = await desktopCtx.newPage();
  await desktopPage.goto("/");
  await desktopPage.waitForSelector('[data-testid^="bbox-"]', { timeout: 15_000 });
  await desktopPage.waitForSelector('[data-testid="timing-badge"]', {
    timeout: 15_000,
  });
  await desktopPage.screenshot({
    path: path.join(RESULTS_DIR, "ui_web_desktop.png"),
    fullPage: true,
  });
  await desktopCtx.close();

  const mobileCtx = await browser.newContext({
    viewport: { width: 390, height: 844 },
  });
  const mobilePage = await mobileCtx.newPage();
  await mobilePage.goto("/");
  await mobilePage.waitForSelector('[data-testid^="bbox-"]', { timeout: 15_000 });
  await expect(mobilePage.getByTestId("mobile-tab-detect")).toBeVisible();

  const hasOverflow = await mobilePage.evaluate(() => {
    return document.documentElement.scrollWidth > window.innerWidth + 2;
  });
  expect(hasOverflow).toBe(false);

  await mobilePage.screenshot({
    path: path.join(RESULTS_DIR, "ui_web_mobile.png"),
    fullPage: true,
  });
  await mobileCtx.close();
});
