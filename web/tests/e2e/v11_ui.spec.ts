import fs from "node:fs";
import path from "node:path";
import AxeBuilder from "@axe-core/playwright";
import { expect, type Page, test } from "@playwright/test";

const OUT_DIR = path.resolve(process.cwd(), "../results/ui_v11");

const SCREENS = [
  { id: "home", desktopTestId: "tab-home", mobileTestId: "mobile-tab-home", viaMore: false },
  { id: "detect", desktopTestId: "tab-detect", mobileTestId: "mobile-tab-detect", viaMore: false },
  { id: "explain", desktopTestId: "tab-explain", mobileTestId: "mobile-tab-explain", viaMore: false },
  { id: "batch", desktopTestId: "tab-batch", mobileTestId: "mobile-tab-batch", viaMore: false },
  { id: "live", desktopTestId: "tab-live", mobileTestId: "more-link-live", viaMore: true },
  { id: "history", desktopTestId: "tab-history", mobileTestId: "more-link-history", viaMore: true },
  { id: "guide", desktopTestId: "tab-guide", mobileTestId: "more-link-guide", viaMore: true },
  { id: "results", desktopTestId: "tab-results", mobileTestId: "more-link-results", viaMore: true },
  { id: "about", desktopTestId: "tab-about", mobileTestId: "more-link-about", viaMore: true },
  { id: "settings", desktopTestId: "tab-settings", mobileTestId: "more-link-settings", viaMore: true },
] as const;

async function assertNoContrastViolations(page: Page, contextLabel: string) {
  const results = await new AxeBuilder({ page })
    .withRules(["color-contrast"])
    .analyze();
  expect(
    results.violations,
    `WCAG AA color-contrast violations on ${contextLabel}: ${JSON.stringify(results.violations, null, 2)}`,
  ).toEqual([]);
}

async function assertNoHorizontalOverflow(page: Page, contextLabel: string) {
  const overflow = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    innerWidth: window.innerWidth,
  }));
  expect(
    overflow.scrollWidth,
    `Horizontal overflow on ${contextLabel}: scrollWidth=${overflow.scrollWidth} > innerWidth=${overflow.innerWidth}`,
  ).toBeLessThanOrEqual(overflow.innerWidth + 2);
}

test.describe("Tracera v1.1 Tam Ekran Matrisi: Axe WCAG AA (Koyu + Açık), Mobil 390px Taşma ve Ekran Görüntüleri", () => {
  test.beforeAll(() => {
    fs.mkdirSync(OUT_DIR, { recursive: true });
  });

  for (const theme of ["dark", "light"] as const) {
    test(`Masaüstü (1440x900) — ${theme} tema: tüm 10 ekran WCAG AA kontrast + ekran görüntüleri`, async ({
      browser,
    }) => {
      test.setTimeout(120_000);
      const context = await browser.newContext({
        viewport: { width: 1440, height: 900 },
        colorScheme: theme,
      });
      const page = await context.newPage();
      await page.goto("/");

      // Önce Mock motoru seçip tespit ve ısı haritası üretelim ki ekranlar dolu görünsün
      await page.getByTestId("backend-select").click();
      await page.getByRole("option", { name: /Mock/i }).click();
      await page.waitForSelector('[data-testid^="bbox-"]', { timeout: 15_000 });
      await page.getByTestId("explain-det-0").click();
      await expect(page.getByTestId("explain-overlay-img")).toBeVisible({
        timeout: 15_000,
      });

      for (const scr of SCREENS) {
        await page.getByTestId(scr.desktopTestId).click();
        await page.waitForTimeout(250);
        await assertNoContrastViolations(page, `desktop-${theme}-${scr.id}`);
        await page.screenshot({
          path: path.join(OUT_DIR, `${scr.id}_desktop_${theme}.png`),
          fullPage: false,
        });
      }

      await context.close();
    });

    test(`Mobil (390x844) — ${theme} tema: tüm 10 ekran WCAG AA kontrast + yatay taşma yok + ekran görüntüleri`, async ({
      browser,
    }) => {
      test.setTimeout(120_000);
      const context = await browser.newContext({
        viewport: { width: 390, height: 844 },
        colorScheme: theme,
      });
      const page = await context.newPage();
      await page.goto("/");

      await page.getByTestId("backend-select").click();
      await page.getByRole("option", { name: /Mock/i }).click();
      await page.waitForSelector('[data-testid^="bbox-"]', { timeout: 15_000 });
      await page.getByTestId("explain-det-0").click();
      await expect(page.getByTestId("explain-overlay-img")).toBeVisible({
        timeout: 15_000,
      });

      for (const scr of SCREENS) {
        if (scr.viaMore) {
          await page.getByTestId("mobile-tab-more").click();
          await page.waitForTimeout(150);
          await page.getByTestId(scr.mobileTestId).click();
        } else {
          await page.getByTestId(scr.mobileTestId).click();
        }
        await page.waitForTimeout(250);
        await assertNoHorizontalOverflow(page, `mobile-${theme}-${scr.id}`);
        await assertNoContrastViolations(page, `mobile-${theme}-${scr.id}`);
        await page.screenshot({
          path: path.join(OUT_DIR, `${scr.id}_mobile_${theme}.png`),
          fullPage: false,
        });
      }

      await context.close();
    });
  }
});
