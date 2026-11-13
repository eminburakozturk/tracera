import fs from "node:fs";
import path from "node:path";
import { expect, test, type Locator, type Page } from "@playwright/test";

const DEMO_DIR = path.resolve(process.cwd(), "../results/demo");

async function installCursorOverlay(page: Page) {
  await page.addInitScript(() => {
    const setup = () => {
      if (document.getElementById("demo-cursor-layer")) return;
      const layer = document.createElement("div");
      layer.id = "demo-cursor-layer";
      layer.style.cssText =
        "position:fixed;inset:0;pointer-events:none;z-index:2147483647;overflow:hidden;";

      const style = document.createElement("style");
      style.textContent = `
        @keyframes demoPulse {
          0% { transform: translate(-50%, -50%) scale(0.45); opacity: 0.95; }
          100% { transform: translate(-50%, -50%) scale(1.7); opacity: 0; }
        }
      `;
      document.head.appendChild(style);

      const cursor = document.createElement("div");
      cursor.id = "demo-cursor";
      cursor.style.cssText =
        "position:fixed;left:640px;top:400px;width:24px;height:24px;transform:translate(-3px,-2px);transition:left 45ms linear,top 45ms linear;filter:drop-shadow(0 2px 4px rgba(0,0,0,0.65));";
      cursor.innerHTML = `<svg viewBox="0 0 24 24" width="24" height="24" fill="none">
        <path d="M4 2L20 11.5L12.5 13.5L9.5 21L4 2Z" fill="#38bdf8" stroke="#0f172a" stroke-width="1.8" stroke-linejoin="round"/>
      </svg>`;

      layer.appendChild(cursor);
      document.body.appendChild(layer);

      window.addEventListener(
        "mousemove",
        (e) => {
          cursor.style.left = `${e.clientX}px`;
          cursor.style.top = `${e.clientY}px`;
        },
        { passive: true },
      );

      window.addEventListener(
        "mousedown",
        (e) => {
          const ring = document.createElement("div");
          ring.style.cssText = `position:fixed;left:${e.clientX}px;top:${e.clientY}px;width:30px;height:30px;border-radius:9999px;border:2.5px solid #38bdf8;background:rgba(56,189,248,0.18);animation:demoPulse 420ms ease-out forwards;pointer-events:none;`;
          layer.appendChild(ring);
          setTimeout(() => ring.remove(), 460);
        },
        { passive: true },
      );
    };

    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", setup, { once: true });
    } else {
      setup();
    }
  });
}

async function moveAndClick(page: Page, locator: Locator, pauseAfterMs = 1000) {
  await locator.scrollIntoViewIfNeeded();
  const box = await locator.boundingBox();
  if (!box) {
    throw new Error("Target locator has no bounding box");
  }
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;
  await page.mouse.move(cx, cy, { steps: 18 });
  await page.waitForTimeout(180);
  await page.mouse.click(cx, cy);
  await page.waitForTimeout(pauseAfterMs);
}

async function dragSliderToRatio(
  page: Page,
  sliderThumb: Locator,
  sliderTrackBox: { x: number; y: number; width: number; height: number },
  targetRatio: number,
  pauseAfterMs = 1100,
) {
  const thumbBox = await sliderThumb.boundingBox();
  if (!thumbBox) throw new Error("Slider thumb missing");
  const startX = thumbBox.x + thumbBox.width / 2;
  const startY = thumbBox.y + thumbBox.height / 2;
  const endX = sliderTrackBox.x + sliderTrackBox.width * targetRatio;
  const endY = sliderTrackBox.y + sliderTrackBox.height / 2;

  await page.mouse.move(startX, startY, { steps: 14 });
  await page.waitForTimeout(150);
  await page.mouse.down();
  await page.mouse.move(endX, endY, { steps: 22 });
  await page.waitForTimeout(120);
  await page.mouse.up();
  await page.waitForTimeout(pauseAfterMs);
}

test("README için otomatik ~40 sn tam akış demo videosu kaydı", async ({
  browser,
}) => {
  fs.mkdirSync(DEMO_DIR, { recursive: true });

  // 1) Vite/API ısınması: kaydın ilk karesi beyaz/boş olmasın
  const warmCtx = await browser.newContext({
    viewport: { width: 1280, height: 800 },
    colorScheme: "dark",
  });
  const warmPage = await warmCtx.newPage();
  await warmPage.goto("/");
  await expect(warmPage.getByTestId("top-bar")).toBeVisible({ timeout: 20_000 });
  await warmCtx.close();

  // 2) Kayıt context'i (1280x800, koyu tema)
  const recordCtx = await browser.newContext({
    viewport: { width: 1280, height: 800 },
    colorScheme: "dark",
    recordVideo: {
      dir: DEMO_DIR,
      size: { width: 1280, height: 800 },
    },
  });
  const ctxCreatedMs = Date.now();
  const page = await recordCtx.newPage();
  await installCursorOverlay(page);

  await page.goto("/");
  await expect(page.getByTestId("top-bar")).toBeVisible({ timeout: 20_000 });
  // İlk boyama tamamlanana kadar bekle; başlangıçtaki boş kareler trim için kaydedilir
  await page.evaluate(
    () =>
      new Promise<void>((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(() => resolve())),
      ),
  );
  const firstPaintedSec = Math.max(0, (Date.now() - ctxCreatedMs) / 1000);

  // Kişisel bilgi sızıntısı olmadığını doğrula
  const bodyText = await page.locator("body").innerText();
  expect(bodyText).not.toMatch(/C:\\Users\\Burak/i);

  // --- ADIM 1: Tespit sekmesi + Yerel FastAPI motoru ---
  await page.waitForTimeout(1000);
  await moveAndClick(page, page.getByTestId("backend-select"), 700);
  const apiOption = page.getByRole("option", { name: /FastAPI/i });
  await moveAndClick(page, apiOption, 1000);
  await expect(page.getByTestId("api-health-badge")).toContainText(/Açık|Online/i, {
    timeout: 15_000,
  });

  // Val örneği seç (05_short) -> Tespit Et -> kutular çıksın
  const secondExample = page.locator('[data-testid^="example-"]').nth(1);
  await moveAndClick(page, secondExample, 1100);
  await moveAndClick(page, page.getByTestId("run-detect-btn"), 600);
  await page.waitForSelector('[data-testid="bbox-0"]', { timeout: 20_000 });
  await page.waitForTimeout(1200);

  // İlk tespit kutusuna tıkla (seçim)
  const firstBox = page.locator('[data-testid="bbox-0"] rect').first();
  await moveAndClick(page, firstBox, 1200);

  // Eşik kaydırıcısını 0.395'ten ~0.600'a çek, sonra geri 0.395'e döndür
  const confRoot = page
    .locator('[aria-label="Güven Eşiği"], [aria-label="Confidence Threshold"]')
    .first();
  await confRoot.scrollIntoViewIfNeeded();
  const confThumb = confRoot.getByRole("slider").first();
  const confTrackBox = await confRoot.boundingBox();
  if (!confTrackBox) throw new Error("Confidence slider track missing");

  // min=0.05, max=0.95 -> 0.60 ratio = (0.60 - 0.05) / 0.90 = 0.611
  await dragSliderToRatio(page, confThumb, confTrackBox, 0.611, 1400);
  // Geri 0.395'e (ratio = (0.395 - 0.05) / 0.90 = 0.3833)
  await dragSliderToRatio(page, confThumb, confTrackBox, 0.3833, 1100);

  // --- ADIM 2: Açıkla (Yerel API LayerCAM) + opaklık kaydırıcısı ---
  await moveAndClick(page, page.getByTestId("explain-det-0"), 400);
  await expect(page.getByTestId("explain-overlay-img")).toBeVisible({
    timeout: 25_000,
  });
  await expect(page.getByTestId("explain-source-badge")).toContainText(/FastAPI/i);
  await page.waitForTimeout(1400);

  const opacityRoot = page
    .locator('[aria-label="Harita Opaklığı"], [aria-label="Heatmap Opacity"]')
    .first();
  await opacityRoot.scrollIntoViewIfNeeded();
  const opacityThumb = opacityRoot.getByRole("slider").first();
  const opacityTrackBox = await opacityRoot.boundingBox();
  if (!opacityTrackBox) throw new Error("Opacity slider track missing");

  // Opaklığı %35'e çek, sonra %85'e çıkar
  await dragSliderToRatio(page, opacityThumb, opacityTrackBox, 0.28, 1100);
  await dragSliderToRatio(page, opacityThumb, opacityTrackBox, 0.85, 1300);
  const steps12EndSec = (Date.now() - ctxCreatedMs) / 1000;

  // --- ADIM 3: Canlı video 6 sn oynar (FPS göstergesi görünür) ---
  await moveAndClick(page, page.getByTestId("tab-live"), 1000);
  await moveAndClick(page, page.getByTestId("live-sample-btn"), 500);
  await page.waitForFunction(
    () =>
      Number(
        document
          .querySelector('[data-testid="live-det-count"]')
          ?.getAttribute("data-frames") ?? "0",
      ) >= 1,
    undefined,
    { timeout: 25_000 },
  );
  await page.waitForTimeout(6000);

  // --- ADIM 4: Sonuçlar sekmesi -> yavaşça aşağı kaydır (CV tablosu, hata galerisi) ---
  await moveAndClick(page, page.getByTestId("tab-results"), 1400);
  for (let i = 0; i < 6; i++) {
    await page.evaluate(() => window.scrollBy({ top: 380, behavior: "smooth" }));
    await page.waitForTimeout(650);
  }
  await page
    .getByTestId("error-analysis-section")
    .scrollIntoViewIfNeeded();
  await page.waitForTimeout(1800);

  const video = page.video();
  await page.close();
  await recordCtx.close();

  if (!video) throw new Error("Playwright video was not recorded");
  const outWebm = path.join(DEMO_DIR, "tracera_demo.webm");
  await video.saveAs(outWebm);

  // Playwright'ın oluşturduğu rastgele isimli geçici webm dosyasını temizle
  const rawPath = await video.path();
  if (rawPath !== outWebm && fs.existsSync(rawPath)) {
    fs.unlinkSync(rawPath);
  }

  fs.writeFileSync(
    path.join(DEMO_DIR, "demo_meta.json"),
    JSON.stringify(
      {
        firstPaintedSec: Number(firstPaintedSec.toFixed(2)),
        steps12EndSec: Number(steps12EndSec.toFixed(2)),
        totalRecordedSec: Number(((Date.now() - ctxCreatedMs) / 1000).toFixed(2)),
      },
      null,
      2,
    ),
    "utf-8",
  );
});
