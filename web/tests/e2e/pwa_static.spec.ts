import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test } from "@playwright/test";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const WEB_SPACE_DIR = path.resolve(__dirname, "../../web_space");

const MIME_TYPES: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".mjs": "application/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".webmanifest": "application/manifest+json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".wasm": "application/wasm",
  ".onnx": "application/octet-stream",
};

function startPlainStaticServer(rootDir: string): Promise<{
  server: http.Server;
  baseUrl: string;
}> {
  if (!fs.existsSync(path.join(rootDir, "index.html"))) {
    throw new Error(
      `web_space/ build missing at ${rootDir}. Run 'npm run build:space' first.`,
    );
  }

  const server = http.createServer((req, res) => {
    const reqPath = (req.url || "/").split("?")[0];
    const rel = reqPath === "/" ? "index.html" : reqPath.replace(/^\/+/, "");
    let fullPath = path.resolve(rootDir, rel);
    if (!fullPath.startsWith(rootDir)) {
      res.statusCode = 403;
      res.end("Forbidden");
      return;
    }
    if (!fs.existsSync(fullPath) || fs.statSync(fullPath).isDirectory()) {
      fullPath = path.join(rootDir, "index.html");
    }
    const ext = path.extname(fullPath).toLowerCase();
    // Statik host testi: COOP/COEP başlığı BİLİNÇLİ OLARAK gönderilmez
    res.setHeader("Content-Type", MIME_TYPES[ext] || "application/octet-stream");
    res.setHeader("Content-Length", String(fs.statSync(fullPath).size));
    res.setHeader("Cache-Control", "no-store");
    fs.createReadStream(fullPath).pipe(res);
  });

  return new Promise((resolve) => {
    server.listen(0, "localhost", () => {
      const addr = server.address();
      if (!addr || typeof addr === "string") {
        throw new Error("Failed to bind static server");
      }
      resolve({ server, baseUrl: `http://localhost:${addr.port}` });
    });
  });
}

test.describe("Adım 4 — PWA Önbellek Stratejisi, Statik Paket ve WebGPU/WASM Doğrulama", () => {
  let staticServer: http.Server;
  let staticBaseUrl = "";

  test.beforeAll(async () => {
    const started = await startPlainStaticServer(WEB_SPACE_DIR);
    staticServer = started.server;
    staticBaseUrl = started.baseUrl;
  });

  test.afterAll(async () => {
    await new Promise<void>((resolve) => staticServer.close(() => resolve()));
  });

  test("Static web_space runs detection cleanly in WebGPU without COOP/COEP", async ({
    browser,
  }) => {
    const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    const page = await context.newPage();

    await page.goto(`${staticBaseUrl}/?ep=webgpu`);

    // v1.1: uygulama Ana Sayfa'da açılır → Muayene sekmesine geç

    await page.getByTestId("tab-detect").click();
    const isolated = await page.evaluate(() => window.crossOriginIsolated);
    expect(isolated).toBe(false);

    await expect(page.getByTestId("model-download-notice")).toContainText("~65 MB");
    await expect(page.getByTestId("run-detect-btn")).toBeEnabled({ timeout: 15_000 });
    await page.getByTestId("run-detect-btn").click();

    await page.waitForSelector('[data-testid^="bbox-"]', { timeout: 45_000 });
    await expect(page.getByText("ort-webgpu")).toBeVisible();
    await expect(page.getByTestId("model-download-notice")).not.toBeVisible();

    await page.getByTestId("example-val_05_short_01").click();
    await page.waitForSelector('[data-testid^="bbox-"]', { timeout: 30_000 });
    await expect(page.getByText("ort-webgpu")).toBeVisible();

    await context.close();
  });

  test("PWA strategy: (a) no ONNX download on open, (b) download & cache on first detect, (c) works offline after reload", async ({
    browser,
  }) => {
    const context = await browser.newContext({ viewport: { width: 1280, height: 800 } });
    const page = await context.newPage();

    const networkUrls: string[] = [];
    page.on("request", (req) => networkUrls.push(req.url()));

    await page.goto(`${staticBaseUrl}/?ep=wasm`);

    // v1.1: uygulama Ana Sayfa'da açılır → Muayene sekmesine geç

    await page.getByTestId("tab-detect").click();
    expect(await page.evaluate(() => window.crossOriginIsolated)).toBe(false);

    // SW kurulumunun tamamlanmasını bekle (yalnız uygulama kabuğu precache edilmeli)
    await page.waitForFunction(
      async () => {
        if (!("serviceWorker" in navigator)) return false;
        const reg = await navigator.serviceWorker.ready;
        return Boolean(reg.active);
      },
      undefined,
      { timeout: 30_000 },
    );

    // (a) İlk açılışta ONNX ve ORT WASM ağdan indirilmemiş olmalı; kullanıcı bilgilendirilmeli
    await expect(page.getByTestId("model-download-notice")).toContainText("~65 MB");
    expect(networkUrls.some((u) => u.includes("tracera_1024.onnx"))).toBe(false);
    expect(networkUrls.some((u) => u.includes(".wasm"))).toBe(false);

    const cachedBeforeDetect = await page.evaluate(async () => ({
      onnx: Boolean(await caches.match("/models/tracera_1024.onnx", { ignoreSearch: true })),
      wasm: Boolean(await caches.match("/ort/ort-wasm-simd-threaded.jsep.wasm", { ignoreSearch: true })),
    }));
    expect(cachedBeforeDetect.onnx).toBe(false);
    expect(cachedBeforeDetect.wasm).toBe(false);

    // (b) İlk tespitte ONNX + ORT WASM indirilir ve runtime CacheFirst ile önbelleğe alınır
    await expect(page.getByTestId("run-detect-btn")).toBeEnabled({ timeout: 15_000 });
    await page.getByTestId("run-detect-btn").click();
    await page.waitForSelector('[data-testid^="bbox-"]', { timeout: 60_000 });
    await expect(page.getByText("ort-wasm")).toBeVisible();
    await expect(page.getByTestId("model-download-notice")).not.toBeVisible();

    expect(networkUrls.some((u) => u.includes("tracera_1024.onnx"))).toBe(true);
    expect(
      networkUrls.some((u) => u.includes("/ort/ort-wasm-simd-threaded.jsep.wasm")),
    ).toBe(true);

    await page.waitForFunction(
      async () => {
        const modelCached = await caches.match("/models/tracera_1024.onnx", {
          ignoreSearch: true,
        });
        const wasmCached = await caches.match(
          "/ort/ort-wasm-simd-threaded.jsep.wasm",
          { ignoreSearch: true },
        );
        const mjsCached = await caches.match(
          "/ort/ort-wasm-simd-threaded.jsep.mjs",
          { ignoreSearch: true },
        );
        return Boolean(modelCached && wasmCached && mjsCached);
      },
      undefined,
      { timeout: 30_000 },
    );

    // (c) Tarayıcıyı çevrimdışı yap, yenile: indirme bilgisi/çubuğu çıkmasın ve tespit çalışsın
    await context.setOffline(true);

    await page.addInitScript(() => {
      const obs = new MutationObserver(() => {
        const el = document.querySelector('[data-testid="ort-progress"]');
        if (el && /indiriliyor/i.test(el.textContent || "")) {
          (window as unknown as { __sawDownloading?: boolean }).__sawDownloading =
            true;
        }
      });
      obs.observe(document.documentElement, {
        childList: true,
        subtree: true,
        characterData: true,
      });
    });

    await page.reload({ waitUntil: "domcontentloaded" });

    await page.getByTestId("tab-detect").click();
    await page.waitForSelector('[data-testid^="bbox-"]', { timeout: 60_000 });
    await expect(page.getByText("ort-wasm")).toBeVisible();
    await expect(page.getByTestId("model-download-notice")).not.toBeVisible();

    const sawDownloading = await page.evaluate(() =>
      Boolean((window as unknown as { __sawDownloading?: boolean }).__sawDownloading),
    );
    expect(sawDownloading).toBe(false);

    // Çevrimdışıyken kullanıcı düğmeyle de yeniden tespit çalıştırabilmeli
    await page.getByTestId("run-detect-btn").click();
    await page.waitForSelector('[data-testid^="bbox-"]', { timeout: 45_000 });
    await expect(page.getByText("ort-wasm")).toBeVisible();

    await context.close();
  });

  test("Private/unreachable HF Space surfaces explicit unavailable message without silent Mock fallback", async ({
    browser,
  }) => {
    const context = await browser.newContext({ viewport: { width: 1280, height: 800 } });
    const page = await context.newPage();

    await page.route("**/*huggingface.co/**", (route) => route.abort("failed"));
    await page.route("**/*.hf.space/**", (route) => route.abort("failed"));
    await page.route("http://127.0.0.1:8765/**", (route) => route.abort("failed"));

    await page.goto(`${staticBaseUrl}/?ep=wasm`);

    // v1.1: uygulama Ana Sayfa'da açılır → Muayene sekmesine geç

    await page.getByTestId("tab-detect").click();
    await expect(page.getByTestId("run-detect-btn")).toBeEnabled({ timeout: 15_000 });
    await page.getByTestId("run-detect-btn").click();
    await page.waitForSelector('[data-testid="bbox-0"]', { timeout: 60_000 });

    await page.getByTestId("explain-det-0").click();
    await expect(page.getByTestId("explain-error-banner")).toContainText(
      "Bulut açıklaması şu an kullanılamıyor",
      { timeout: 20_000 },
    );
    await expect(page.getByTestId("explain-zoom-img")).not.toBeVisible();

    await context.close();
  });
});
