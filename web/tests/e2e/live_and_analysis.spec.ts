import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test } from "@playwright/test";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const WEB_SPACE_DIR = path.resolve(__dirname, "../../web_space");
const REPO_ROOT = path.resolve(__dirname, "../../..");
// Resmi sonuç dosyasına yalnız TRACERA_WRITE_RESULTS=1 iken yazılır; aksi halde test-results/ altına.
const FPS_CSV_PATH =
  process.env.TRACERA_WRITE_RESULTS === "1"
    ? path.join(REPO_ROOT, "results", "web_video_fps.csv")
    : path.join(REPO_ROOT, "web", "test-results", "web_video_fps.csv");

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
  ".mp4": "video/mp4",
  ".webm": "video/webm",
  ".wasm": "application/wasm",
  ".onnx": "application/octet-stream",
};

function startStaticServerWithRange(rootDir: string): Promise<{
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

    const stat = fs.statSync(fullPath);
    const totalSize = stat.size;
    const ext = path.extname(fullPath).toLowerCase();
    const contentType = MIME_TYPES[ext] || "application/octet-stream";

    res.setHeader("Accept-Ranges", "bytes");
    res.setHeader("Content-Type", contentType);
    res.setHeader("Cache-Control", "no-store");

    const range = req.headers.range;
    if (range) {
      const m = /bytes=(\d+)-(\d*)/.exec(range);
      if (m) {
        const start = Number(m[1]);
        const end = m[2] ? Math.min(Number(m[2]), totalSize - 1) : totalSize - 1;
        if (start <= end && start < totalSize) {
          res.statusCode = 206;
          res.setHeader("Content-Range", `bytes ${start}-${end}/${totalSize}`);
          res.setHeader("Content-Length", String(end - start + 1));
          fs.createReadStream(fullPath, { start, end }).pipe(res);
          return;
        }
      }
    }

    res.setHeader("Content-Length", String(totalSize));
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

interface BenchmarkRow {
  backend: string;
  ort_ep: string;
  frames: number;
  duration_s: number;
  avg_fps: number;
  rolling_fps: number;
  model_ms: number;
  dropped_frames: number;
  total_boxes: number;
}

test.describe("Adım 5 — Canlı Video Modu (WebGPU & WASM) + Hata Analizi Galerisi", () => {
  let staticServer: http.Server;
  let staticBaseUrl = "";

  test.beforeAll(async () => {
    const started = await startStaticServerWithRange(WEB_SPACE_DIR);
    staticServer = started.server;
    staticBaseUrl = started.baseUrl;
  });

  test.afterAll(async () => {
    await new Promise<void>((resolve) => staticServer.close(() => resolve()));
  });

  test("Video mode plays val_pan.mp4, draws boxes, benchmarks WebGPU & WASM FPS, and renders Error Analysis Gallery", async ({
    browser,
  }) => {
    test.setTimeout(180_000);
    const benchmarkRows: BenchmarkRow[] = [];

    for (const ep of ["webgpu", "wasm"] as const) {
      const context = await browser.newContext({
        viewport: { width: 1440, height: 900 },
      });
      const page = await context.newPage();

      await page.goto(`${staticBaseUrl}/?ep=${ep}`);
      await page.getByTestId("tab-live").click();

      await expect(page.getByTestId("live-camera-note")).toContainText(
        "Deneysel — model yukarıdan çekilmiş yüksek çözünürlüklü kart fotoğraflarıyla eğitildi; telefon kamerasında doğruluk düşebilir.",
      );

      // Warm-up: wait until at least 1 frame finishes so model/WASM init is excluded
      await page.waitForFunction(
        () => {
          const el = document.querySelector('[data-testid="live-det-count"]');
          return Number(el?.getAttribute("data-frames") ?? "0") >= 1;
        },
        undefined,
        { timeout: 60_000 },
      );

      // Reset telemetry by clicking Sample Video button, then measure >= 5s playback
      await page.getByTestId("live-sample-btn").click();
      const tStart = Date.now();
      await page.waitForTimeout(5_200);

      // Ensure at least 2 frames completed (even on slower WASM 1024)
      await page.waitForFunction(
        () => {
          const el = document.querySelector('[data-testid="live-det-count"]');
          return Number(el?.getAttribute("data-frames") ?? "0") >= 2;
        },
        undefined,
        { timeout: 30_000 },
      );
      const durationSec = (Date.now() - tStart) / 1000;

      const stats = await page.evaluate(() => {
        const detEl = document.querySelector('[data-testid="live-det-count"]');
        const fpsEl = document.querySelector('[data-testid="live-fps"]');
        const msEl = document.querySelector('[data-testid="live-model-ms"]');
        const dropEl = document.querySelector('[data-testid="live-dropped"]');
        const backendEl = document.querySelector('[data-testid="live-backend"]');
        const canvas = document.querySelector(
          '[data-testid="live-overlay-canvas"]',
        ) as HTMLCanvasElement | null;

        let hasCanvasPixels = false;
        if (canvas && canvas.width > 0 && canvas.height > 0) {
          const ctx = canvas.getContext("2d");
          if (ctx) {
            const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
            for (let i = 3; i < data.length; i += 4) {
              if (data[i] > 0) {
                hasCanvasPixels = true;
                break;
              }
            }
          }
        }

        const fpsNum = Number.parseFloat((fpsEl?.textContent || "").replace(/[^\d.]/g, "")) || 0;
        const msNum = Number.parseFloat((msEl?.textContent || "").replace(/[^\d.]/g, "")) || 0;
        return {
          frames: Number(detEl?.getAttribute("data-frames") ?? "0"),
          totalBoxes: Number(detEl?.getAttribute("data-total-boxes") ?? "0"),
          dropped: Number(dropEl?.textContent ?? "0"),
          rollingFps: fpsNum,
          modelMs: msNum,
          backend: (backendEl?.textContent || "").trim(),
          hasCanvasPixels,
        };
      });

      expect(stats.backend).toBe(`ort-${ep}`);
      expect(stats.frames).toBeGreaterThanOrEqual(2);
      expect(stats.totalBoxes).toBeGreaterThanOrEqual(1);
      expect(stats.hasCanvasPixels).toBe(true);

      if (ep === "wasm") {
        await expect(page.getByTestId("live-wasm-warning")).toContainText(
          "WebGPU yok; canlı mod bu cihazda çok yavaş (<1 FPS) çalışır.",
        );
      } else {
        await expect(page.getByTestId("live-wasm-warning")).toHaveCount(0);
      }

      // Verify pause button toggles cleanly
      await page.getByTestId("live-pause-btn").click();
      await expect(page.getByTestId("live-pause-btn")).toContainText("Devam Et");
      await page.getByTestId("live-pause-btn").click();
      await expect(page.getByTestId("live-pause-btn")).toContainText("Duraklat");

      const avgFps = stats.frames / durationSec;
      benchmarkRows.push({
        backend: ep,
        ort_ep: stats.backend,
        frames: stats.frames,
        duration_s: Number(durationSec.toFixed(2)),
        avg_fps: Number(avgFps.toFixed(2)),
        rolling_fps: Number(stats.rollingFps.toFixed(2)),
        model_ms: Number(stats.modelMs.toFixed(1)),
        dropped_frames: stats.dropped,
        total_boxes: stats.totalBoxes,
      });

      if (ep === "webgpu") {
        // Check Error Analysis Gallery inside Sonuçlar tab
        await page.getByTestId("tab-results").click();
        const section = page.getByTestId("error-analysis-section");
        await section.scrollIntoViewIfNeeded();
        await expect(section).toBeVisible();
        await expect(section).not.toContainText("158");
        await expect(page.getByTestId("fp-total-test")).toContainText("142");
        await expect(page.getByTestId("fp-total-val")).toContainText("16");
        await expect(page.getByTestId("template-spot-mouse_bite")).toContainText(
          "Evet (0,5569)",
        );
        await expect(page.getByTestId("template-spot-spur")).toContainText(
          "Hayır (<0,395)",
        );
        await expect(page.getByTestId("quant-parity-table")).toContainText("0,395");
        await expect(page.getByTestId("device-bench-btn")).toContainText(
          "Bu cihazda hız testi",
        );

        const imagesLoaded = await page.evaluate(() => {
          const imgs = Array.from(
            document.querySelectorAll('[data-testid="error-analysis-section"] img'),
          ) as HTMLImageElement[];
          return imgs.length === 4 && imgs.every((img) => img.complete && img.naturalWidth > 0);
        });
        expect(imagesLoaded).toBe(true);
      }

      await context.close();
    }

    const csvLines = [
      "backend,ort_ep,frames,duration_s,avg_fps,rolling_fps,model_ms,dropped_frames,total_boxes",
      ...benchmarkRows.map((r) =>
        [
          r.backend,
          r.ort_ep,
          r.frames,
          r.duration_s,
          r.avg_fps,
          r.rolling_fps,
          r.model_ms,
          r.dropped_frames,
          r.total_boxes,
        ].join(","),
      ),
    ];
    fs.mkdirSync(path.dirname(FPS_CSV_PATH), { recursive: true });
    fs.writeFileSync(FPS_CSV_PATH, `${csvLines.join("\n")}\n`, "utf-8");
  });
});
