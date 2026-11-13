import fs from "node:fs";
import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";
import { VitePWA } from "vite-plugin-pwa";

function ortAssetsPlugin(): Plugin {
  const ortDist = path.resolve(__dirname, "node_modules/onnxruntime-web/dist");
  const spikeValDir = path.resolve(__dirname, "../web_spike/val_images");
  const processedValDir = path.resolve(__dirname, "../data/yolo/grouped/val/images");

  const mimeMap: Record<string, string> = {
    ".wasm": "application/wasm",
    ".mjs": "application/javascript; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".jpg": "image/jpeg",
    ".json": "application/json; charset=utf-8",
  };

  return {
    name: "tracera-ort-assets",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const cleanUrl = (req.url || "").split("?")[0];
        if (cleanUrl.startsWith("/ort/")) {
          const fname = path.basename(cleanUrl);
          const full = path.join(ortDist, fname);
          if (fs.existsSync(full) && fs.statSync(full).isFile()) {
            const ext = path.extname(fname).toLowerCase();
            res.setHeader("Content-Type", mimeMap[ext] || "application/octet-stream");
            fs.createReadStream(full).pipe(res);
            return;
          }
        }
        // Parite testi için spike val görüntüleri (yalnız geliştirme/test sunucusu)
        if (cleanUrl.startsWith("/spike-val/")) {
          const fname = path.basename(cleanUrl);
          const candidates = [
            path.join(spikeValDir, fname),
            path.join(processedValDir, fname),
          ];
          const full = candidates.find(
            (p) => fs.existsSync(p) && fs.statSync(p).isFile(),
          );
          if (full) {
            res.setHeader("Content-Type", "image/jpeg");
            fs.createReadStream(full).pipe(res);
            return;
          }
        }
        next();
      });
    },
    writeBundle(options) {
      const outDir = options.dir ? path.resolve(options.dir, "ort") : path.resolve(__dirname, "dist/ort");
      if (!fs.existsSync(ortDist)) return;
      fs.mkdirSync(outDir, { recursive: true });
      const requiredOrtFiles = new Set([
        "ort-wasm-simd-threaded.jsep.mjs",
        "ort-wasm-simd-threaded.jsep.wasm",
      ]);
      for (const fname of fs.readdirSync(ortDist)) {
        if (requiredOrtFiles.has(fname)) {
          fs.copyFileSync(path.join(ortDist, fname), path.join(outDir, fname));
        }
      }

      // Vite'ın assets/ içine kopyaladığı kullanılmayan kopya ort-wasm-* ve .map dosyalarını temizle
      const assetsDir = options.dir
        ? path.resolve(options.dir, "assets")
        : path.resolve(__dirname, "dist/assets");
      if (fs.existsSync(assetsDir)) {
        for (const fname of fs.readdirSync(assetsDir)) {
          if (
            (fname.startsWith("ort-wasm-") && fname.endsWith(".wasm")) ||
            fname.endsWith(".map")
          ) {
            fs.rmSync(path.join(assetsDir, fname), { force: true });
          }
        }
      }
    },
  };
}

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    ortAssetsPlugin(),
    VitePWA({
      registerType: "autoUpdate",
      includeAssets: [
        "icons/icon.svg",
        "icons/icon-192.png",
        "icons/icon-512.png",
      ],
      manifest: {
        name: "Tracera — Açıklanabilir PCB Hata Tespiti",
        short_name: "Tracera",
        description:
          "Baskı devre kartı (PCB) üretim kusuru tespiti ve LayerCAM açıklanabilirlik aracı.",
        start_url: "/",
        scope: "/",
        display: "standalone",
        background_color: "#090d14",
        theme_color: "#090d14",
        lang: "tr",
        icons: [
          {
            src: "/icons/icon-192.png",
            sizes: "192x192",
            type: "image/png",
            purpose: "any",
          },
          {
            src: "/icons/icon-512.png",
            sizes: "512x512",
            type: "image/png",
            purpose: "any maskable",
          },
          {
            src: "/icons/icon.svg",
            sizes: "any",
            type: "image/svg+xml",
            purpose: "any",
          },
        ],
      },
      workbox: {
        maximumFileSizeToCacheInBytes: 3 * 1024 * 1024,
        globPatterns: [
          "**/*.{js,css,html,ico,png,svg,webmanifest}",
          "metrics.json",
          "models/meta.json",
        ],
        globIgnores: [
          "**/tracera_1024_int8.onnx",
          "**/*.map",
          "models/*.onnx",
          "ort/**",
          "examples/**",
        ],
        navigateFallback: "index.html",
        runtimeCaching: [
          {
            urlPattern: /\/models\/.*\.onnx$/i,
            handler: "CacheFirst",
            options: {
              cacheName: "tracera-models-v1",
              cacheableResponse: { statuses: [0, 200] },
              expiration: {
                maxEntries: 4,
                maxAgeSeconds: 30 * 24 * 60 * 60,
              },
            },
          },
          {
            urlPattern: /\/ort\/.*\.(wasm|mjs)$/i,
            handler: "CacheFirst",
            options: {
              cacheName: "tracera-ort-v1",
              cacheableResponse: { statuses: [0, 200] },
              expiration: {
                maxEntries: 8,
                maxAgeSeconds: 30 * 24 * 60 * 60,
              },
            },
          },
          {
            urlPattern: /\/examples\/.*\.(jpg|jpeg|png)$/i,
            handler: "CacheFirst",
            options: {
              cacheName: "tracera-examples-v1",
              cacheableResponse: { statuses: [0, 200] },
              expiration: {
                maxEntries: 16,
                maxAgeSeconds: 30 * 24 * 60 * 60,
              },
            },
          },
        ],
      },
    }),
  ],
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },
  worker: {
    format: "es",
  },
  server: {
    port: 5173,
    host: "localhost",
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "credentialless",
    },
  },
  preview: {
    port: 5173,
    host: "localhost",
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "credentialless",
    },
  },
});
