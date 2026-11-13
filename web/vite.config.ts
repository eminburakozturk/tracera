import fs from "node:fs";
import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";

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
      for (const fname of fs.readdirSync(ortDist)) {
        if (fname.endsWith(".wasm") || fname.endsWith(".mjs")) {
          fs.copyFileSync(path.join(ortDist, fname), path.join(outDir, fname));
        }
      }
    },
  };
}

export default defineConfig({
  plugins: [react(), tailwindcss(), ortAssetsPlugin()],
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
