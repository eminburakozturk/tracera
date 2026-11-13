import { execSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const WEB_DIR = path.resolve(__dirname, "..");
const DIST_DIR = path.join(WEB_DIR, "dist");
const OUT_DIR = path.join(WEB_DIR, "web_space");

function main() {
  console.log("[build_web_space] Running npm run build...");
  execSync("npm run build", { cwd: WEB_DIR, stdio: "inherit" });

  if (!fs.existsSync(DIST_DIR)) {
    throw new Error(`dist/ not found at ${DIST_DIR}`);
  }

  if (fs.existsSync(OUT_DIR)) {
    fs.rmSync(OUT_DIR, { recursive: true, force: true });
  }
  fs.mkdirSync(OUT_DIR, { recursive: true });

  fs.cpSync(DIST_DIR, OUT_DIR, {
    recursive: true,
    filter: (src) => {
      const normalized = src.replace(/\\/g, "/");
      // Kullanılmayan quantized ONNX modelini statik Space paketine koyma
      if (normalized.endsWith("/models/tracera_1024_int8.onnx")) {
        return false;
      }
      return true;
    },
  });

  const requiredFiles = [
    "index.html",
    "manifest.webmanifest",
    "sw.js",
    "metrics.json",
    "models/meta.json",
    "models/tracera_1024.onnx",
    "ort/ort-wasm-simd-threaded.jsep.wasm",
  ];
  for (const rel of requiredFiles) {
    const target = path.join(OUT_DIR, rel);
    if (!fs.existsSync(target)) {
      throw new Error(`[build_web_space] Missing required artifact: ${rel}`);
    }
  }

  const readmeContent = [
    "---",
    "title: Tracera Web",
    "emoji: 🔬",
    "colorFrom: blue",
    "colorTo: slate",
    "sdk: static",
    "app_file: index.html",
    "pinned: false",
    "license: agpl-3.0",
    "---",
    "",
    "# Tracera — Açıklanabilir PCB Hata Tespiti",
    "",
    "Baskı devre kartlarında (PCB) 6 üretim kusurunu tarayıcıda ONNX Runtime Web (WebGPU/WASM) ile tespit eden, yerel FastAPI sunucusu veya Gradio Space (`ebboo/tracera`) üzerinden LayerCAM açıklaması üreten PWA uygulaması.",
    "",
  ].join("\n");

  fs.writeFileSync(path.join(OUT_DIR, "README.md"), readmeContent, {
    encoding: "utf8",
  });

  const modelSizeMb = (
    fs.statSync(path.join(OUT_DIR, "models/tracera_1024.onnx")).size /
    (1024 * 1024)
  ).toFixed(2);
  console.log(
    `[build_web_space] Ready at ${OUT_DIR} (tracera_1024.onnx: ${modelSizeMb} MB)`,
  );
}

main();
