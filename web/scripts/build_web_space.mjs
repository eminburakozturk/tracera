import { execSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const WEB_DIR = path.resolve(__dirname, "..");
const DIST_DIR = path.join(WEB_DIR, "dist");
const OUT_DIR = path.join(WEB_DIR, "web_space");

const ALLOWED_ORT_FILES = new Set([
  "ort-wasm-simd-threaded.jsep.mjs",
  "ort-wasm-simd-threaded.jsep.wasm",
]);

function walkFiles(dir) {
  const out = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      out.push(...walkFiles(full));
    } else if (entry.isFile()) {
      out.push({ path: full, size: fs.statSync(full).size });
    }
  }
  return out;
}

function measurePrecache(swPath, baseDir) {
  const swText = fs.readFileSync(swPath, "utf8");
  const urls = new Set();
  for (const m of swText.matchAll(/url:\s*"([^"]+)"/g)) {
    urls.add(m[1].replace(/^\/+/, ""));
  }
  let totalBytes = 0;
  for (const rel of urls) {
    const target = path.join(baseDir, rel);
    if (fs.existsSync(target) && fs.statSync(target).isFile()) {
      totalBytes += fs.statSync(target).size;
    }
  }
  return { count: urls.size, totalBytes };
}

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
      const base = path.basename(normalized);
      if (normalized.endsWith("/models/tracera_1024_int8.onnx")) return false;
      if (normalized.endsWith(".map")) return false;
      if (/\/assets\/ort-wasm-.*\.wasm$/i.test(normalized)) return false;
      if (/\/ort\/[^/]+$/i.test(normalized) && !ALLOWED_ORT_FILES.has(base)) {
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
    "ort/ort-wasm-simd-threaded.jsep.mjs",
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

  const precache = measurePrecache(path.join(OUT_DIR, "sw.js"), OUT_DIR);
  const precacheMB = precache.totalBytes / (1024 * 1024);
  console.log(
    `[build_web_space] Precache total: ${precacheMB.toFixed(2)} MB (${precache.count} entries, ${(precache.totalBytes / 1024).toFixed(1)} KiB)`,
  );
  if (precacheMB >= 3.0) {
    throw new Error(
      `[build_web_space] Precache exceeds 3 MB limit: ${precacheMB.toFixed(2)} MB`,
    );
  }

  const allFiles = walkFiles(OUT_DIR).sort((a, b) => b.size - a.size);
  const totalBytes = allFiles.reduce((acc, f) => acc + f.size, 0);
  console.log(
    `[build_web_space] web_space/ total: ${(totalBytes / (1024 * 1024)).toFixed(2)} MB (${allFiles.length} files)`,
  );
  console.log("[build_web_space] Top 15 largest files:");
  for (const item of allFiles.slice(0, 15)) {
    const rel = path.relative(OUT_DIR, item.path).replace(/\\/g, "/");
    console.log(`  ${(item.size / (1024 * 1024)).toFixed(2).padStart(7)} MB  ${rel}`);
  }
}

main();
