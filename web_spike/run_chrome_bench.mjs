import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SPIKE_DIR = __dirname;
const CHROME_PATH = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const VITE_PORT = 5179;
const CDP_PORT = 9233;

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function waitForHttp(url, timeoutMs = 15000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const resp = await fetch(url);
      if (resp.ok) return;
    } catch {
      // retry
    }
    await sleep(250);
  }
  throw new Error(`HTTP hazır olmadı: ${url}`);
}

class CDPClient {
  constructor(wsUrl) {
    this.ws = new WebSocket(wsUrl);
    this.nextId = 1;
    this.pending = new Map();
    this.ready = new Promise((resolve, reject) => {
      this.ws.addEventListener("open", () => resolve());
      this.ws.addEventListener("error", (err) => reject(err));
    });
    this.ws.addEventListener("message", (ev) => {
      const msg = JSON.parse(String(ev.data));
      if (msg.id && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        if (msg.error) {
          reject(new Error(JSON.stringify(msg.error)));
        } else {
          resolve(msg.result);
        }
      }
    });
  }

  async call(method, params = {}) {
    await this.ready;
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }

  async evaluate(expression) {
    const res = await this.call("Runtime.evaluate", {
      expression,
      awaitPromise: true,
      returnByValue: true,
    });
    if (res.exceptionDetails) {
      const errText =
        res.exceptionDetails.exception?.description ||
        res.exceptionDetails.text ||
        JSON.stringify(res.exceptionDetails);
      throw new Error(errText);
    }
    return res.result?.value;
  }

  close() {
    this.ws.close();
  }
}

async function main() {
  if (!fs.existsSync(CHROME_PATH)) {
    throw new Error(`Chrome bulunamadı: ${CHROME_PATH}`);
  }

  const outDir = path.join(SPIKE_DIR, "out");
  fs.mkdirSync(outDir, { recursive: true });

  console.log("🚀 Vite dev sunucusu başlatılıyor...");
  const viteBin = path.join(SPIKE_DIR, "node_modules", "vite", "bin", "vite.js");
  const viteProc = spawn(
    process.execPath,
    [viteBin, "--host", "127.0.0.1", "--port", String(VITE_PORT), "--strictPort"],
    { cwd: SPIKE_DIR, stdio: ["ignore", "pipe", "pipe"] }
  );
  let viteLog = "";
  viteProc.stdout.on("data", (d) => { viteLog += String(d); });
  viteProc.stderr.on("data", (d) => { viteLog += String(d); });

  const userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), "tracera-chrome-"));
  let chromeProc = null;
  let cdp = null;

  try {
    try {
      await waitForHttp(`http://127.0.0.1:${VITE_PORT}/`);
    } catch (e) {
      console.error("Vite log:", viteLog);
      throw e;
    }
    console.log("✅ Vite hazır.");

    console.log("🌐 Desktop Chrome başlatılıyor...");
    chromeProc = spawn(
      CHROME_PATH,
      [
        "--headless=new",
        "--enable-unsafe-webgpu",
        "--ignore-gpu-blocklist",
        `--remote-debugging-port=${CDP_PORT}`,
        `--user-data-dir=${userDataDir}`,
        "--no-first-run",
        "--no-default-browser-check",
        `http://127.0.0.1:${VITE_PORT}/`,
      ],
      { stdio: ["ignore", "pipe", "pipe"] }
    );

    await waitForHttp(`http://127.0.0.1:${CDP_PORT}/json/list`, 20000);
    const targetsResp = await fetch(`http://127.0.0.1:${CDP_PORT}/json/list`);
    const targets = await targetsResp.json();
    const pageTarget = targets.find((t) => t.type === "page") || targets[0];
    if (!pageTarget?.webSocketDebuggerUrl) {
      throw new Error("Chrome page target bulunamadı.");
    }

    cdp = new CDPClient(pageTarget.webSocketDebuggerUrl);
    await cdp.call("Runtime.enable");

    // window.__traceraSpike hazır olana kadar bekle
    for (let i = 0; i < 60; i++) {
      const ready = await cdp.evaluate("Boolean(window.__traceraSpike)");
      if (ready) break;
      await sleep(250);
    }

    const gpuInfo = await cdp.evaluate(`(async () => {
      if (!navigator.gpu) return { available: false, reason: 'navigator.gpu yok' };
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) return { available: false, reason: 'requestAdapter null' };
      return { available: true, info: adapter.info ? { vendor: adapter.info.vendor, arch: adapter.info.architecture, device: adapter.info.device, desc: adapter.info.description } : {} };
    })()`);
    console.log("🖥️ Chrome WebGPU bilgisi:", JSON.stringify(gpuInfo));

    const combos = [
      { pref: "webgpu", modelFile: "tracera_1024.onnx" },
      { pref: "webgpu", modelFile: "tracera_1024_int8.onnx" },
      { pref: "wasm", modelFile: "tracera_1024.onnx" },
      { pref: "wasm", modelFile: "tracera_1024_int8.onnx" },
    ];

    const summary = [];
    for (const c of combos) {
      console.log(`\n⏱️ Benchmark başlıyor: backend=${c.pref}, model=${c.modelFile} ...`);
      try {
        const res = await cdp.evaluate(
          `window.__traceraSpike.benchmarkAll({ modelFile: ${JSON.stringify(c.modelFile)}, pref: ${JSON.stringify(c.pref)}, warmupRuns: 2, timedRunsPerImage: 2 })`
        );
        const outFile = path.join(outDir, `browser_${c.pref}_${c.modelFile.replace(".onnx", "")}.json`);
        fs.writeFileSync(outFile, JSON.stringify(res, null, 2), "utf-8");

        const allMed = [];
        for (const item of res.results) {
          allMed.push(...item.allModelMs);
        }
        allMed.sort((a, b) => a - b);
        const mid = Math.floor(allMed.length / 2);
        const medMs = allMed.length % 2 === 0 ? (allMed[mid - 1] + allMed[mid]) / 2 : allMed[mid];

        console.log(
          `✅ Tamamlandı: actualBackend=${res.actualBackend}, warmup=${res.warmupMs.map((v) => v.toFixed(1)).join(",")} ms, medyan=${medMs.toFixed(2)} ms`
        );
        summary.push({ ...c, status: "ok", actualBackend: res.actualBackend, medyan_model_ms: medMs, outFile });
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.error(`❌ Hata (${c.pref}, ${c.modelFile}):`, msg);
        summary.push({ ...c, status: "error", error: msg });
      }
    }

    fs.writeFileSync(path.join(outDir, "bench_summary.json"), JSON.stringify({ gpuInfo, summary }, null, 2), "utf-8");
    console.log("\n🏁 Tüm Chrome benchmark koşuları tamamlandı.");
  } finally {
    cdp?.close();
    if (chromeProc) chromeProc.kill();
    viteProc.kill();
    try {
      fs.rmSync(userDataDir, { recursive: true, force: true });
    } catch {
      // ignore cleanup lock on Windows
    }
  }
}

main().catch((err) => {
  console.error("FATAL:", err);
  process.exit(1);
});
