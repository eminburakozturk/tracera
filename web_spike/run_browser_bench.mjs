import { execFileSync, spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const SPIKE_DIR = path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1'));
const SAMPLES_PATH = path.join(SPIKE_DIR, 'samples.json');
const OUT_DIR = path.join(SPIKE_DIR, 'browser_out');
const CHROME_PATH = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const VITE_PORT = 5179;
const CDP_PORT = 9292;

async function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function waitForHttp(url, timeoutMs = 20000) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    try {
      const resp = await fetch(url);
      if (resp.ok) return;
    } catch {
      // retry
    }
    await sleep(250);
  }
  throw new Error(`Zaman aşımı: ${url}`);
}

class CdpClient {
  constructor(wsUrl) {
    this.ws = new WebSocket(wsUrl);
    this.nextId = 1;
    this.pending = new Map();
    this.ready = new Promise((resolve, reject) => {
      this.ws.addEventListener('open', () => resolve());
      this.ws.addEventListener('error', (err) => reject(err));
    });
    this.ws.addEventListener('message', (ev) => {
      const msg = JSON.parse(String(ev.data));
      if (msg.id && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        if (msg.error) {
          reject(new Error(`CDP error: ${JSON.stringify(msg.error)}`));
        } else {
          resolve(msg.result);
        }
      }
    });
  }

  async send(method, params = {}) {
    await this.ready;
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }

  async evalAsync(expression, timeoutMs = 90000) {
    const evalPromise = this.send('Runtime.evaluate', {
      expression,
      awaitPromise: true,
      returnByValue: true,
    });
    const res = await Promise.race([
      evalPromise,
      sleep(timeoutMs).then(() => {
        throw new Error(`CDP eval zaman aşımı (${timeoutMs} ms)`);
      }),
    ]);
    if (res.exceptionDetails) {
      throw new Error(
        res.exceptionDetails.exception?.description ||
          res.exceptionDetails.text ||
          JSON.stringify(res.exceptionDetails),
      );
    }
    return res.result?.value;
  }

  close() {
    this.ws.close();
  }
}

async function main() {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  const samples = JSON.parse(fs.readFileSync(SAMPLES_PATH, 'utf-8'));

  const viteBin = path.join(SPIKE_DIR, 'node_modules', 'vite', 'bin', 'vite.js');
  const viteProc = spawn(process.execPath, [viteBin, '--port', String(VITE_PORT), '--strictPort'], {
    cwd: SPIKE_DIR,
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  let chromeProc = null;
  const userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'tracera-chrome-'));

  try {
    await waitForHttp(`http://127.0.0.1:${VITE_PORT}/samples.json`);
    console.log('Vite sunucusu hazır:', `http://127.0.0.1:${VITE_PORT}`);

    const chromeArgs = [
      `--remote-debugging-port=${CDP_PORT}`,
      `--user-data-dir=${userDataDir}`,
      '--no-first-run',
      '--no-default-browser-check',
      '--enable-unsafe-webgpu',
      '--ignore-gpu-blocklist',
      '--window-size=1280,900',
      `http://127.0.0.1:${VITE_PORT}/`,
    ];
    chromeProc = spawn(CHROME_PATH, chromeArgs, {
      stdio: ['ignore', 'ignore', 'ignore'],
    });

    await waitForHttp(`http://127.0.0.1:${CDP_PORT}/json/list`, 20000);
    let pageTarget = null;
    for (let i = 0; i < 40; i++) {
      const targets = await (await fetch(`http://127.0.0.1:${CDP_PORT}/json/list`)).json();
      pageTarget = targets.find((t) => t.type === 'page' && t.url.includes(`127.0.0.1:${VITE_PORT}`));
      if (pageTarget?.webSocketDebuggerUrl) break;
      await sleep(250);
    }
    if (!pageTarget) {
      throw new Error('Chrome hedef sayfası CDP üzerinden bulunamadı.');
    }

    const cdp = new CdpClient(pageTarget.webSocketDebuggerUrl);
    await cdp.send('Runtime.enable');

    // __TRACERA_SPIKE__ hazır olana kadar bekle
    for (let i = 0; i < 40; i++) {
      const ok = await cdp.evalAsync('Boolean(window.__TRACERA_SPIKE__)');
      if (ok) break;
      await sleep(250);
    }

    const gpuAvailable = await cdp.evalAsync('window.__TRACERA_SPIKE__.hasWebGPU()');
    const ua = await cdp.evalAsync('navigator.userAgent');
    console.log('Chrome UA:', ua);
    console.log('WebGPU kullanılabilir mi:', gpuAvailable);

    const combos = [
      { backend: 'webgpu', model: 'fp32' },
      { backend: 'wasm', model: 'fp32' },
      { backend: 'wasm', model: 'int8' },
      { backend: 'webgpu', model: 'int8' },
    ];

    const summary = [];
    for (const combo of combos) {
      console.log(`\n>>> Çalıştırılıyor: backend=${combo.backend}, model=${combo.model} ...`);
      try {
        const res = await cdp.evalAsync(
          `window.__TRACERA_SPIKE__.runBatch(${JSON.stringify({
            backend: combo.backend,
            model: combo.model,
            samples,
            warmupRuns: 2,
            benchRuns: 10,
          })})`,
          combo.backend === 'webgpu' && combo.model === 'int8' ? 25000 : 90000,
        );
        const outFile = path.join(OUT_DIR, `browser_${combo.backend}_${combo.model}.json`);
        fs.writeFileSync(outFile, JSON.stringify({ ...res, userAgent: ua, status: 'ok' }, null, 2));
        console.log(
          `[BAŞARILI] ${combo.backend} / ${combo.model} | active=${res.activeBackend} | ` +
            `model medyan=${res.medianModelMs} ms | total medyan=${res.medianTotalMs} ms`,
        );
        summary.push({
          backend: res.activeBackend,
          requestedBackend: combo.backend,
          model: combo.model,
          status: 'ok',
          medianModelMs: res.medianModelMs,
          medianTotalMs: res.medianTotalMs,
        });
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.error(`[HATA] ${combo.backend} / ${combo.model}: ${msg}`);
        const outFile = path.join(OUT_DIR, `browser_${combo.backend}_${combo.model}.json`);
        fs.writeFileSync(
          outFile,
          JSON.stringify(
            {
              backend: combo.backend,
              model: combo.model,
              status: 'error',
              error: msg,
              userAgent: ua,
            },
            null,
            2,
          ),
        );
        summary.push({
          backend: `ort-${combo.backend}`,
          requestedBackend: combo.backend,
          model: combo.model,
          status: 'error',
          error: msg,
        });
      }
    }

    fs.writeFileSync(path.join(OUT_DIR, 'bench_summary.json'), JSON.stringify(summary, null, 2));
    cdp.close();
  } finally {
    if (chromeProc?.pid) {
      try {
        execFileSync('taskkill', ['/PID', String(chromeProc.pid), '/T', '/F'], { stdio: 'ignore' });
      } catch {
        try { chromeProc.kill(); } catch { /* ignore */ }
      }
    }
    if (viteProc?.pid) {
      try {
        execFileSync('taskkill', ['/PID', String(viteProc.pid), '/T', '/F'], { stdio: 'ignore' });
      } catch {
        try { viteProc.kill(); } catch { /* ignore */ }
      }
    }
    await sleep(300);
    try {
      fs.rmSync(userDataDir, { recursive: true, force: true });
    } catch {
      // ignore
    }
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
