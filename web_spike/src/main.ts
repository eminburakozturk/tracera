import type { InferenceSession } from 'onnxruntime-web';
import {
  type BackendChoice,
  type DetectResult,
  type MetaConfig,
  createOrtSession,
  hasWebGPU,
  loadMeta,
  runDetect,
} from './inference';

const backendSelect = document.getElementById('backendSelect') as HTMLSelectElement;
const modelSelect = document.getElementById('modelSelect') as HTMLSelectElement;
const sampleSelect = document.getElementById('sampleSelect') as HTMLSelectElement;
const fileInput = document.getElementById('fileInput') as HTMLInputElement;
const confInput = document.getElementById('confInput') as HTMLInputElement;
const runBtn = document.getElementById('runBtn') as HTMLButtonElement;
const statusEl = document.getElementById('status') as HTMLDivElement;
const activeBackendEl = document.getElementById('activeBackend') as HTMLSpanElement;
const canvasEl = document.getElementById('previewCanvas') as HTMLCanvasElement;
const tableBody = document.getElementById('detTableBody') as HTMLTableSectionElement;
const jsonOut = document.getElementById('jsonOut') as HTMLPreElement;

let meta: MetaConfig | null = null;
let session: InferenceSession | null = null;
let activeBackend: 'ort-webgpu' | 'ort-wasm' = 'ort-wasm';
let loadedKey = '';
const workCanvas = document.createElement('canvas');

function getModelUrl(modelKind: string, m: MetaConfig): string {
  return modelKind === 'int8' ? '/models/tracera_1024_int8.onnx' : `/models/${m.onnx}`;
}

async function ensureSession(): Promise<{ session: InferenceSession; backend: 'ort-webgpu' | 'ort-wasm'; meta: MetaConfig }> {
  if (!meta) {
    meta = await loadMeta();
  }
  const backendPref = backendSelect.value as BackendChoice;
  const modelKind = modelSelect.value;
  const key = `${backendPref}::${modelKind}`;
  if (session && loadedKey === key) {
    return { session, backend: activeBackend, meta };
  }
  statusEl.textContent = `Model yükleniyor (${modelKind}, backend=${backendPref})...`;
  const modelUrl = getModelUrl(modelKind, meta);
  const created = await createOrtSession(modelUrl, backendPref);
  session = created.session;
  activeBackend = created.activeBackend;
  loadedKey = key;
  activeBackendEl.textContent = activeBackend;
  return { session, backend: activeBackend, meta };
}

async function loadImageFromUrl(url: string): Promise<ImageBitmap> {
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`Görüntü yüklenemedi: ${url} (HTTP ${resp.status})`);
  }
  const blob = await resp.blob();
  return await createImageBitmap(blob, {
    colorSpaceConversion: 'none',
    premultiplyAlpha: 'none',
  });
}

function drawDetections(img: HTMLImageElement | ImageBitmap, res: DetectResult, m: MetaConfig): void {
  const w = img instanceof HTMLImageElement ? img.naturalWidth : img.width;
  const h = img instanceof HTMLImageElement ? img.naturalHeight : img.height;
  canvasEl.width = w;
  canvasEl.height = h;
  const ctx = canvasEl.getContext('2d');
  if (!ctx) return;

  ctx.drawImage(img, 0, 0);
  const lineWidth = Math.max(3, Math.round(Math.max(w, h) / 450));
  const fontSize = Math.max(16, Math.round(Math.max(w, h) / 70));
  ctx.lineWidth = lineWidth;
  ctx.font = `bold ${fontSize}px ui-monospace, monospace`;

  for (const det of res.detections) {
    const [x1, y1, x2, y2] = det.box;
    const color = m.colors[det.cls_id] ?? '#38bdf8';
    ctx.strokeStyle = color;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    const label = `#${det.id} ${det.cls} ${det.conf.toFixed(2)}`;
    const tm = ctx.measureText(label);
    const padX = 6;
    const boxH = fontSize + 8;
    const labelY = Math.max(0, y1 - boxH);
    ctx.fillStyle = color;
    ctx.fillRect(x1, labelY, tm.width + padX * 2, boxH);
    ctx.fillStyle = '#09090b';
    ctx.fillText(label, x1 + padX, labelY + fontSize + 1);
  }
}

function renderTable(res: DetectResult): void {
  tableBody.innerHTML = '';
  for (const d of res.detections) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${d.id}</td><td>${d.cls}</td><td>${d.conf.toFixed(4)}</td><td>[${d.box.map((v) => v.toFixed(1)).join(', ')}]</td>`;
    tableBody.appendChild(tr);
  }
}

async function getCurrentImage(): Promise<ImageBitmap> {
  const file = fileInput.files?.[0];
  if (file) {
    return await createImageBitmap(file, {
      colorSpaceConversion: 'none',
      premultiplyAlpha: 'none',
    });
  }
  const sample = sampleSelect.value;
  if (!sample) {
    throw new Error('Örnek veya dosya seçilmedi.');
  }
  return await loadImageFromUrl(`/val-images/${encodeURIComponent(sample)}`);
}

async function executeSingle(): Promise<DetectResult> {
  const { session: sess, backend, meta: m } = await ensureSession();
  const img = await getCurrentImage();
  const conf = Number(confInput.value) || m.conf;
  const res = await runDetect(sess, backend, img, m, conf, workCanvas);
  drawDetections(img, res, m);
  renderTable(res);
  jsonOut.textContent = JSON.stringify(res, null, 2);
  statusEl.textContent = `Backend: ${res.backend} | Tespit: ${res.detections.length} | Model: ${res.timing.model_ms} ms | Toplam: ${res.timing.total_ms} ms`;
  return res;
}

runBtn.addEventListener('click', () => {
  executeSingle().catch((err) => {
    statusEl.textContent = `Hata: ${err instanceof Error ? err.message : String(err)}`;
    console.error(err);
  });
});

// Otomatik doğrulama ve benchmark için tarayıcı API'si
declare global {
  interface Window {
    __TRACERA_SPIKE__: {
      hasWebGPU: typeof hasWebGPU;
      runBatch: (options: {
        backend: BackendChoice;
        model: 'fp32' | 'int8';
        samples: string[];
        warmupRuns?: number;
        benchRuns?: number;
      }) => Promise<{
        activeBackend: string;
        model: string;
        results: Record<string, DetectResult>;
        warmupModelMs: number[];
        benchModelMs: number[];
        benchTotalMs: number[];
        medianModelMs: number;
        medianTotalMs: number;
      }>;
    };
  }
}

function median(vals: number[]): number {
  const sorted = [...vals].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

window.__TRACERA_SPIKE__ = {
  hasWebGPU,
  async runBatch({ backend, model, samples, warmupRuns = 2, benchRuns = 10 }) {
    backendSelect.value = backend;
    modelSelect.value = model;
    const { session: sess, backend: activeEp, meta: m } = await ensureSession();
    const loadedImgs: Array<{ name: string; img: ImageBitmap }> = [];
    for (const s of samples) {
      loadedImgs.push({ name: s, img: await loadImageFromUrl(`/val-images/${encodeURIComponent(s)}`) });
    }

    const warmupModelMs: number[] = [];
    for (let w = 0; w < warmupRuns; w++) {
      const r = await runDetect(sess, activeEp, loadedImgs[0].img, m, m.conf, workCanvas);
      warmupModelMs.push(r.timing.model_ms);
    }

    const results: Record<string, DetectResult> = {};
    const benchModelMs: number[] = [];
    const benchTotalMs: number[] = [];

    for (const item of loadedImgs) {
      const r = await runDetect(sess, activeEp, item.img, m, m.conf, workCanvas);
      results[item.name] = r;
    }

    for (let i = 0; i < benchRuns; i++) {
      const item = loadedImgs[i % loadedImgs.length];
      const r = await runDetect(sess, activeEp, item.img, m, m.conf, workCanvas);
      benchModelMs.push(r.timing.model_ms);
      benchTotalMs.push(r.timing.total_ms);
    }

    return {
      activeBackend: activeEp,
      model,
      results,
      warmupModelMs,
      benchModelMs,
      benchTotalMs,
      medianModelMs: Number(median(benchModelMs).toFixed(2)),
      medianTotalMs: Number(median(benchTotalMs).toFixed(2)),
    };
  },
};

async function init(): Promise<void> {
  meta = await loadMeta();
  confInput.value = String(meta.conf);
  const gpuOk = await hasWebGPU();
  activeBackendEl.textContent = gpuOk ? 'WebGPU hazır (auto)' : 'WASM fallback hazır (auto)';

  const samplesResp = await fetch('/samples.json');
  const samples = (await samplesResp.json()) as string[];
  sampleSelect.innerHTML = samples.map((s) => `<option value="${s}">${s}</option>`).join('');
  statusEl.textContent = 'Hazır. Bir örnek seçin veya görüntü yükleyip Çalıştır basın.';
}

init().catch((err) => {
  statusEl.textContent = `Başlatma hatası: ${err instanceof Error ? err.message : String(err)}`;
  console.error(err);
});
