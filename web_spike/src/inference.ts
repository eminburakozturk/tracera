import * as ort from 'onnxruntime-web';

export interface MetaConfig {
  version: string;
  model: string;
  imgsz: number;
  conf: number;
  iou_nms: number;
  letterbox_pad: number;
  classes: string[];
  colors: string[];
  onnx: string;
}

export interface Detection {
  id: number;
  cls: string;
  cls_id: number;
  conf: number;
  box: [number, number, number, number];
}

export interface Timing {
  pre_ms: number;
  model_ms: number;
  post_ms: number;
  total_ms: number;
}

export interface DetectResult {
  width: number;
  height: number;
  detections: Detection[];
  timing: Timing;
  backend: 'ort-webgpu' | 'ort-wasm' | string;
}

export type BackendChoice = 'auto' | 'webgpu' | 'wasm';

export interface LetterboxMeta {
  ratio: number;
  left: number;
  top: number;
  unpadW: number;
  unpadH: number;
  origW: number;
  origH: number;
}

ort.env.wasm.wasmPaths = '/ort/';

export async function loadMeta(): Promise<MetaConfig> {
  const resp = await fetch('/models/meta.json');
  if (!resp.ok) {
    throw new Error(`meta.json yüklenemedi: HTTP ${resp.status}`);
  }
  return (await resp.json()) as MetaConfig;
}

export async function hasWebGPU(): Promise<boolean> {
  const nav = navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown | null> } };
  if (!nav.gpu) return false;
  try {
    const adapter = await nav.gpu.requestAdapter();
    return adapter !== null;
  } catch {
    return false;
  }
}

export async function createOrtSession(
  modelUrl: string,
  backendPref: BackendChoice,
): Promise<{ session: ort.InferenceSession; activeBackend: 'ort-webgpu' | 'ort-wasm' }> {
  const gpuAvailable = await hasWebGPU();
  const tryOrder: Array<'webgpu' | 'wasm'> =
    backendPref === 'webgpu'
      ? ['webgpu']
      : backendPref === 'wasm'
        ? ['wasm']
        : gpuAvailable
          ? ['webgpu', 'wasm']
          : ['wasm'];

  let lastErr: unknown = null;
  for (const ep of tryOrder) {
    try {
      if (ep === 'wasm') {
        const threads = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated
          ? Math.min(4, navigator.hardwareConcurrency || 4)
          : 1;
        ort.env.wasm.numThreads = threads;
      }
      const session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: [ep],
        graphOptimizationLevel: 'all',
      });
      return {
        session,
        activeBackend: ep === 'webgpu' ? 'ort-webgpu' : 'ort-wasm',
      };
    } catch (err) {
      lastErr = err;
      if (backendPref !== 'auto') {
        throw err;
      }
    }
  }
  throw lastErr instanceof Error ? lastErr : new Error(String(lastErr));
}

export function preprocessLetterbox(
  img: HTMLImageElement | ImageBitmap,
  imgsz: number,
  padVal: number,
  reusableCanvas?: HTMLCanvasElement,
): { tensor: ort.Tensor; lb: LetterboxMeta } {
  const origW = img instanceof HTMLImageElement ? img.naturalWidth : img.width;
  const origH = img instanceof HTMLImageElement ? img.naturalHeight : img.height;

  const ratio = Math.min(imgsz / origH, imgsz / origW);
  const unpadW = Math.round(origW * ratio);
  const unpadH = Math.round(origH * ratio);
  const dw = (imgsz - unpadW) / 2.0;
  const dh = (imgsz - unpadH) / 2.0;
  const top = Math.round(dh - 0.1);
  const left = Math.round(dw - 0.1);

  // Orijinal pikselleri tam boyutta oku (Canvas downscale/area filtresi farkını önlemek için)
  const srcCanvas = reusableCanvas ?? document.createElement('canvas');
  srcCanvas.width = origW;
  srcCanvas.height = origH;
  const srcCtx = srcCanvas.getContext('2d', { willReadFrequently: true });
  if (!srcCtx) {
    throw new Error('2D canvas bağlamı oluşturulamadı.');
  }
  srcCtx.drawImage(img, 0, 0, origW, origH);
  const srcRgba = srcCtx.getImageData(0, 0, origW, origH).data;

  const planeSize = imgsz * imgsz;
  const chw = new Float32Array(3 * planeSize);
  const inv255 = 1.0 / 255.0;
  chw.fill(padVal * inv255);

  // OpenCV cv2.resize(..., INTER_LINEAR) ile birebir çift doğrusal örnekleme
  const scaleX = origW / unpadW;
  const scaleY = origH / unpadH;

  const x0Arr = new Int32Array(unpadW);
  const x1Arr = new Int32Array(unpadW);
  const fxArr = new Float32Array(unpadW);
  for (let dx = 0; dx < unpadW; dx++) {
    let sx = (dx + 0.5) * scaleX - 0.5;
    let ix0 = Math.floor(sx);
    let fx = sx - ix0;
    if (ix0 < 0) {
      ix0 = 0;
      fx = 0;
    } else if (ix0 >= origW - 1) {
      ix0 = origW - 1;
      fx = 0;
    }
    x0Arr[dx] = ix0 * 4;
    x1Arr[dx] = Math.min(ix0 + 1, origW - 1) * 4;
    fxArr[dx] = fx;
  }

  for (let dy = 0; dy < unpadH; dy++) {
    const sy = (dy + 0.5) * scaleY - 0.5;
    let iy0 = Math.floor(sy);
    let fy = sy - iy0;
    if (iy0 < 0) {
      iy0 = 0;
      fy = 0;
    } else if (iy0 >= origH - 1) {
      iy0 = origH - 1;
      fy = 0;
    }
    const iy1 = Math.min(iy0 + 1, origH - 1);
    const row0Base = iy0 * origW * 4;
    const row1Base = iy1 * origW * 4;
    const fy0 = 1.0 - fy;
    const dstRowOffset = (top + dy) * imgsz + left;

    for (let dx = 0; dx < unpadW; dx++) {
      const fx = fxArr[dx];
      const fx0 = 1.0 - fx;
      const p00 = row0Base + x0Arr[dx];
      const p01 = row0Base + x1Arr[dx];
      const p10 = row1Base + x0Arr[dx];
      const p11 = row1Base + x1Arr[dx];

      const r = Math.round(
        (srcRgba[p00] * fx0 + srcRgba[p01] * fx) * fy0 +
          (srcRgba[p10] * fx0 + srcRgba[p11] * fx) * fy,
      );
      const g = Math.round(
        (srcRgba[p00 + 1] * fx0 + srcRgba[p01 + 1] * fx) * fy0 +
          (srcRgba[p10 + 1] * fx0 + srcRgba[p11 + 1] * fx) * fy,
      );
      const b = Math.round(
        (srcRgba[p00 + 2] * fx0 + srcRgba[p01 + 2] * fx) * fy0 +
          (srcRgba[p10 + 2] * fx0 + srcRgba[p11 + 2] * fx) * fy,
      );

      const dstIdx = dstRowOffset + dx;
      chw[dstIdx] = r * inv255;
      chw[planeSize + dstIdx] = g * inv255;
      chw[2 * planeSize + dstIdx] = b * inv255;
    }
  }

  return {
    tensor: new ort.Tensor('float32', chw, [1, 3, imgsz, imgsz]),
    lb: { ratio, left, top, unpadW, unpadH, origW, origH },
  };
}

interface RawCandidate {
  clsId: number;
  conf: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

function computeIoU(a: RawCandidate, b: RawCandidate): number {
  const ix1 = Math.max(a.x1, b.x1);
  const iy1 = Math.max(a.y1, b.y1);
  const ix2 = Math.min(a.x2, b.x2);
  const iy2 = Math.min(a.y2, b.y2);
  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  if (inter <= 0) return 0;
  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
  const union = areaA + areaB - inter;
  return union > 0 ? inter / union : 0;
}

export function postprocessYolo(
  outputData: Float32Array,
  dims: readonly number[],
  lb: LetterboxMeta,
  meta: MetaConfig,
  confThreshold: number,
): Detection[] {
  const numChannels = dims[1];
  const numAnchors = dims[2];
  const numClasses = meta.classes.length;
  if (numChannels !== 4 + numClasses) {
    throw new Error(`Beklenen kanal sayısı ${4 + numClasses}, gelen: ${numChannels}`);
  }

  // Ön adayları topla ve yüksekten düşüğe sırala
  const candidates: RawCandidate[] = [];
  for (let a = 0; a < numAnchors; a++) {
    let bestCls = 0;
    let bestConf = -Infinity;
    for (let c = 0; c < numClasses; c++) {
      const score = outputData[(4 + c) * numAnchors + a];
      if (score > bestConf) {
        bestConf = score;
        bestCls = c;
      }
    }
    if (bestConf < confThreshold) continue;

    const cx = outputData[a];
    const cy = outputData[numAnchors + a];
    const w = outputData[2 * numAnchors + a];
    const h = outputData[3 * numAnchors + a];

    candidates.push({
      clsId: bestCls,
      conf: bestConf,
      x1: cx - w / 2,
      y1: cy - h / 2,
      x2: cx + w / 2,
      y2: cy + h / 2,
    });
  }

  candidates.sort((a, b) => b.conf - a.conf);

  // Sınıf bazlı NMS (IoU 0.7) -> orijinal koordinata geri ölçekle
  const byClass = new Map<number, RawCandidate[]>();
  for (const cand of candidates) {
    const arr = byClass.get(cand.clsId);
    if (arr) {
      arr.push(cand);
    } else {
      byClass.set(cand.clsId, [cand]);
    }
  }

  const kept: RawCandidate[] = [];
  for (const clsCandidates of byClass.values()) {
    const selected: RawCandidate[] = [];
    for (const cand of clsCandidates) {
      let suppressed = false;
      for (const prev of selected) {
        if (computeIoU(cand, prev) > meta.iou_nms) {
          suppressed = true;
          break;
        }
      }
      if (!suppressed) {
        selected.push(cand);
        kept.push(cand);
      }
    }
  }

  // Güven skoruna göre azalan sıraya sok
  kept.sort((a, b) => b.conf - a.conf);

  return kept
    .filter((k) => k.conf >= confThreshold)
    .map((k, idx) => {
      const ox1 = Math.max(0, Math.min((k.x1 - lb.left) / lb.ratio, lb.origW));
      const oy1 = Math.max(0, Math.min((k.y1 - lb.top) / lb.ratio, lb.origH));
      const ox2 = Math.max(0, Math.min((k.x2 - lb.left) / lb.ratio, lb.origW));
      const oy2 = Math.max(0, Math.min((k.y2 - lb.top) / lb.ratio, lb.origH));
      return {
        id: idx,
        cls: meta.classes[k.clsId] ?? `class_${k.clsId}`,
        cls_id: k.clsId,
        conf: Number(k.conf.toFixed(6)),
        box: [
          Number(ox1.toFixed(3)),
          Number(oy1.toFixed(3)),
          Number(ox2.toFixed(3)),
          Number(oy2.toFixed(3)),
        ],
      };
    });
}

export async function runDetect(
  session: ort.InferenceSession,
  activeBackend: 'ort-webgpu' | 'ort-wasm',
  img: HTMLImageElement | ImageBitmap,
  meta: MetaConfig,
  confThreshold: number,
  reusableCanvas?: HTMLCanvasElement,
): Promise<DetectResult> {
  const tStart = performance.now();
  const { tensor, lb } = preprocessLetterbox(img, meta.imgsz, meta.letterbox_pad, reusableCanvas);
  const tPre = performance.now();

  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  const feeds: Record<string, ort.Tensor> = { [inputName]: tensor };
  const outputs = await session.run(feeds);
  const outTensor = outputs[outputName];
  const outCopy = new Float32Array(outTensor.data as Float32Array);
  const dims = outTensor.dims;
  tensor.dispose?.();
  outTensor.dispose?.();

  const tModel = performance.now();
  const detections = postprocessYolo(outCopy, dims, lb, meta, confThreshold);
  const tEnd = performance.now();

  return {
    width: lb.origW,
    height: lb.origH,
    detections,
    timing: {
      pre_ms: Number((tPre - tStart).toFixed(3)),
      model_ms: Number((tModel - tPre).toFixed(3)),
      post_ms: Number((tEnd - tModel).toFixed(3)),
      total_ms: Number((tEnd - tStart).toFixed(3)),
    },
    backend: activeBackend,
  };
}
