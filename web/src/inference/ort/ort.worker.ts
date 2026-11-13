import * as ort from "onnxruntime-web";
import type { DetectResult, ModelMeta } from "../../types";
import type { OrtProgress } from "../types";
import { computeLetterbox, postprocessYolo } from "./postprocess";

ort.env.wasm.wasmPaths = "/ort/";

interface DetectRequestMsg {
  type: "detect";
  id: number;
  buffer: ArrayBuffer;
  mimeType: string;
  conf: number;
  meta: ModelMeta;
}

type WorkerOutMsg =
  | { type: "progress"; id: number; progress: OrtProgress }
  | { type: "result"; id: number; result: DetectResult }
  | { type: "error"; id: number; message: string };

let cachedModelUrl: string | null = null;
let cachedSession: ort.InferenceSession | null = null;
let cachedBackend: "ort-webgpu" | "ort-wasm" = "ort-wasm";
let offscreenCanvas: OffscreenCanvas | null = null;

function postOut(msg: WorkerOutMsg) {
  (self as unknown as { postMessage: (m: WorkerOutMsg) => void }).postMessage(msg);
}

async function hasWebGPU(): Promise<boolean> {
  const nav = self.navigator as Navigator & {
    gpu?: { requestAdapter(): Promise<unknown | null> };
  };
  if (!nav?.gpu) return false;
  try {
    const adapter = await nav.gpu.requestAdapter();
    return adapter !== null;
  } catch {
    return false;
  }
}

async function fetchModelBytesWithProgress(
  modelUrl: string,
  reqId: number,
): Promise<Uint8Array> {
  postOut({
    type: "progress",
    id: reqId,
    progress: {
      stage: "downloading",
      percent: 2,
      message: "FP32 ONNX modeli indiriliyor...",
    },
  });

  const resp = await fetch(modelUrl);
  if (!resp.ok) {
    throw new Error(`Model indirilemedi (${modelUrl}): HTTP ${resp.status}`);
  }

  const contentLengthHeader = resp.headers.get("Content-Length");
  const totalBytes = contentLengthHeader ? Number(contentLengthHeader) : 0;

  if (!resp.body) {
    const buf = new Uint8Array(await resp.arrayBuffer());
    postOut({
      type: "progress",
      id: reqId,
      progress: {
        stage: "downloading",
        percent: 80,
        message: "Model indirildi, oturum hazırlanıyor...",
      },
    });
    return buf;
  }

  const reader = resp.body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) {
      chunks.push(value);
      received += value.byteLength;
      if (totalBytes > 0) {
        const pct = Math.min(80, Math.max(2, (received / totalBytes) * 80));
        postOut({
          type: "progress",
          id: reqId,
          progress: {
            stage: "downloading",
            percent: pct,
            message: `Model indiriliyor (${(received / 1048576).toFixed(1)} / ${(totalBytes / 1048576).toFixed(1)} MB)...`,
          },
        });
      }
    }
  }

  const combined = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    combined.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return combined;
}

async function ensureSession(
  meta: ModelMeta,
  reqId: number,
): Promise<{ session: ort.InferenceSession; backend: "ort-webgpu" | "ort-wasm" }> {
  const modelUrl = `/models/${meta.onnx}`;
  if (cachedSession && cachedModelUrl === modelUrl) {
    return { session: cachedSession, backend: cachedBackend };
  }

  const modelBytes = await fetchModelBytesWithProgress(modelUrl, reqId);
  const gpuAvailable = await hasWebGPU();
  const tryOrder: Array<"webgpu" | "wasm"> = gpuAvailable
    ? ["webgpu", "wasm"]
    : ["wasm"];

  let lastErr: unknown = null;
  for (const ep of tryOrder) {
    try {
      postOut({
        type: "progress",
        id: reqId,
        progress: {
          stage: "initializing",
          percent: 88,
          message:
            ep === "webgpu"
              ? "WebGPU motoru derleniyor..."
              : "WASM motoru hazırlanıyor...",
        },
      });

      // OrtDetector zaten Web Worker içinde çalıştığından iç içe pthread worker kilitlenmesini önlemek için 1 tutulur.
      ort.env.wasm.numThreads = 1;

      const session = await ort.InferenceSession.create(modelBytes, {
        executionProviders: [ep],
        graphOptimizationLevel: "all",
      });

      cachedSession = session;
      cachedModelUrl = modelUrl;
      cachedBackend = ep === "webgpu" ? "ort-webgpu" : "ort-wasm";

      postOut({
        type: "progress",
        id: reqId,
        progress: { stage: "ready", percent: 100, message: "Hazır" },
      });

      return { session, backend: cachedBackend };
    } catch (err) {
      lastErr = err;
    }
  }

  throw new Error(
    `ONNX Runtime oturumu başlatılamadı: ${
      lastErr instanceof Error ? lastErr.message : String(lastErr)
    }`,
  );
}

function preprocessBilinearLetterbox(
  bitmap: ImageBitmap,
  imgsz: number,
  padVal: number,
): {
  tensor: ort.Tensor;
  lb: ReturnType<typeof computeLetterbox>;
  origW: number;
  origH: number;
} {
  const origW = bitmap.width;
  const origH = bitmap.height;
  const lb = computeLetterbox(origW, origH, imgsz);

  if (!offscreenCanvas) {
    offscreenCanvas = new OffscreenCanvas(origW, origH);
  } else {
    offscreenCanvas.width = origW;
    offscreenCanvas.height = origH;
  }

  const ctx = offscreenCanvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) {
    throw new Error("Worker içinde 2D OffscreenCanvas bağlamı oluşturulamadı.");
  }

  ctx.drawImage(bitmap, 0, 0, origW, origH);
  const srcRgba = ctx.getImageData(0, 0, origW, origH).data;

  const planeSize = imgsz * imgsz;
  const chw = new Float32Array(3 * planeSize);
  const inv255 = 1.0 / 255.0;
  chw.fill(padVal * inv255);

  const scaleX = origW / lb.unpadW;
  const scaleY = origH / lb.unpadH;

  const x0Arr = new Int32Array(lb.unpadW);
  const x1Arr = new Int32Array(lb.unpadW);
  const fxArr = new Float32Array(lb.unpadW);
  for (let dx = 0; dx < lb.unpadW; dx++) {
    const sx = (dx + 0.5) * scaleX - 0.5;
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

  for (let dy = 0; dy < lb.unpadH; dy++) {
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
    const dstRowOffset = (lb.top + dy) * imgsz + lb.left;

    for (let dx = 0; dx < lb.unpadW; dx++) {
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
    tensor: new ort.Tensor("float32", chw, [1, 3, imgsz, imgsz]),
    lb,
    origW,
    origH,
  };
}

let queue: Promise<void> = Promise.resolve();

self.onmessage = (ev: MessageEvent<DetectRequestMsg>) => {
  const msg = ev.data;
  if (msg.type !== "detect") return;

  queue = queue
    .then(async () => {
      const tTotal0 = performance.now();
      const { session, backend } = await ensureSession(msg.meta, msg.id);

      const tPre0 = performance.now();
      const blob = new Blob([msg.buffer], {
        type: msg.mimeType || "image/jpeg",
      });
      const bitmap = await createImageBitmap(blob, {
        colorSpaceConversion: "none",
        premultiplyAlpha: "none",
      });

      let tensor: ort.Tensor | null = null;
      let outTensor: ort.Tensor | null = null;
      try {
        const pre = preprocessBilinearLetterbox(
          bitmap,
          msg.meta.imgsz,
          msg.meta.letterbox_pad,
        );
        tensor = pre.tensor;
        const tPre1 = performance.now();

        const tModel0 = performance.now();
        const inputName = session.inputNames[0] ?? "images";
        const outputs = await session.run({ [inputName]: tensor });
        const outputName = session.outputNames[0] ?? Object.keys(outputs)[0];
        outTensor = outputs[outputName];
        const rawData = (await outTensor.getData()) as Float32Array;
        const tModel1 = performance.now();

        const tPost0 = performance.now();
        const detections = postprocessYolo(
          rawData,
          outTensor.dims,
          pre.lb,
          msg.meta,
          msg.conf,
        );
        const tPost1 = performance.now();

        const tTotal1 = performance.now();
        const result: DetectResult = {
          width: pre.origW,
          height: pre.origH,
          detections,
          timing: {
            pre_ms: Number((tPre1 - tPre0).toFixed(2)),
            model_ms: Number((tModel1 - tModel0).toFixed(2)),
            post_ms: Number((tPost1 - tPost0).toFixed(2)),
            total_ms: Number((tTotal1 - tTotal0).toFixed(2)),
          },
          backend,
        };

        postOut({ type: "result", id: msg.id, result });
      } finally {
        bitmap.close();
        tensor?.dispose();
        outTensor?.dispose();
      }
    })
    .catch((err) => {
      postOut({
        type: "error",
        id: msg.id,
        message: err instanceof Error ? err.message : String(err),
      });
    });
};
