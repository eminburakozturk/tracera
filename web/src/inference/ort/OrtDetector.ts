import { FALLBACK_META } from "../../lib/meta";
import type { DetectResult, ModelMeta } from "../../types";
import type { Detector, OrtProgress } from "../types";

export type OrtEpPreference = "auto" | "webgpu" | "wasm";

type WorkerOutMsg =
  | { type: "progress"; id: number; progress: OrtProgress }
  | { type: "result"; id: number; result: DetectResult }
  | { type: "error"; id: number; message: string };

interface PendingRequest {
  resolve: (res: DetectResult) => void;
  reject: (err: Error) => void;
  onProgress?: (progress: OrtProgress) => void;
}

let sharedWorker: Worker | null = null;
let nextRequestId = 1;
const pending = new Map<number, PendingRequest>();

function getWorker(): Worker {
  if (!sharedWorker) {
    sharedWorker = new Worker(new URL("./ort.worker.ts", import.meta.url), {
      type: "module",
    });
    sharedWorker.onmessage = (ev: MessageEvent<WorkerOutMsg>) => {
      const msg = ev.data;
      const entry = pending.get(msg.id);
      if (!entry) return;
      if (msg.type === "progress") {
        entry.onProgress?.(msg.progress);
      } else if (msg.type === "result") {
        pending.delete(msg.id);
        entry.resolve(msg.result);
      } else if (msg.type === "error") {
        pending.delete(msg.id);
        entry.reject(new Error(msg.message));
      }
    };
    sharedWorker.onerror = (ev) => {
      const err = new Error(ev.message || "ONNX Runtime Web Worker hatası.");
      for (const [id, entry] of pending.entries()) {
        pending.delete(id);
        entry.reject(err);
      }
      sharedWorker?.terminate();
      sharedWorker = null;
    };
  }
  return sharedWorker;
}

export class OrtDetector implements Detector {
  private meta: ModelMeta;
  private onProgress?: (progress: OrtProgress) => void;
  private epPref: OrtEpPreference;

  constructor(
    meta: ModelMeta = FALLBACK_META,
    onProgress?: (progress: OrtProgress) => void,
    epPref: OrtEpPreference = "auto",
  ) {
    this.meta = meta;
    this.onProgress = onProgress;
    this.epPref = epPref;
  }

  async detect(file: File, conf: number): Promise<DetectResult> {
    const worker = getWorker();
    const buffer = await file.arrayBuffer();
    const id = nextRequestId++;

    return new Promise<DetectResult>((resolve, reject) => {
      pending.set(id, {
        resolve,
        reject,
        onProgress: this.onProgress,
      });
      worker.postMessage(
        {
          type: "detect",
          id,
          buffer,
          mimeType: file.type || "image/jpeg",
          conf,
          meta: this.meta,
          epPref: this.epPref,
        },
        [buffer],
      );
    });
  }
}
