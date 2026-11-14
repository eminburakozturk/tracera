import { openDB, type IDBPDatabase } from "idb";
import type { DetectResult, HistoryRecord } from "../types";

const DB_NAME = "tracera_v11_db";
const DB_VERSION = 1;
const STORE_NAME = "inspections";

let memoryFallback: HistoryRecord[] = [];

async function getDb(): Promise<IDBPDatabase | null> {
  if (typeof indexedDB === "undefined") {
    return null;
  }
  try {
    return await openDB(DB_NAME, DB_VERSION, {
      upgrade(db) {
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          const store = db.createObjectStore(STORE_NAME, { keyPath: "id" });
          store.createIndex("createdAt", "createdAt");
        }
      },
    });
  } catch {
    return null;
  }
}

export async function createThumbnailDataUrl(
  fileOrUrl: File | string,
  maxSide = 280,
): Promise<string> {
  if (typeof document === "undefined") {
    return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";
  }
  const srcUrl =
    typeof fileOrUrl === "string"
      ? fileOrUrl
      : URL.createObjectURL(fileOrUrl);
  const shouldRevoke = typeof fileOrUrl !== "string";

  try {
    const img = await new Promise<HTMLImageElement>((resolve, reject) => {
      const el = new Image();
      el.crossOrigin = "anonymous";
      el.onload = () => resolve(el);
      el.onerror = () => reject(new Error("Image load failed"));
      el.src = srcUrl;
    });
    const w = img.naturalWidth || img.width || 280;
    const h = img.naturalHeight || img.height || 220;
    const scale = Math.min(1, maxSide / Math.max(w, h));
    const cw = Math.max(1, Math.round(w * scale));
    const ch = Math.max(1, Math.round(h * scale));
    const canvas = document.createElement("canvas");
    canvas.width = cw;
    canvas.height = ch;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";
    }
    ctx.drawImage(img, 0, 0, cw, ch);
    return canvas.toDataURL("image/jpeg", 0.78);
  } catch {
    return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";
  } finally {
    if (shouldRevoke) {
      URL.revokeObjectURL(srcUrl);
    }
  }
}

export async function saveInspectionToHistory(params: {
  filename: string;
  fileOrUrl: File | string;
  confThreshold: number;
  result: DetectResult;
  idOverride?: string;
}): Promise<HistoryRecord> {
  const thumbnailDataUrl = await createThumbnailDataUrl(params.fileOrUrl);
  const activeDetections = params.result.detections.filter(
    (d) => d.conf >= params.confThreshold,
  );
  const record: HistoryRecord = {
    id:
      params.idOverride ??
      `insp_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    createdAt: new Date().toISOString(),
    filename: params.filename,
    thumbnailDataUrl,
    confThreshold: params.confThreshold,
    verdict: activeDetections.length > 0 ? "FAIL" : "PASS",
    defectCount: activeDetections.length,
    result: params.result,
  };

  const db = await getDb();
  if (db) {
    await db.put(STORE_NAME, record);
  } else {
    memoryFallback = [
      record,
      ...memoryFallback.filter((r) => r.id !== record.id),
    ];
  }
  return record;
}

export async function listHistoryRecords(): Promise<HistoryRecord[]> {
  const db = await getDb();
  if (db) {
    const all = (await db.getAll(STORE_NAME)) as HistoryRecord[];
    return all.sort((a, b) => b.createdAt.localeCompare(a.createdAt));
  }
  return [...memoryFallback].sort((a, b) =>
    b.createdAt.localeCompare(a.createdAt),
  );
}

export async function deleteHistoryRecord(id: string): Promise<void> {
  const db = await getDb();
  if (db) {
    await db.delete(STORE_NAME, id);
  } else {
    memoryFallback = memoryFallback.filter((r) => r.id !== id);
  }
}

export async function clearAllHistoryRecords(): Promise<void> {
  const db = await getDb();
  if (db) {
    await db.clear(STORE_NAME);
  }
  memoryFallback = [];
}

export async function estimateHistoryStorageBytes(): Promise<number> {
  const records = await listHistoryRecords();
  return new Blob([JSON.stringify(records)]).size;
}
