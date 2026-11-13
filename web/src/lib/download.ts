import type { DetectResult, ModelMeta } from "../types";

function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function downloadDetectJson(result: DetectResult, baseName: string): void {
  const cleanName = baseName.replace(/\.[^.]+$/, "");
  const blob = new Blob([JSON.stringify(result, null, 2)], {
    type: "application/json;charset=utf-8",
  });
  triggerDownload(blob, `${cleanName}_detections.json`);
}

export async function downloadAnnotatedPng(
  imageUrl: string,
  result: DetectResult,
  meta: ModelMeta,
  baseName: string,
): Promise<void> {
  const img = new Image();
  img.crossOrigin = "anonymous";
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = () => reject(new Error("Görsel PNG dışa aktarma için yüklenemedi."));
    img.src = imageUrl;
  });

  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth || result.width;
  canvas.height = img.naturalHeight || result.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas bağlamı oluşturulamadı.");

  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  const scale = Math.max(canvas.width, canvas.height) / 1000;
  const lineWidth = Math.max(3, Math.round(scale * 3.2));
  const fontSize = Math.max(16, Math.round(scale * 16));
  ctx.font = `bold ${fontSize}px ui-monospace, monospace`;

  for (const det of result.detections) {
    const [x1, y1, x2, y2] = det.box;
    const color = meta.colors[det.cls_id] ?? "#38bdf8";

    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    const label = `#${det.id} ${det.cls} ${det.conf.toFixed(2)}`;
    const metrics = ctx.measureText(label);
    const padX = Math.round(6 * Math.max(1, scale * 0.8));
    const boxH = fontSize + 10;
    const labelY = Math.max(0, y1 - boxH);

    ctx.fillStyle = color;
    ctx.fillRect(x1, labelY, metrics.width + padX * 2, boxH);
    ctx.fillStyle = "#06131f";
    ctx.fillText(label, x1 + padX, labelY + fontSize + 2);
  }

  const blob = await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((b) => {
      if (b) resolve(b);
      else reject(new Error("PNG dönüştürme başarısız."));
    }, "image/png");
  });
  const cleanName = baseName.replace(/\.[^.]+$/, "");
  triggerDownload(blob, `${cleanName}_annotated.png`);
}

export function downloadBase64Png(base64Png: string, filename: string): void {
  const binary = atob(base64Png);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  const blob = new Blob([bytes], { type: "image/png" });
  triggerDownload(blob, filename);
}
