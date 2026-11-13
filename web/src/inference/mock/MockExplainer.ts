import type { Detection, ExplainResult } from "../../types";
import type { Explainer } from "../types";

function canvasToRawBase64(canvas: HTMLCanvasElement): string {
  const dataUrl = canvas.toDataURL("image/png");
  const idx = dataUrl.indexOf("base64,");
  return idx >= 0 ? dataUrl.slice(idx + "base64,".length) : dataUrl;
}

function loadFileBitmapOrImg(file: File): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      resolve(img);
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("MockExplainer: görüntü okunamadı."));
    };
    img.src = url;
  });
}

export class MockExplainer implements Explainer {
  async explain(file: File, det: Detection): Promise<ExplainResult> {
    const t0 = performance.now();
    const img = await loadFileBitmapOrImg(file);
    const w = img.naturalWidth || 1024;
    const h = img.naturalHeight || 1024;

    const fullCanvas = document.createElement("canvas");
    fullCanvas.width = w;
    fullCanvas.height = h;
    const ctx = fullCanvas.getContext("2d");
    if (!ctx) throw new Error("Canvas context oluşturulamadı.");

    ctx.drawImage(img, 0, 0, w, h);

    const [x1, y1, x2, y2] = det.box;
    const cx = (x1 + x2) / 2;
    const cy = (y1 + y2) / 2;
    const radius = Math.max(48, Math.max(x2 - x1, y2 - y1) * 1.4);

    // Örnek ısı haritası gradyanı (Adım 1 mock görünümü)
    const grad = ctx.createRadialGradient(cx, cy, 4, cx, cy, radius);
    grad.addColorStop(0.0, "rgba(239, 68, 68, 0.75)");
    grad.addColorStop(0.35, "rgba(250, 204, 21, 0.52)");
    grad.addColorStop(0.7, "rgba(56, 189, 248, 0.25)");
    grad.addColorStop(1.0, "rgba(15, 23, 42, 0.0)");

    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fill();

    // Kutu ve hedef işaretçisi
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = Math.max(3, Math.round(Math.max(w, h) / 450));
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    const halfCrop = Math.max(180, Math.round(Math.max(x2 - x1, y2 - y1) * 2.5));
    const zx1 = Math.max(0, Math.round(cx - halfCrop));
    const zy1 = Math.max(0, Math.round(cy - halfCrop));
    const zx2 = Math.min(w, Math.round(cx + halfCrop));
    const zy2 = Math.min(h, Math.round(cy + halfCrop));
    const zw = Math.max(64, zx2 - zx1);
    const zh = Math.max(64, zy2 - zy1);

    const zoomCanvas = document.createElement("canvas");
    zoomCanvas.width = zw;
    zoomCanvas.height = zh;
    const zCtx = zoomCanvas.getContext("2d");
    if (!zCtx) throw new Error("Zoom canvas context oluşturulamadı.");
    zCtx.drawImage(fullCanvas, zx1, zy1, zw, zh, 0, 0, zw, zh);

    const totalMs = Number((performance.now() - t0).toFixed(2));
    return {
      overlay_png: canvasToRawBase64(fullCanvas),
      zoom_png: canvasToRawBase64(zoomCanvas),
      method: "layercam",
      layer: 4,
      timing: {
        pre_ms: 8.0,
        model_ms: Math.max(25.0, totalMs * 0.7),
        post_ms: 4.0,
        total_ms: Math.max(37.0, totalMs),
      },
    };
  }
}
