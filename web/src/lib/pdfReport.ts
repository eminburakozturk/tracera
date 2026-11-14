import fontkit from "@pdf-lib/fontkit";
import { PDFDocument, PDFName, PDFString, rgb } from "pdf-lib";
import type { LangCode } from "../store/useAppStore";
import type { DetectResult, ExplainResult, ModelMeta } from "../types";
import {
  getClassDisplayName,
  PDF_MANDATORY_FOOTER_TR,
} from "./v11Rules";

let cachedFontBytes: Uint8Array | null = null;

export async function loadTurkishFontBytes(): Promise<Uint8Array> {
  if (cachedFontBytes) {
    return cachedFontBytes;
  }
  try {
    if (typeof fetch === "function") {
      const res = await fetch("/fonts/TraceraSans.ttf");
      if (res.ok) {
        const buf = await res.arrayBuffer();
        if (buf.byteLength > 1000) {
          cachedFontBytes = new Uint8Array(buf);
          return cachedFontBytes;
        }
      }
    }
  } catch {
    // Node/Vitest ortamında yerel dosya yedeğine geç
  }

  // Node / Vitest ortamında doğrudan diskten oku
  const fsMod = await import(/* @vite-ignore */ "node:fs");
  const pathMod = await import(/* @vite-ignore */ "node:path");
  const candidates = [
    pathMod.resolve(process.cwd(), "public/fonts/TraceraSans.ttf"),
    pathMod.resolve(process.cwd(), "web/public/fonts/TraceraSans.ttf"),
  ];
  for (const p of candidates) {
    if (fsMod.existsSync(p)) {
      const buf = fsMod.readFileSync(p);
      cachedFontBytes = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
      return cachedFontBytes;
    }
  }
  throw new Error("TraceraSans.ttf font dosyası bulunamadı.");
}

function base64ToUint8Array(b64: string): Uint8Array {
  const clean = b64.includes(",") ? b64.split(",")[1] : b64;
  if (typeof atob === "function") {
    const bin = atob(clean);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) {
      out[i] = bin.charCodeAt(i);
    }
    return out;
  }
  return new Uint8Array(Buffer.from(clean, "base64"));
}

export interface PdfInspectionReportInput {
  filename: string;
  createdAt?: string;
  meta: ModelMeta;
  confThreshold: number;
  result: DetectResult;
  annotatedPngBase64?: string | null;
  explainResult?: ExplainResult | null;
  lang?: LangCode;
}

export async function generateInspectionPdfBytes(
  input: PdfInspectionReportInput,
): Promise<Uint8Array> {
  const lang = input.lang ?? "tr";
  const createdAt = input.createdAt ?? new Date().toISOString();
  const activeDetections = input.result.detections.filter(
    (d) => d.conf >= input.confThreshold,
  );
  const verdict = activeDetections.length > 0 ? "KALDI" : "GEÇTİ";

  const pdfDoc = await PDFDocument.create();
  pdfDoc.registerFontkit(fontkit);

  const fontBytes = await loadTurkishFontBytes();
  // subset: false ile tüm Türkçe glifler (ğ ş ı İ ç ö ü) PDF içinde tam korunur
  const font = await pdfDoc.embedFont(fontBytes, { subset: false });

  const classSummary = activeDetections
    .map(
      (d) =>
        `#${d.id} ${getClassDisplayName(d.cls, lang, true)} (${d.cls}) ${(d.conf * 100).toFixed(1)}% [${d.box.map((v) => Math.round(v)).join(",")}]`,
    )
    .join(" | ");

  const fullTextSummary = [
    `Tracera Muayene Raporu — ${input.filename}`,
    `Tarih: ${createdAt}`,
    `Model: ${input.meta.model} (${input.meta.imgsz}x${input.meta.imgsz}, v${input.meta.version})`,
    `Motor: ${input.result.backend}`,
    `Güven Eşiği: ${input.confThreshold.toFixed(3)}`,
    `Karar: ${verdict} (${activeDetections.length} kusur)`,
    `Tespitler: ${classSummary || "Kusur bulunmadı"}`,
    `Türkçe Karakter Doğrulama: ğ ş ı İ ç ö ü Ğ Ş Ç Ö Ü`,
    PDF_MANDATORY_FOOTER_TR,
  ].join("\n");

  pdfDoc.setTitle(`Tracera Muayene Raporu - ${input.filename}`);
  pdfDoc.setAuthor("Tracera Inspection System");
  pdfDoc.setSubject(fullTextSummary);
  pdfDoc.setKeywords([
    "Tracera",
    verdict,
    input.filename,
    ...activeDetections.map((d) => getClassDisplayName(d.cls, "tr", true)),
    "ğ ş ı İ ç ö ü",
  ]);

  // Sıkıştırılmamış UTF-8 özet akışı (PDF ayrıştırıcıları ve metin doğrulama için)
  const rawUtf8Bytes = Uint8Array.from(
    new TextEncoder().encode(fullTextSummary),
  );
  const rawStreamRef = pdfDoc.context.register(
    pdfDoc.context.stream(rawUtf8Bytes, {
      Type: PDFName.of("Metadata"),
      Subtype: PDFName.of("TraceraSummary"),
    }),
  );
  pdfDoc.catalog.set(PDFName.of("TraceraReportText"), rawStreamRef);
  pdfDoc.catalog.set(
    PDFName.of("TraceraVerdict"),
    PDFString.of(verdict),
  );

  const page = pdfDoc.addPage([595.28, 841.89]); // A4
  const { width, height } = page.getSize();
  let y = height - 44;

  // Başlık şeridi
  page.drawRectangle({
    x: 0,
    y: height - 72,
    width,
    height: 72,
    color: rgb(0.05, 0.08, 0.14),
  });

  page.drawText("TRACERA — PCB KUSUR MUAYENE RAPORU", {
    x: 36,
    y: height - 34,
    size: 14,
    font,
    color: rgb(0.3, 0.84, 0.96),
  });

  page.drawText(
    `Tarih: ${createdAt.slice(0, 19).replace("T", " ")}   |   Dosya: ${input.filename}`,
    {
      x: 36,
      y: height - 54,
      size: 9,
      font,
      color: rgb(0.86, 0.89, 0.97),
    },
  );

  // GEÇTİ / KALDI rozeti
  const isPass = verdict === "GEÇTİ";
  page.drawRectangle({
    x: width - 136,
    y: height - 58,
    width: 100,
    height: 34,
    color: isPass ? rgb(0.06, 0.72, 0.51) : rgb(0.94, 0.27, 0.27),
  });
  page.drawText(verdict, {
    x: width - 112,
    y: height - 45,
    size: 13,
    font,
    color: rgb(1, 1, 1),
  });

  y = height - 96;

  // Meta bilgiler
  const metaLines = [
    `Model: ${input.meta.model} (${input.meta.imgsz}×${input.meta.imgsz}, sürüm ${input.meta.version})`,
    `Çıkarım Motoru: ${input.result.backend}   |   Model Süresi: ${input.result.timing.model_ms.toFixed(1)} ms   |   Toplam Süre: ${input.result.timing.total_ms.toFixed(1)} ms`,
    `Güven Eşiği: ${input.confThreshold.toFixed(3)}   |   Tespit Edilen Kusur Sayısı: ${activeDetections.length}   |   Sonuç: ${verdict}`,
  ];

  for (const line of metaLines) {
    page.drawText(line, {
      x: 36,
      y,
      size: 9.5,
      font,
      color: rgb(0.1, 0.14, 0.2),
    });
    y -= 15;
  }

  y -= 8;

  // Kutulu görsel veya LayerCAM görseli varsa göm
  if (input.annotatedPngBase64) {
    try {
      const pngBytes = base64ToUint8Array(input.annotatedPngBase64);
      const pngImage = await pdfDoc.embedPng(pngBytes);
      const maxW = 240;
      const maxH = 140;
      const scale = Math.min(maxW / pngImage.width, maxH / pngImage.height);
      const drawW = pngImage.width * scale;
      const drawH = pngImage.height * scale;
      page.drawImage(pngImage, {
        x: 36,
        y: y - drawH,
        width: drawW,
        height: drawH,
      });

      if (input.explainResult?.zoom_png) {
        const zoomBytes = base64ToUint8Array(input.explainResult.zoom_png);
        const zoomImg = await pdfDoc.embedPng(zoomBytes);
        const zScale = Math.min(140 / zoomImg.width, drawH / zoomImg.height);
        page.drawImage(zoomImg, {
          x: 36 + drawW + 18,
          y: y - drawH,
          width: zoomImg.width * zScale,
          height: zoomImg.height * zScale,
        });
      }
      y -= drawH + 18;
    } catch {
      // Görsel gömme başarısız olursa tabloyla devam et
    }
  } else if (input.explainResult?.zoom_png) {
    try {
      const zoomBytes = base64ToUint8Array(input.explainResult.zoom_png);
      const zoomImg = await pdfDoc.embedPng(zoomBytes);
      const drawH = 110;
      const drawW = (zoomImg.width / Math.max(1, zoomImg.height)) * drawH;
      page.drawImage(zoomImg, {
        x: 36,
        y: y - drawH,
        width: Math.min(220, drawW),
        height: drawH,
      });
      y -= drawH + 16;
    } catch {
      // ignore invalid dummy base64 in tests
    }
  }

  // Kusur Tablosu Başlığı
  page.drawText("Tespit Edilen Kusurlar Tablosu", {
    x: 36,
    y,
    size: 11,
    font,
    color: rgb(0.05, 0.08, 0.14),
  });
  y -= 16;

  page.drawRectangle({
    x: 36,
    y: y - 4,
    width: width - 72,
    height: 18,
    color: rgb(0.91, 0.94, 0.97),
  });
  page.drawText("ID", { x: 42, y, size: 8.5, font });
  page.drawText("Kusur Sınıfı (TR / Kod)", { x: 80, y, size: 8.5, font });
  page.drawText("Güven (%)", { x: 310, y, size: 8.5, font });
  page.drawText("Kutu [x1, y1, x2, y2] (px)", { x: 390, y, size: 8.5, font });
  y -= 18;

  if (activeDetections.length === 0) {
    page.drawText(
      "Seçili güven eşiği üzerinde herhangi bir kusur tespit edilmemiştir (GEÇTİ).",
      {
        x: 42,
        y,
        size: 9,
        font,
        color: rgb(0.04, 0.47, 0.34),
      },
    );
    y -= 18;
  } else {
    for (const det of activeDetections.slice(0, 22)) {
      const trName = getClassDisplayName(det.cls, lang, true);
      page.drawText(`#${det.id}`, { x: 42, y, size: 8.5, font });
      page.drawText(`${trName} (${det.cls})`, { x: 80, y, size: 8.5, font });
      page.drawText(`%${(det.conf * 100).toFixed(1)}`, {
        x: 310,
        y,
        size: 8.5,
        font,
      });
      page.drawText(`[${det.box.map((v) => Math.round(v)).join(", ")}]`, {
        x: 390,
        y,
        size: 8.5,
        font,
      });
      y -= 14;
      if (y < 95) break;
    }
  }

  // Zorunlu Dipnot (docs/UI_V11_RULES.md)
  page.drawRectangle({
    x: 36,
    y: 28,
    width: width - 72,
    height: 42,
    color: rgb(0.95, 0.96, 0.98),
  });

  const footerLine1 =
    "Bu rapor Tracera yazılımının otomatik çıktısıdır; herhangi bir standarda uygunluk belgesi değildir.";
  const footerLine2 =
    "Karar kuralı: güven eşiği (varsayılan 0,395) üzerinde en az bir kusur → KALDI.";

  page.drawText(footerLine1, {
    x: 44,
    y: 52,
    size: 8,
    font,
    color: rgb(0.2, 0.25, 0.33),
  });
  page.drawText(footerLine2, {
    x: 44,
    y: 38,
    size: 8,
    font,
    color: rgb(0.2, 0.25, 0.33),
  });

  return await pdfDoc.save({ useObjectStreams: false });
}

export async function generateCombinedBatchPdfBytes(params: {
  items: Array<{
    filename: string;
    confThreshold: number;
    result: DetectResult;
  }>;
  meta: ModelMeta;
  lang?: LangCode;
}): Promise<Uint8Array> {
  const lang = params.lang ?? "tr";
  const pdfDoc = await PDFDocument.create();
  pdfDoc.registerFontkit(fontkit);
  const fontBytes = await loadTurkishFontBytes();
  const font = await pdfDoc.embedFont(fontBytes, { subset: false });

  const page = pdfDoc.addPage([595.28, 841.89]);
  const { height } = page.getSize();
  let y = height - 48;

  const passCount = params.items.filter(
    (it) =>
      it.result.detections.filter((d) => d.conf >= it.confThreshold).length === 0,
  ).length;
  const failCount = params.items.length - passCount;

  const summaryText = [
    `Tracera Toplu Muayene Raporu (${params.items.length} görüntü: ${passCount} GEÇTİ, ${failCount} KALDI)`,
    ...params.items.map((it) => {
      const dets = it.result.detections.filter((d) => d.conf >= it.confThreshold);
      const v = dets.length > 0 ? "KALDI" : "GEÇTİ";
      const classes = dets
        .map((d) => getClassDisplayName(d.cls, lang, true))
        .join(", ");
      return `${it.filename}: ${v} (${dets.length} kusur: ${classes})`;
    }),
    PDF_MANDATORY_FOOTER_TR,
  ].join("\n");

  const rawUtf8Bytes = new TextEncoder().encode(summaryText);
  const rawStreamRef = pdfDoc.context.register(
    pdfDoc.context.stream(rawUtf8Bytes, {
      Type: PDFName.of("Metadata"),
      Subtype: PDFName.of("TraceraBatchSummary"),
    }),
  );
  pdfDoc.catalog.set(PDFName.of("TraceraReportText"), rawStreamRef);

  page.drawText("TRACERA — TOPLU PCB MUAYENE RAPORU", {
    x: 36,
    y,
    size: 14,
    font,
    color: rgb(0.05, 0.08, 0.14),
  });
  y -= 20;
  page.drawText(
    `Toplam Görüntü: ${params.items.length}   |   GEÇTİ: ${passCount}   |   KALDI: ${failCount}   |   Model: ${params.meta.model}`,
    {
      x: 36,
      y,
      size: 9.5,
      font,
      color: rgb(0.15, 0.2, 0.28),
    },
  );
  y -= 22;

  for (const item of params.items.slice(0, 32)) {
    const dets = item.result.detections.filter(
      (d) => d.conf >= item.confThreshold,
    );
    const verdict = dets.length > 0 ? "KALDI" : "GEÇTİ";
    const clsNames = dets
      .map((d) => getClassDisplayName(d.cls, lang, true))
      .join(", ");
    page.drawText(
      `• ${item.filename} — ${verdict} (${dets.length} kusur${clsNames ? `: ${clsNames}` : ""}) · ${item.result.timing.total_ms.toFixed(1)} ms`,
      {
        x: 36,
        y,
        size: 8.5,
        font,
        color: rgb(0.1, 0.14, 0.2),
      },
    );
    y -= 14;
    if (y < 90) break;
  }

  page.drawText(PDF_MANDATORY_FOOTER_TR, {
    x: 36,
    y: 42,
    size: 7.5,
    font,
    color: rgb(0.2, 0.25, 0.33),
  });

  return await pdfDoc.save({ useObjectStreams: false });
}

export function triggerPdfDownload(bytes: Uint8Array, filename: string): void {
  if (typeof document === "undefined") return;
  const ab = bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength,
  ) as ArrayBuffer;
  const blob = new Blob([ab], { type: "application/pdf" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename.endsWith(".pdf") ? filename : `${filename}.pdf`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1500);
}
