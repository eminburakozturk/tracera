import type { Detection, ModelMeta } from "../../types";

export interface LetterboxMeta {
  ratio: number;
  left: number;
  top: number;
  right: number;
  bottom: number;
  unpadW: number;
  unpadH: number;
  origW: number;
  origH: number;
}

export interface RawCandidate {
  clsId: number;
  conf: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export function roundHalfToEven(val: number): number {
  const floorVal = Math.floor(val);
  const frac = val - floorVal;
  let rounded: number;
  if (Math.abs(frac - 0.5) < 1e-9) {
    rounded = floorVal % 2 === 0 ? floorVal : floorVal + 1;
  } else {
    rounded = Math.round(val);
  }
  return rounded === 0 ? 0 : rounded;
}

/**
 * Ultralytics letterbox parametrelerini hesaplar (oran korunur, ortalanır).
 */
export function computeLetterbox(origW: number, origH: number, imgsz: number): LetterboxMeta {
  if (origW <= 0 || origH <= 0 || imgsz <= 0) {
    throw new Error("Geçersiz görüntü veya model giriş boyutu.");
  }
  const ratio = Math.min(imgsz / origH, imgsz / origW);
  const unpadW = roundHalfToEven(origW * ratio);
  const unpadH = roundHalfToEven(origH * ratio);
  const dw = (imgsz - unpadW) / 2.0;
  const dh = (imgsz - unpadH) / 2.0;
  const left = roundHalfToEven(dw - 0.1);
  const right = roundHalfToEven(dw + 0.1);
  const top = roundHalfToEven(dh - 0.1);
  const bottom = roundHalfToEven(dh + 0.1);

  return {
    ratio,
    left,
    top,
    right,
    bottom,
    unpadW,
    unpadH,
    origW,
    origH,
  };
}

export function computeIoU(
  a: Pick<RawCandidate, "x1" | "y1" | "x2" | "y2">,
  b: Pick<RawCandidate, "x1" | "y1" | "x2" | "y2">,
): number {
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

export function unscaleBox(
  boxLb: [number, number, number, number],
  lb: LetterboxMeta,
): [number, number, number, number] {
  const [x1, y1, x2, y2] = boxLb;
  const clamp = (v: number, lo: number, hi: number) => (v < lo ? lo : v > hi ? hi : v);
  const ox1 = clamp((x1 - lb.left) / lb.ratio, 0, lb.origW);
  const oy1 = clamp((y1 - lb.top) / lb.ratio, 0, lb.origH);
  const ox2 = clamp((x2 - lb.left) / lb.ratio, 0, lb.origW);
  const oy2 = clamp((y2 - lb.top) / lb.ratio, 0, lb.origH);
  return [
    Number(ox1.toFixed(3)),
    Number(oy1.toFixed(3)),
    Number(ox2.toFixed(3)),
    Number(oy2.toFixed(3)),
  ];
}

/**
 * Sınıf bazlı NMS (IoU 0.7) ve conf eşiği uygular.
 */
export function classWiseNms(
  candidates: RawCandidate[],
  iouThreshold: number,
  maxDet: number = 300,
): RawCandidate[] {
  const sorted = [...candidates].sort((a, b) => b.conf - a.conf);
  const byClass = new Map<number, RawCandidate[]>();
  for (const cand of sorted) {
    const list = byClass.get(cand.clsId);
    if (list) list.push(cand);
    else byClass.set(cand.clsId, [cand]);
  }

  const kept: RawCandidate[] = [];
  for (const clsCands of byClass.values()) {
    const selected: RawCandidate[] = [];
    for (const cand of clsCands) {
      let suppressed = false;
      for (const prev of selected) {
        if (computeIoU(cand, prev) > iouThreshold) {
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

  kept.sort((a, b) => b.conf - a.conf);
  return kept.slice(0, maxDet);
}

/**
 * CONTRACT §1 ham ONNX çıktısı ([1, 4 + C, N]) çözümleyici.
 */
export function postprocessYolo(
  outputData: Float32Array,
  dims: readonly number[],
  lb: LetterboxMeta,
  meta: ModelMeta,
  confThreshold: number,
  maxDet: number = 300,
): Detection[] {
  const numChannels = dims[1];
  const numAnchors = dims[2];
  const numClasses = meta.classes.length;
  if (numChannels !== 4 + numClasses) {
    throw new Error(`Beklenen kanal sayısı ${4 + numClasses}, gelen: ${numChannels}`);
  }

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

  const kept = classWiseNms(candidates, meta.iou_nms, maxDet);
  return kept
    .filter((k) => k.conf >= confThreshold)
    .map((k, idx) => ({
      id: idx,
      cls: meta.classes[k.clsId] ?? `class_${k.clsId}`,
      cls_id: k.clsId,
      conf: Number(k.conf.toFixed(6)),
      box: unscaleBox([k.x1, k.y1, k.x2, k.y2], lb),
    }));
}
