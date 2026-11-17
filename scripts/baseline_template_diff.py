"""Klasik şablon farkı karşılaştırması (KARARLAR.md 2026-11-14 ön kaydı).

Alt komutlar:
  yolo    : FP32 ONNX (tracera_1024.onnx) ile sınıfsız karşılaştırma için YOLO tahminlerini
            conf >= 0.25 olarak önbelleğe alır (results/baseline/yolo_preds_<split>.json).
            --parity: web_spike/ref PyTorch referanslarıyla numpy ön/son işlem paritesini ölçer.
  tune    : şablon farkı parametrelerini YALNIZ train kartlarında ızgara aramasıyla seçer.
  eval    : seçilen parametrelerle val ve test'i bir kez değerlendirir; YOLO ile aynı eşleştirme.

Sınıfsız değerlendirme: açgözlü eşleştirme (en yüksek IoU önce), IoU >= 0.5 ve gevşek IoU >= 0.1.
Şablon farkı yöntemi hizalama yapmaz: PKU kusurlu görüntüleri PCB_USED şablonuyla piksel hizalıdır
(ölçüldü: faz korelasyonu kayması ~0.001 px).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw" / "PCB_DATASET"
META_CSV = ROOT / "data" / "yolo" / "meta.csv"
OUT = ROOT / "results" / "baseline"
ONNX = ROOT / "web" / "public" / "models" / "tracera_1024.onnx"
IMGSZ = 1024
PAD = 114
NMS_IOU = 0.7
CLASSES = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]
FOLDER = {
    "missing_hole": "Missing_hole",
    "mouse_bite": "Mouse_bite",
    "open_circuit": "Open_circuit",
    "short": "Short",
    "spur": "Spur",
    "spurious_copper": "Spurious_copper",
}


# ---------------------------------------------------------------- veri
def load_split(split: str) -> list[dict]:
    rows = []
    with META_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["grouped_split"] != split:
                continue
            cls = r["sınıf"]
            img = DATA / "images" / FOLDER[cls] / r["dosya"]
            xml = DATA / "Annotations" / FOLDER[cls] / (Path(r["dosya"]).stem + ".xml")
            rows.append({"file": r["dosya"], "kart": f"{int(r['kart_no']):02d}", "img": img, "xml": xml})
    rows.sort(key=lambda r: r["file"])
    return rows


def gt_boxes(xml: Path) -> list[tuple[float, float, float, float]]:
    out = []
    for o in ET.parse(xml).getroot().findall("object"):
        b = o.find("bndbox")
        out.append(tuple(float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")))
    return out


# ---------------------------------------------------------------- eşleştirme
def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match(preds, gts, thr: float) -> tuple[int, int, int]:
    """Açgözlü sınıfsız eşleştirme → (TP, FP, FN)."""
    pairs = sorted(
        ((iou(p, g), i, j) for i, p in enumerate(preds) for j, g in enumerate(gts)),
        reverse=True,
    )
    used_p, used_g, tp = set(), set(), 0
    for v, i, j in pairs:
        if v < thr:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        tp += 1
    return tp, len(preds) - tp, len(gts) - tp


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "P": round(p, 4), "R": round(r, 4), "F1": round(f, 4)}


# ---------------------------------------------------------------- şablon farkı
_TEMPLATES: dict[str, np.ndarray] = {}


def template(kart: str) -> np.ndarray:
    if kart not in _TEMPLATES:
        _TEMPLATES[kart] = cv2.imread(str(DATA / "PCB_USED" / f"{kart}.JPG"), cv2.IMREAD_GRAYSCALE)
    return _TEMPLATES[kart]


def diff_map(img_path: Path, kart: str, blur: int) -> np.ndarray:
    im = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    d = cv2.absdiff(im, template(kart))
    if blur > 1:
        d = cv2.GaussianBlur(d, (blur, blur), 0)
    return d


def boxes_from_diff(d: np.ndarray, thr: int, dilate: int, min_area: int) -> list[tuple]:
    m = (d >= thr).astype(np.uint8)
    if dilate > 1:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if area < min_area:
            continue
        out.append((float(x), float(y), float(x + w), float(y + h)))
    return out


GRID = {
    "blur": [1, 5],
    "thr": [20, 30, 40, 60],
    "dilate": [1, 9, 21],
    "min_area": [20, 100, 300],
}
# Ön kayıttaki ızgaranın en iyisi sınırda çıktı (thr=20, dilate=21) → yalnız train'de genişletildi
# (KARARLAR 2026-11-14; val/test bu noktada hiç değerlendirilmemişti).
GRID_EXT = {
    "blur": [1, 3],
    "thr": [8, 12, 16, 20],
    "dilate": [21, 31, 45, 61],
    "min_area": [20, 300, 1000],
}


def cmd_tune(_args) -> None:
    rows = load_split("train")
    grid = GRID_EXT if _args.grid == "ext" else GRID
    sfx = "_ext" if _args.grid == "ext" else ""
    combos = list(itertools.product(grid["blur"], grid["thr"], grid["dilate"], grid["min_area"]))
    OUT.mkdir(parents=True, exist_ok=True)
    part = OUT / f"template_diff_tune_train{sfx}.partial.json"
    done: dict[str, list] = json.loads(part.read_text(encoding="utf-8")) if part.exists() else {}
    t0 = time.time()
    for r in rows:
        if r["file"] in done:
            continue
        if time.time() - t0 > _args.budget:
            part.write_text(json.dumps(done), encoding="utf-8")
            print(f"[tune] {len(done)}/{len(rows)} — süre doldu, devam için tekrar çalıştır")
            return
        gts = gt_boxes(r["xml"])
        cache = {b: diff_map(r["img"], r["kart"], b) for b in grid["blur"]}
        per = []
        for c in combos:
            blur, thr, dil, ma = c
            per.append(list(match(boxes_from_diff(cache[blur], thr, dil, ma), gts, 0.5)))
        done[r["file"]] = per
    part.write_text(json.dumps(done), encoding="utf-8")
    acc = {c: [0, 0, 0] for c in combos}
    for per in done.values():
        for c, (tp, fp, fn) in zip(combos, per):
            acc[c][0] += tp
            acc[c][1] += fp
            acc[c][2] += fn
    res = []
    for c, (tp, fp, fn) in acc.items():
        m = prf(tp, fp, fn)
        res.append({"blur": c[0], "thr": c[1], "dilate": c[2], "min_area": c[3], **m})
    res.sort(key=lambda x: (-x["F1"], -x["R"]))
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"template_diff_tune_train{sfx}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0].keys()))
        w.writeheader()
        w.writerows(res)
    best = res[0]
    (OUT / "template_diff_params.json").write_text(
        json.dumps({"selected_on": "train (01,04,07,08,09,12)", **best}, indent=2), encoding="utf-8"
    )
    print("[tune] en iyi (train):", best)


# ---------------------------------------------------------------- YOLO (ONNX, numpy)
def letterbox(img: np.ndarray):
    h, w = img.shape[:2]
    r = min(IMGSZ / h, IMGSZ / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (IMGSZ - nw) / 2, (IMGSZ - nh) / 2
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD, PAD, PAD))
    return out, r, left, top


def nms_classwise(boxes, scores, cls, iou_thr):
    keep = []
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        order = idx[np.argsort(-scores[idx])]
        sel = []
        for i in order:
            if all(iou(boxes[i], boxes[j]) <= iou_thr for j in sel):
                sel.append(i)
        keep.extend(sel)
    keep.sort(key=lambda i: -scores[i])
    return keep[:300]


_SESS = None


def yolo_detect(img_path: Path, conf: float = 0.25) -> list[dict]:
    global _SESS
    import onnxruntime as ort

    if _SESS is None:
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        _SESS = ort.InferenceSession(str(ONNX), so, providers=["CPUExecutionProvider"])
    img = cv2.imread(str(img_path))
    h0, w0 = img.shape[:2]
    lb, r, left, top = letterbox(img)
    x = lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    out = _SESS.run(None, {_SESS.get_inputs()[0].name: np.ascontiguousarray(x)})[0][0]  # (10, N)
    scores_all = out[4:]
    cls = scores_all.argmax(0)
    scores = scores_all.max(0)
    m = scores >= conf
    cx, cy, w, h = out[0][m], out[1][m], out[2][m], out[3][m]
    b = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    s, c = scores[m], cls[m]
    keep = nms_classwise(b, s, c, NMS_IOU)
    dets = []
    for i in keep:
        x1 = float(np.clip((b[i, 0] - left) / r, 0, w0))
        y1 = float(np.clip((b[i, 1] - top) / r, 0, h0))
        x2 = float(np.clip((b[i, 2] - left) / r, 0, w0))
        y2 = float(np.clip((b[i, 3] - top) / r, 0, h0))
        dets.append({"cls": CLASSES[int(c[i])], "conf": float(s[i]), "box": [x1, y1, x2, y2]})
    return dets


def cmd_yolo(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if args.parity:
        refs = json.loads((ROOT / "web_spike" / "samples.json").read_text(encoding="utf-8"))
        meta = {r["file"]: r for s in ("val",) for r in load_split(s)}
        n_ref = n_match = 0
        dconf, ious = [], []
        for f in refs:
            if f not in meta:
                continue
            ref = json.loads((ROOT / "web_spike" / "ref" / f"{Path(f).stem}__pytorch_c025.json").read_text())
            mine = yolo_detect(meta[f]["img"], 0.25)
            for rd in ref["detections"]:
                n_ref += 1
                best = max(
                    ((iou(rd["box"], md["box"]), md) for md in mine if md["cls"] == rd["cls"]),
                    default=(0.0, None),
                    key=lambda t: t[0],
                )
                if best[0] >= 0.5:
                    n_match += 1
                    ious.append(best[0])
                    dconf.append(abs(best[1]["conf"] - rd["conf"]))
            print(f"[parity] {f}: ref {len(ref['detections'])} / numpy {len(mine)}", flush=True)
        rep = {
            "ref_dets": n_ref,
            "matched": n_match,
            "match_rate": round(n_match / max(1, n_ref), 4),
            "mean_iou": round(float(np.mean(ious)), 4) if ious else None,
            "mean_abs_dconf": round(float(np.mean(dconf)), 5) if dconf else None,
            "max_abs_dconf": round(float(np.max(dconf)), 5) if dconf else None,
        }
        (OUT / "yolo_numpy_parity.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print("[parity]", rep)
        return
    t0 = time.time()
    for split in args.splits:
        rows = load_split(split)
        path = OUT / f"yolo_preds_{split}.json"
        part = OUT / f"yolo_preds_{split}.partial.json"
        if path.exists():
            print(f"[yolo] {split} zaten tamam")
            continue
        preds = json.loads(part.read_text(encoding="utf-8")) if part.exists() else {}
        for r in rows:
            if r["file"] in preds:
                continue
            if time.time() - t0 > args.budget:
                part.write_text(json.dumps(preds), encoding="utf-8")
                print(f"[yolo] {split} {len(preds)}/{len(rows)} — süre doldu, devam için tekrar çalıştır")
                return
            preds[r["file"]] = yolo_detect(r["img"], 0.25)
        path.write_text(json.dumps(preds), encoding="utf-8")
        if part.exists():
            part.unlink()
        print(f"[yolo] {split} tamam ({len(rows)} görüntü)")


# ---------------------------------------------------------------- değerlendirme
def cmd_eval(_args) -> None:
    params = json.loads((OUT / "template_diff_params.json").read_text(encoding="utf-8"))
    rows_out = []
    for split in ("val", "test"):
        rows = load_split(split)
        yolo = json.loads((OUT / f"yolo_preds_{split}.json").read_text(encoding="utf-8"))
        acc = {k: [0, 0, 0] for k in ("td_50", "td_10", "yolo_50", "yolo_10")}
        per_card: dict[str, dict] = {}
        for r in rows:
            gts = gt_boxes(r["xml"])
            d = diff_map(r["img"], r["kart"], params["blur"])
            td = boxes_from_diff(d, params["thr"], params["dilate"], params["min_area"])
            yb = [p["box"] for p in yolo[r["file"]] if p["conf"] >= 0.395]
            for key, preds, thr in (("td_50", td, 0.5), ("td_10", td, 0.1), ("yolo_50", yb, 0.5), ("yolo_10", yb, 0.1)):
                tp, fp, fn = match(preds, gts, thr)
                a = acc[key]
                a[0] += tp
                a[1] += fp
                a[2] += fn
                pc = per_card.setdefault(r["kart"], {k: [0, 0, 0] for k in acc})[key]
                pc[0] += tp
                pc[1] += fp
                pc[2] += fn
        for key, (tp, fp, fn) in acc.items():
            method, thr = key.split("_")
            rows_out.append({"split": split, "kart": "tümü", "yöntem": "şablon farkı" if method == "td" else "YOLO11s (0.395)",
                             "iou": f"0.{thr[0]}" if thr != "50" else "0.5", **prf(tp, fp, fn)})
        for kart, d in sorted(per_card.items()):
            for key, (tp, fp, fn) in d.items():
                method, thr = key.split("_")
                rows_out.append({"split": split, "kart": kart, "yöntem": "şablon farkı" if method == "td" else "YOLO11s (0.395)",
                                 "iou": "0.1" if thr == "10" else "0.5", **prf(tp, fp, fn)})
    with (OUT / "template_diff_vs_yolo.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    for r in rows_out:
        if r["kart"] == "tümü":
            print(r)



def cmd_classaware(_args) -> None:
    """Uygulamanın kare letterbox ön işlemesiyle (ONNX FP32) sınıflı P/R/F1 @ 0.395, IoU 0.5.
    Raporlanan çalışma noktası DetectionValidator rect=True ile ölçüldü; bu tablo farkı gösterir."""
    rows_out = []
    for split in ("val", "test"):
        preds = json.loads((OUT / f"yolo_preds_{split}.json").read_text(encoding="utf-8"))
        tp = fp = fn = 0
        for r in load_split(split):
            objs = ET.parse(r["xml"]).getroot().findall("object")
            g = [(o.find("name").text, tuple(float(o.find("bndbox").find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))) for o in objs]
            for c in CLASSES:
                pp = [d["box"] for d in preds[r["file"]] if d["conf"] >= 0.395 and d["cls"] == c]
                gg = [b for n, b in g if n == c]
                a, b2, c2 = match(pp, gg, 0.5)
                tp += a
                fp += b2
                fn += c2
        rows_out.append({"split": split, "on_isleme": "kare letterbox (uygulama)", **prf(tp, fp, fn)})
    with (OUT / "yolo_square_classaware.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    for r in rows_out:
        print(r)

def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    y = sub.add_parser("yolo")
    y.add_argument("--splits", nargs="+", default=["val", "test"])
    y.add_argument("--parity", action="store_true")
    y.add_argument("--budget", type=float, default=150.0)
    t = sub.add_parser("tune")
    t.add_argument("--budget", type=float, default=150.0)
    t.add_argument("--grid", choices=["base", "ext"], default="base")
    sub.add_parser("eval")
    sub.add_parser("classaware")
    a = ap.parse_args()
    {"yolo": cmd_yolo, "tune": cmd_tune, "eval": cmd_eval, "classaware": cmd_classaware}[a.cmd](a)


if __name__ == "__main__":
    main()
