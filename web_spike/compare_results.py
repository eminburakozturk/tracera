"""Tarayıcı ORT-Web (WebGPU/WASM) çıktıları ile Python referans çıktılarını karşılaştırır ve results/web_spike.csv üretir."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "web" / "public" / "models"
SPIKE_DIR = REPO_ROOT / "web_spike"
REF_DIR = SPIKE_DIR / "ref"
BROWSER_DIR = SPIKE_DIR / "browser_out"
SAMPLES_JSON = SPIKE_DIR / "samples.json"
OUT_CSV = REPO_ROOT / "results" / "web_spike.csv"


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(0.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def match_detections(
    browser_dets: list[dict],
    ref_dets: list[dict],
    iou_min: float = 0.5,
) -> tuple[list[tuple[float, float]], int, int]:
    """Aynı sınıftaki kutuları en yüksek IoU ile açgözlü eşler. Döner: ([(iou, |dconf|)], matched_count, ref_total)."""
    pairs: list[tuple[float, int, int]] = []
    for bi, b in enumerate(browser_dets):
        for ri, r in enumerate(ref_dets):
            if int(b["cls_id"]) != int(r["cls_id"]):
                continue
            iou = compute_iou(b["box"], r["box"])
            if iou >= iou_min:
                pairs.append((iou, bi, ri))

    pairs.sort(key=lambda x: x[0], reverse=True)
    used_b: set[int] = set()
    used_r: set[int] = set()
    matched_metrics: list[tuple[float, float]] = []

    for iou, bi, ri in pairs:
        if bi in used_b or ri in used_r:
            continue
        used_b.add(bi)
        used_r.add(ri)
        dconf = abs(float(browser_dets[bi]["conf"]) - float(ref_dets[ri]["conf"]))
        matched_metrics.append((iou, dconf))

    return matched_metrics, len(matched_metrics), len(ref_dets)


def evaluate_against_ref(browser_results: dict[str, dict], samples: list[str], ref_tag: str) -> dict[str, float]:
    count_matches = 0
    all_ious: list[float] = []
    all_dconfs: list[float] = []
    total_matched = 0
    total_ref = 0

    for fname in samples:
        stem = Path(fname).stem
        ref_path = REF_DIR / f"{stem}__{ref_tag}.json"
        ref_data = json.loads(ref_path.read_text(encoding="utf-8"))
        b_data = browser_results[fname]

        b_dets = b_data["detections"]
        r_dets = ref_data["detections"]
        if len(b_dets) == len(r_dets):
            count_matches += 1

        matched, n_m, n_r = match_detections(b_dets, r_dets, iou_min=0.3)
        total_matched += n_m
        total_ref += n_r
        for iou, dconf in matched:
            all_ious.append(iou)
            all_dconfs.append(dconf)

    n_imgs = len(samples)
    match_rate = count_matches / n_imgs if n_imgs > 0 else 0.0
    ort_iou = sum(all_ious) / len(all_ious) if all_ious else 0.0
    ort_dconf = sum(all_dconfs) / len(all_dconfs) if all_dconfs else 0.0
    return {
        "count_match_rate": round(match_rate, 4),
        "box_recall_vs_ref": round(total_matched / total_ref, 4) if total_ref > 0 else 0.0,
        "ort_iou": round(ort_iou, 4),
        "ort_dconf": round(ort_dconf, 6),
        "count_matches": count_matches,
        "n_imgs": n_imgs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tarayıcı vs Python karşılaştırması")
    parser.add_argument("--ref", default="pytorch", choices=["pytorch", "openvino-int8"])
    args = parser.parse_args()

    samples = json.loads(SAMPLES_JSON.read_text(encoding="utf-8"))
    model_sizes = {
        "fp32": round((MODELS_DIR / "tracera_1024.onnx").stat().st_size / (1024 * 1024), 2),
        "int8": round((MODELS_DIR / "tracera_1024_int8.onnx").stat().st_size / (1024 * 1024), 2),
    }

    combos = [
        ("webgpu", "fp32", "ort-webgpu", "FP32 ONNX"),
        ("webgpu", "int8", "ort-webgpu", "INT8 Dynamic ONNX"),
        ("wasm", "fp32", "ort-wasm", "FP32 ONNX"),
        ("wasm", "int8", "ort-wasm", "INT8 Dynamic ONNX"),
    ]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    print(f"=== Tarayıcı vs Python ({args.ref}) Karşılaştırması (n={len(samples)}) ===")
    for b_key, m_key, b_label, m_label in combos:
        fpath = BROWSER_DIR / f"browser_{b_key}_{m_key}.json"
        if not fpath.is_file():
            print(f"[EKSİK] {fpath}")
            continue
        payload = json.loads(fpath.read_text(encoding="utf-8"))
        boyut_mb = model_sizes[m_key]

        if payload.get("status") != "ok":
            err = str(payload.get("error", "bilinmeyen hata"))
            print(f"{b_label:<12} | {m_label:<18} | HATA: {err[:120]}")
            rows.append(
                {
                    "backend": b_label,
                    "model": m_label,
                    "boyut_mb": boyut_mb,
                    "medyan_ms": "",
                    "eşleşme_oranı": "",
                    "ort_iou": "",
                    "ort_dconf": "",
                }
            )
            continue

        metrics_pt = evaluate_against_ref(payload["results"], samples, "pytorch")
        metrics_ov = evaluate_against_ref(payload["results"], samples, "openvino-int8")
        chosen = metrics_pt if args.ref == "pytorch" else metrics_ov
        med_ms = payload["medianModelMs"]

        print(
            f"{payload['activeBackend']:<12} | {m_label:<18} | {boyut_mb:>6.2f} MB | "
            f"medyan_model={med_ms:>7.2f} ms (total={payload['medianTotalMs']:>7.2f} ms) | "
            f"vs PT: eşleşme={metrics_pt['count_match_rate']:.2f} ({metrics_pt['count_matches']}/{metrics_pt['n_imgs']}), "
            f"IoU={metrics_pt['ort_iou']:.4f}, |dconf|={metrics_pt['ort_dconf']:.5f} | "
            f"vs OV-INT8: eşleşme={metrics_ov['count_match_rate']:.2f}, IoU={metrics_ov['ort_iou']:.4f}"
        )

        rows.append(
            {
                "backend": payload["activeBackend"],
                "model": m_label,
                "boyut_mb": boyut_mb,
                "medyan_ms": med_ms,
                "eşleşme_oranı": chosen["count_match_rate"],
                "ort_iou": chosen["ort_iou"],
                "ort_dconf": chosen["ort_dconf"],
            }
        )

    fieldnames = ["backend", "model", "boyut_mb", "medyan_ms", "eşleşme_oranı", "ort_iou", "ort_dconf"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\nKaydedildi: {OUT_CSV}")


if __name__ == "__main__":
    main()
