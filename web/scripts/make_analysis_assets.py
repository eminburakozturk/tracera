"""Hata analizi galerisi verilerini (analysis.json) ve kırpım görsellerini üretir.

Kaynaklar (salt okunur):
- results/error_analysis.csv
- results/template_check.csv
- results/quant_parity.csv
- results/template_check_06.png
- results/fp_montage_mouse_bite.png
- results/fp_montage_spur.png

Not: scripts/make_web_metrics.py çıktısı web/public/metrics.json'a dokunmaz.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
OUT_JSON = REPO_ROOT / "web" / "public" / "analysis.json"
OUT_IMG_DIR = REPO_ROOT / "web" / "public" / "analysis"

CLASSES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]


def crop_save(src_path: Path, x: int, y: int, w: int, h: int, dst_path: Path, out_size: int = 384) -> None:
    img = cv2.imread(str(src_path))
    if img is None:
        raise FileNotFoundError(f"Görsel okunamadı: {src_path}")
    crop = img[y : y + h, x : x + w]
    resized = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst_path), resized, [cv2.IMWRITE_PNG_COMPRESSION, 9])


def build_board_class_matrix(err_rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], int]:
    grouped: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    totals: dict[tuple[str, str], int] = Counter()
    splits: dict[str, str] = {}

    for row in err_rows:
        card = row["kart_no"]
        cls = row["sınıf"]
        split = row["split"]
        splits[card] = split
        grouped[(split, card)][cls] += 1
        totals[(split, card)] += 1

    order_keys = sorted(totals.keys(), key=lambda k: (0 if k[0] == "test" else 1, k[1]))
    rows_out: list[dict[str, object]] = []
    for split, card in order_keys:
        counts = {cls: int(grouped[(split, card)][cls]) for cls in CLASSES}
        rows_out.append(
            {
                "split": split,
                "kart_no": card,
                "counts": counts,
                "total": int(totals[(split, card)]),
            }
        )
    return rows_out, len(err_rows)


def build_template_spots(
    err_rows: list[dict[str, str]],
    tpl_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    specs = [
        {
            "id": "b06_mouse_bite_1001_1380",
            "kart_no": "06",
            "cls": "mouse_bite",
            "cx": 1001,
            "cy": 1380,
            "fp_crop": "/analysis/fp_mouse_bite_spot.png",
            "template_crop": "/analysis/template_06_mouse_bite.png",
        },
        {
            "id": "b06_spur_1708_1050",
            "kart_no": "06",
            "cls": "spur",
            "cx": 1708,
            "cy": 1050,
            "fp_crop": "/analysis/fp_spur_spot.png",
            "template_crop": "/analysis/template_06_spur.png",
        },
    ]

    out: list[dict[str, object]] = []
    for sp in specs:
        confs: list[float] = []
        for row in err_rows:
            if row["kart_no"] != sp["kart_no"] or row["sınıf"] != sp["cls"]:
                continue
            rx = (float(row["x1"]) + float(row["x2"])) / 2.0
            ry = (float(row["y1"]) + float(row["y2"])) / 2.0
            if math.hypot(rx - sp["cx"], ry - sp["cy"]) <= 35.0:
                confs.append(float(row["güven"]))

        clean_matches = []
        for trow in tpl_rows:
            if trow["kart_no"] != sp["kart_no"] or trow["sınıf"] != sp["cls"]:
                continue
            tx = float(trow["cx"])
            ty = float(trow["cy"])
            if math.hypot(tx - sp["cx"], ty - sp["cy"]) <= 35.0:
                clean_matches.append(float(trow["güven"]))

        out.append(
            {
                "id": sp["id"],
                "kart_no": sp["kart_no"],
                "cls": sp["cls"],
                "cx": sp["cx"],
                "cy": sp["cy"],
                "fp_count": len(confs),
                "conf_min": round(min(confs), 4) if confs else None,
                "conf_max": round(max(confs), 4) if confs else None,
                "template_alarm": len(clean_matches) > 0,
                "template_conf": round(max(clean_matches), 4) if clean_matches else None,
                "fp_crop": sp["fp_crop"],
                "template_crop": sp["template_crop"],
            }
        )
    return out


def main() -> None:
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

    with (RESULTS_DIR / "error_analysis.csv").open(encoding="utf-8") as f:
        err_rows = list(csv.DictReader(f))
    with (RESULTS_DIR / "template_check.csv").open(encoding="utf-8") as f:
        tpl_rows = list(csv.DictReader(f))
    with (RESULTS_DIR / "quant_parity.csv").open(encoding="utf-8") as f:
        qp_rows = list(csv.DictReader(f))

    # Montaj ve şablon kontrolü görsellerinden 4 kırpım üret
    crop_save(RESULTS_DIR / "fp_montage_mouse_bite.png", 16, 163, 558, 558, OUT_IMG_DIR / "fp_mouse_bite_spot.png")
    crop_save(RESULTS_DIR / "fp_montage_spur.png", 1284, 791, 558, 558, OUT_IMG_DIR / "fp_spur_spot.png")
    crop_save(RESULTS_DIR / "template_check_06.png", 18, 1186, 940, 940, OUT_IMG_DIR / "template_06_mouse_bite.png")
    crop_save(RESULTS_DIR / "template_check_06.png", 1089, 1186, 940, 940, OUT_IMG_DIR / "template_06_spur.png")

    board_class_matrix, total_fp_rows = build_board_class_matrix(err_rows)
    template_spots = build_template_spots(err_rows, tpl_rows)

    template_clean_rows = [
        {
            "kart_no": r["kart_no"],
            "dosya": r["dosya"],
            "cls": r["sınıf"],
            "conf": round(float(r["güven"]), 4),
            "cx": round(float(r["cx"]), 1),
            "cy": round(float(r["cy"]), 1),
        }
        for r in tpl_rows
    ]

    quant_rows = [
        {
            "conf_esik": float(r["conf_esik"]),
            "eslesen": int(r["eslesen"]),
            "yalniz_fp32": int(r["yalniz_fp32"]),
            "yalniz_int8": int(r["yalniz_int8"]),
            "ort_iou": round(float(r["ort_iou"]), 6),
            "ort_dconf": round(float(r["ort_dconf"]), 6),
        }
        for r in qp_rows
    ]

    payload = {
        "classes": CLASSES,
        "total_fp_rows": total_fp_rows,
        "board_class_matrix": board_class_matrix,
        "template_spots": template_spots,
        "template_clean_rows": template_clean_rows,
        "quant_parity": quant_rows,
    }

    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[make_analysis_assets] Yazıldı: {OUT_JSON} ({total_fp_rows} FP satırı, {len(template_spots)} sabit nokta)")


if __name__ == "__main__":
    main()
