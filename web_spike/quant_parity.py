"""PyTorch FP32 vs OpenVINO INT8 niceleme kıyası (conf >= 0.25 ve conf >= 0.395)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
REF_DIR = REPO_ROOT / "web_spike" / "ref"
SAMPLES_JSON = REPO_ROOT / "web_spike" / "samples.json"
OUT_CSV = REPO_ROOT / "results" / "quant_parity.csv"


def box_iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / max(1e-9, area_a + area_b - inter)


def match_detections(
    fp32_dets: list[dict[str, Any]],
    int8_dets: list[dict[str, Any]],
    iou_thresh: float = 0.5,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any], float]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Aynı sınıf + IoU >= 0.5 ile iki yönlü birebir (greedy max-IoU) eşleştirme yapar."""
    candidates: list[tuple[float, int, int]] = []
    for i, df in enumerate(fp32_dets):
        for j, di in enumerate(int8_dets):
            if int(df["cls_id"]) != int(di["cls_id"]):
                continue
            iou = box_iou(df["box"], di["box"])
            if iou >= iou_thresh:
                candidates.append((iou, i, j))
    candidates.sort(key=lambda t: (-t[0], t[1], t[2]))

    used_fp32: set[int] = set()
    used_int8: set[int] = set()
    matched: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    for iou, i, j in candidates:
        if i in used_fp32 or j in used_int8:
            continue
        used_fp32.add(i)
        used_int8.add(j)
        matched.append((fp32_dets[i], int8_dets[j], iou))

    only_fp32 = [d for idx, d in enumerate(fp32_dets) if idx not in used_fp32]
    only_int8 = [d for idx, d in enumerate(int8_dets) if idx not in used_int8]
    return matched, only_fp32, only_int8


def main() -> None:
    samples = json.loads(SAMPLES_JSON.read_text(encoding="utf-8"))
    thresholds = [0.25, 0.395]
    rows: list[dict[str, Any]] = []

    for thr in thresholds:
        total_matched = 0
        total_only_fp32 = 0
        total_only_int8 = 0
        ious: list[float] = []
        dconfs: list[float] = []
        unmatched_details: list[str] = []

        for fname in samples:
            stem = Path(fname).stem
            pt_data = json.loads((REF_DIR / f"{stem}__pytorch_c025.json").read_text(encoding="utf-8"))
            ov_data = json.loads((REF_DIR / f"{stem}__openvino-int8_c025.json").read_text(encoding="utf-8"))

            fp32_dets = [d for d in pt_data["detections"] if float(d["conf"]) >= thr]
            int8_dets = [d for d in ov_data["detections"] if float(d["conf"]) >= thr]

            matched, only_fp32, only_int8 = match_detections(fp32_dets, int8_dets, iou_thresh=0.5)
            total_matched += len(matched)
            total_only_fp32 += len(only_fp32)
            total_only_int8 += len(only_int8)

            for df, di, iou in matched:
                ious.append(iou)
                dconfs.append(abs(float(df["conf"]) - float(di["conf"])))

            for d in only_fp32:
                unmatched_details.append(
                    f"  [yalnız FP32] {fname} | {d['cls']} | conf={d['conf']:.6f} | box={d['box']}"
                )
            for d in only_int8:
                unmatched_details.append(
                    f"  [yalnız INT8] {fname} | {d['cls']} | conf={d['conf']:.6f} | box={d['box']}"
                )

        mean_iou = sum(ious) / len(ious) if ious else 0.0
        mean_dconf = sum(dconfs) / len(dconfs) if dconfs else 0.0
        rows.append(
            {
                "conf_esik": thr,
                "eslesen": total_matched,
                "yalniz_fp32": total_only_fp32,
                "yalniz_int8": total_only_int8,
                "ort_iou": round(mean_iou, 6),
                "ort_dconf": round(mean_dconf, 6),
            }
        )

        print(
            f"\n=== conf >= {thr} ===\n"
            f"eşleşen={total_matched} | yalnız_fp32={total_only_fp32} | yalnız_int8={total_only_int8} | "
            f"ort_iou={mean_iou:.6f} | ort_dconf={mean_dconf:.6f}"
        )
        if unmatched_details:
            print("Yalnız birinde olan kutular:")
            for line in unmatched_details:
                print(line)
        else:
            print("Yalnız birinde olan kutu yok.")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["conf_esik", "eslesen", "yalniz_fp32", "yalniz_int8", "ort_iou", "ort_dconf"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nYazıldı: {OUT_CSV}")


if __name__ == "__main__":
    main()
