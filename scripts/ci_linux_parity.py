"""CI Linux OpenVINO INT8 parite kontrol scripti (yalniz bilgilendirme / raporlama)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracera_core.config import (  # noqa: E402
    DEFAULT_IOU_NMS,
    IMGSZ,
    OPENVINO_MODEL_PATH,
    ensure_weights_from_hf,
)

SAMPLES_JSON = REPO_ROOT / "web_spike" / "samples.json"
REF_DIR = REPO_ROOT / "web_spike" / "ref"
DATA_VAL_DIR = REPO_ROOT / "data" / "yolo" / "grouped" / "val" / "images"
SPIKE_VAL_DIR = REPO_ROOT / "web_spike" / "val_images"


def compute_iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0.0 else 0.0


def resolve_val_image(filename: str) -> Path | None:
    for base in (DATA_VAL_DIR, SPIKE_VAL_DIR):
        cand = base / filename
        if cand.is_file():
            return cand
    return None


def match_detections(
    ref_dets: list[dict[str, Any]],
    cur_dets: list[dict[str, Any]],
    iou_thresh: float = 0.5,
) -> tuple[list[tuple[float, float]], int, int]:
    """Ayni sinif + IoU >= 0.5 acgozlu eslestirme: [(iou, |dconf|), ...]."""
    ref_sorted = sorted(ref_dets, key=lambda d: float(d["conf"]), reverse=True)
    cur_sorted = sorted(cur_dets, key=lambda d: float(d["conf"]), reverse=True)
    used_cur: set[int] = set()
    matches: list[tuple[float, float]] = []

    for r in ref_sorted:
        best_idx = -1
        best_iou = 0.0
        for j, c in enumerate(cur_sorted):
            if j in used_cur or r["cls"] != c["cls"]:
                continue
            iou = compute_iou(r["box"], c["box"])
            if iou >= iou_thresh and iou > best_iou:
                best_iou = iou
                best_idx = j
        if best_idx >= 0:
            used_cur.add(best_idx)
            dconf = abs(float(cur_sorted[best_idx]["conf"]) - float(r["conf"]))
            matches.append((best_iou, dconf))

    only_ref = len(ref_sorted) - len(matches)
    only_cur = len(cur_sorted) - len(matches)
    return matches, only_cur, only_ref


def write_csv(out_path: Path, rows: list[dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "status",
        "ref_count_025",
        "linux_count_025",
        "count_diff",
        "matched",
        "only_linux",
        "only_ref",
        "mean_iou",
        "mean_abs_dconf",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Linux OpenVINO INT8 parite kontrolu (bilgilendirme)")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "linux_parity.csv",
        help="Cikti CSV yolu",
    )
    args = parser.parse_args()

    if not ensure_weights_from_hf() or not (OPENVINO_MODEL_PATH / "last.xml").is_file():
        print("[linux-parity] OpenVINO agirliklari yok ve TRACERA_MODEL_REPO indirilemedi; atlaniyor.")
        write_csv(
            args.out,
            [
                {
                    "image": "SUMMARY",
                    "status": "skipped_no_weights",
                    "ref_count_025": 0,
                    "linux_count_025": 0,
                    "count_diff": 0,
                    "matched": 0,
                    "only_linux": 0,
                    "only_ref": 0,
                    "mean_iou": "",
                    "mean_abs_dconf": "",
                }
            ],
        )
        return 0

    from ultralytics import YOLO  # lazy import so skipped environments don't need ultralytics

    ov = YOLO(str(OPENVINO_MODEL_PATH), task="detect")
    samples: list[str] = json.loads(SAMPLES_JSON.read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    all_ious: list[float] = []
    all_dconfs: list[float] = []
    tot_ref = 0
    tot_cur = 0
    tot_matched = 0
    tot_only_cur = 0
    tot_only_ref = 0

    for fname in samples:
        stem = Path(fname).stem
        ref_path = REF_DIR / f"{stem}__openvino-int8_c025.json"
        img_path = resolve_val_image(fname)
        if not ref_path.is_file() or img_path is None:
            rows.append(
                {
                    "image": fname,
                    "status": "skipped_missing_file",
                    "ref_count_025": "",
                    "linux_count_025": "",
                    "count_diff": "",
                    "matched": "",
                    "only_linux": "",
                    "only_ref": "",
                    "mean_iou": "",
                    "mean_abs_dconf": "",
                }
            )
            continue

        ref_data = json.loads(ref_path.read_text(encoding="utf-8"))
        ref_dets = ref_data.get("detections", [])

        preds = ov.predict(
            source=str(img_path),
            imgsz=IMGSZ,
            conf=0.25,
            iou=DEFAULT_IOU_NMS,
            verbose=False,
        )
        res = preds[0]
        names = res.names
        cur_dets: list[dict[str, Any]] = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy()
            for idx in range(len(xyxy)):
                cid = int(clss[idx])
                cls_name = names[cid] if isinstance(names, (dict, list)) else str(cid)
                cur_dets.append(
                    {
                        "cls": cls_name,
                        "cls_id": cid,
                        "conf": float(confs[idx]),
                        "box": [float(v) for v in xyxy[idx].tolist()],
                    }
                )

        matches, only_cur, only_ref = match_detections(ref_dets, cur_dets)
        mean_iou = sum(m[0] for m in matches) / len(matches) if matches else 0.0
        mean_dconf = sum(m[1] for m in matches) / len(matches) if matches else 0.0

        all_ious.extend(m[0] for m in matches)
        all_dconfs.extend(m[1] for m in matches)
        tot_ref += len(ref_dets)
        tot_cur += len(cur_dets)
        tot_matched += len(matches)
        tot_only_cur += only_cur
        tot_only_ref += only_ref

        rows.append(
            {
                "image": fname,
                "status": "ok",
                "ref_count_025": len(ref_dets),
                "linux_count_025": len(cur_dets),
                "count_diff": len(cur_dets) - len(ref_dets),
                "matched": len(matches),
                "only_linux": only_cur,
                "only_ref": only_ref,
                "mean_iou": f"{mean_iou:.6f}",
                "mean_abs_dconf": f"{mean_dconf:.6f}",
            }
        )

    sum_iou = sum(all_ious) / len(all_ious) if all_ious else 0.0
    sum_dconf = sum(all_dconfs) / len(all_dconfs) if all_dconfs else 0.0
    rows.append(
        {
            "image": "SUMMARY",
            "status": "ok" if tot_ref > 0 else "empty",
            "ref_count_025": tot_ref,
            "linux_count_025": tot_cur,
            "count_diff": tot_cur - tot_ref,
            "matched": tot_matched,
            "only_linux": tot_only_cur,
            "only_ref": tot_only_ref,
            "mean_iou": f"{sum_iou:.6f}",
            "mean_abs_dconf": f"{sum_dconf:.6f}",
        }
    )

    write_csv(args.out, rows)
    print(
        f"[linux-parity] ref={tot_ref} linux={tot_cur} diff={tot_cur - tot_ref:+d} "
        f"matched={tot_matched} mean_iou={sum_iou:.4f} mean_|dconf|={sum_dconf:.6f} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
