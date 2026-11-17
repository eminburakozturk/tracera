"""Val ve kilitli Test seti tespitlerini güven aralıklarına göre gruplayıp reliability_val.json ve reliability_test.json üretir."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.error_analysis import (  # noqa: E402
    CONF_THRESHOLD,
    IOU_THRESHOLD,
    analyze_split_detections,
    collect_split_predictions,
)
from scripts.generate_pr_curve_val import (  # noqa: E402
    OPERATING_POINT_CSV,
    WEIGHTS_PATH,
    resolve_val_yaml,
)

OUT_VAL_PATH = REPO_ROOT / "web" / "public" / "data" / "reliability_val.json"
OUT_TEST_PATH = REPO_ROOT / "web" / "public" / "data" / "reliability_test.json"
OUT_PATH = OUT_VAL_PATH

TEST_YAML_CANDIDATES = (
    REPO_ROOT / "runs" / "data_eval_test_last_1024.yaml",
    REPO_ROOT / "configs" / "data_grouped.yaml",
)

BINS: list[tuple[str, float, float, bool]] = [
    ("0.395-0.5", 0.395, 0.5, False),
    ("0.5-0.6", 0.5, 0.6, False),
    ("0.6-0.7", 0.6, 0.7, False),
    ("0.7-0.8", 0.7, 0.8, False),
    ("0.8-0.9", 0.8, 0.9, False),
    ("0.9-1.0", 0.9, 1.0, True),
]


def resolve_test_yaml() -> Path:
    for cand in TEST_YAML_CANDIDATES:
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"Test dataset YAML bulunamadı: {TEST_YAML_CANDIDATES}")


def generate_reliability_split(split: str, out_path: Path) -> dict[str, Any]:
    data_yaml = resolve_val_yaml() if split == "val" else resolve_test_yaml()
    assert WEIGHTS_PATH.is_file(), f"Model ağırlığı bulunamadı: {WEIGHTS_PATH}"
    assert OPERATING_POINT_CSV.is_file(), f"operating_point.csv bulunamadı: {OPERATING_POINT_CSV}"

    op_df = pd.read_csv(OPERATING_POINT_CSV)
    split_row = op_df[op_df["split"] == split].iloc[0]

    items = collect_split_predictions(WEIGHTS_PATH, data_yaml, split, imgsz=1024)
    tp_rows, fp_rows = analyze_split_detections(
        items,
        split=split,
        conf_thres=CONF_THRESHOLD,
        iou_thres=IOU_THRESHOLD,
    )

    assert len(tp_rows) == int(split_row["TP"]), f"{split} TP uyumsuz: {len(tp_rows)} != {int(split_row['TP'])}"
    assert len(fp_rows) == int(split_row["FP"]), f"{split} FP uyumsuz: {len(fp_rows)} != {int(split_row['FP'])}"
    if split == "test":
        assert len(tp_rows) == 416 and len(fp_rows) == 142, f"Test TP/FP 416/142 değil: {len(tp_rows)}/{len(fp_rows)}"

    all_dets = [
        {"conf": float(r["güven_raw"]), "is_tp": True}
        for r in tp_rows
    ] + [
        {"conf": float(r["güven_raw"]), "is_tp": False}
        for r in fp_rows
    ]

    bins_out: list[dict[str, Any]] = []
    assigned = 0
    for label, lo, hi, inclusive_hi in BINS:
        if inclusive_hi:
            subset = [d for d in all_dets if lo <= d["conf"] <= hi]
        else:
            subset = [d for d in all_dets if lo <= d["conf"] < hi]
        n = len(subset)
        tp = sum(1 for d in subset if d["is_tp"])
        fp = n - tp
        assigned += n
        precision = round(tp / n, 4) if n > 0 else None
        mean_conf = round(sum(d["conf"] for d in subset) / n, 4) if n > 0 else None
        bins_out.append(
            {
                "bin": label,
                "min": lo,
                "max": hi,
                "n": n,
                "TP": tp,
                "FP": fp,
                "precision": precision,
                "mean_conf": mean_conf,
            }
        )

    assert assigned == len(all_dets), f"Bazı tespitler aralığa düşmedi: {assigned} != {len(all_dets)}"
    assert sum(b["TP"] for b in bins_out) == int(split_row["TP"])
    assert sum(b["FP"] for b in bins_out) == int(split_row["FP"])

    payload: dict[str, Any] = {
        "model": "yolo11s",
        "weights": "runs/detect/g_y11s_1024_full/weights/last.pt",
        "split": split,
        "imgsz": 1024,
        "conf_threshold": CONF_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "total_detections": len(all_dets),
        "total_TP": len(tp_rows),
        "total_FP": len(fp_rows),
        "overall_precision": round(len(tp_rows) / len(all_dets), 4),
        "bins": bins_out,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] {out_path} yazıldı ({out_path.stat().st_size} bayt)")
    for b in bins_out:
        print(f"  {b['bin']:<10} n={b['n']:>3} TP={b['TP']:>3} FP={b['FP']:>3} precision={b['precision']} mean_conf={b['mean_conf']}")
    return payload


def generate_reliability_val(out_path: Path = OUT_VAL_PATH) -> dict[str, Any]:
    return generate_reliability_split("val", out_path)


def generate_reliability_test(out_path: Path = OUT_TEST_PATH) -> dict[str, Any]:
    return generate_reliability_split("test", out_path)


if __name__ == "__main__":
    generate_reliability_val()
    generate_reliability_test()
