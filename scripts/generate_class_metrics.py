"""operating_point.csv ve cv_resolution.csv dosyalarından web/public/data/class_metrics.json üretir."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CLASSES: list[str] = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]

OPERATING_POINT_CSV = REPO_ROOT / "results" / "operating_point.csv"
CV_RESOLUTION_CSV = REPO_ROOT / "results" / "cv_resolution.csv"
OUT_PATH = REPO_ROOT / "web" / "public" / "data" / "class_metrics.json"


def _extract_split_class_metrics(row: pd.Series, cls: str) -> dict[str, Any]:
    return {
        "TP": int(row[f"{cls}_TP"]),
        "FP": int(row[f"{cls}_FP"]),
        "FN": int(row[f"{cls}_FN"]),
        "P": round(float(row[f"{cls}_P"]), 4),
        "R": round(float(row[f"{cls}_R"]), 4),
        "F1": round(float(row[f"{cls}_F1"]), 4),
    }


def generate_class_metrics(out_path: Path = OUT_PATH) -> dict[str, Any]:
    assert OPERATING_POINT_CSV.is_file(), f"operating_point.csv bulunamadı: {OPERATING_POINT_CSV}"
    assert CV_RESOLUTION_CSV.is_file(), f"cv_resolution.csv bulunamadı: {CV_RESOLUTION_CSV}"

    op_df = pd.read_csv(OPERATING_POINT_CSV)
    val_row = op_df[op_df["split"] == "val"].iloc[0]
    test_row = op_df[op_df["split"] == "test"].iloc[0]
    excl_df = op_df[op_df["split"] == "test_sablon_noktalari_haric"]
    excl_row = excl_df.iloc[0] if len(excl_df) > 0 else None

    cv_df = pd.read_csv(CV_RESOLUTION_CSV)
    cv_1024_last = cv_df[(cv_df["imgsz"] == 1024) & (cv_df["ağırlık"] == "last")].sort_values("kat")
    assert len(cv_1024_last) == 5, f"5 kat bekleniyordu, bulunan: {len(cv_1024_last)}"

    classes_out: list[dict[str, Any]] = []
    for cls in CLASSES:
        vals = cv_1024_last[cls].astype(float)
        cls_entry: dict[str, Any] = {
            "class": cls,
            "val": _extract_split_class_metrics(val_row, cls),
            "test": _extract_split_class_metrics(test_row, cls),
            "cv_1024_last": {
                "folds": [round(float(v), 4) for v in vals.tolist()],
                "mAP50_95_mean": round(float(vals.mean()), 4),
                "mAP50_95_std": round(float(vals.std(ddof=1)), 4),
            },
        }
        if excl_row is not None:
            cls_entry["test_excluding_template"] = _extract_split_class_metrics(excl_row, cls)
        classes_out.append(cls_entry)

    payload: dict[str, Any] = {
        "conf_threshold": float(val_row["conf_threshold"]),
        "iou_threshold": float(val_row["iou_threshold"]),
        "cv_source": "results/cv_resolution.csv (imgsz=1024, last.pt, 5-fold grouped CV, ddof=1)",
        "operating_point_source": "results/operating_point.csv",
        "overall": {
            "val": {
                "P": round(float(val_row["P"]), 4),
                "R": round(float(val_row["R"]), 4),
                "F1": round(float(val_row["F1"]), 4),
            },
            "test": {
                "P": round(float(test_row["P"]), 4),
                "R": round(float(test_row["R"]), 4),
                "F1": round(float(test_row["F1"]), 4),
            },
            "cv_1024_last": {
                "mAP50_mean": round(float(cv_1024_last["mAP50"].mean()), 4),
                "mAP50_std": round(float(cv_1024_last["mAP50"].std(ddof=1)), 4),
                "mAP50_95_mean": round(float(cv_1024_last["mAP50-95"].mean()), 4),
                "mAP50_95_std": round(float(cv_1024_last["mAP50-95"].std(ddof=1)), 4),
            },
        },
        "classes": classes_out,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] {out_path} yazıldı ({out_path.stat().st_size} bayt)")
    for c in classes_out:
        cv_info = c["cv_1024_last"]
        print(
            f"  {c['class']:<16} val F1={c['val']['F1']:.4f} | test F1={c['test']['F1']:.4f} | "
            f"CV mAP50-95={cv_info['mAP50_95_mean']:.4f} ± {cv_info['mAP50_95_std']:.4f}"
        )
    return payload


if __name__ == "__main__":
    generate_class_metrics()
