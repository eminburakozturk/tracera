"""Val seti (last.pt @1024, IoU=0.5) için conf 0.05–0.95 (0.01 adım) + 0.395 PR eğrisi üretir."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plot_confusion import (  # noqa: E402
    build_confusion_matrix,
    collect_validator_batches,
    metrics_from_matrix,
)

WEIGHTS_PATH = REPO_ROOT / "runs" / "detect" / "g_y11s_1024_full" / "weights" / "last.pt"
VAL_YAML_CANDIDATES = (
    REPO_ROOT / "runs" / "data_eval_val_last_1024.yaml",
    REPO_ROOT / "configs" / "data_grouped.yaml",
)
OPERATING_POINT_CSV = REPO_ROOT / "results" / "operating_point.csv"
OUT_PATH = REPO_ROOT / "web" / "public" / "data" / "pr_curve_val.json"
OPERATING_CONF = 0.395
IOU_THRESHOLD = 0.50


def resolve_val_yaml() -> Path:
    for cand in VAL_YAML_CANDIDATES:
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"Val dataset YAML bulunamadı: {VAL_YAML_CANDIDATES}")


def generate_pr_curve_val(out_path: Path = OUT_PATH) -> dict[str, Any]:
    val_yaml = resolve_val_yaml()
    assert WEIGHTS_PATH.is_file(), f"Model ağırlığı bulunamadı: {WEIGHTS_PATH}"
    assert OPERATING_POINT_CSV.is_file(), f"operating_point.csv bulunamadı: {OPERATING_POINT_CSV}"

    op_df = pd.read_csv(OPERATING_POINT_CSV)
    val_row = op_df[op_df["split"] == "val"].iloc[0]

    records, _ = collect_validator_batches(WEIGHTS_PATH, val_yaml, "val", imgsz=1024)

    grid = [round(0.05 + i * 0.01, 2) for i in range(91)]  # 0.05 .. 0.95
    thresholds = sorted(set(grid) | {OPERATING_CONF})

    points: list[dict[str, Any]] = []
    op_point_data: dict[str, Any] | None = None

    for conf in thresholds:
        mat = build_confusion_matrix(records, conf_thres=float(conf), iou_thres=IOU_THRESHOLD)
        m = metrics_from_matrix(mat)
        is_op = abs(float(conf) - OPERATING_CONF) < 1e-6
        row = {
            "conf": float(conf),
            "TP": int(m["TP"]),
            "FP": int(m["FP"]),
            "FN": int(m["FN"]),
            "P": float(m["P"]),
            "R": float(m["R"]),
            "F1": float(m["F1"]),
            "is_operating_point": is_op,
        }
        points.append(row)
        if is_op:
            op_point_data = row

    assert op_point_data is not None, "0.395 çalışma noktası üretilmedi."
    assert op_point_data["TP"] == int(val_row["TP"]), (
        f"TP uyumsuz: {op_point_data['TP']} != {int(val_row['TP'])}"
    )
    assert op_point_data["FP"] == int(val_row["FP"]), (
        f"FP uyumsuz: {op_point_data['FP']} != {int(val_row['FP'])}"
    )
    assert op_point_data["FN"] == int(val_row["FN"]), (
        f"FN uyumsuz: {op_point_data['FN']} != {int(val_row['FN'])}"
    )
    assert op_point_data["P"] == float(val_row["P"]), (
        f"P uyumsuz: {op_point_data['P']} != {float(val_row['P'])}"
    )
    assert op_point_data["R"] == float(val_row["R"]), (
        f"R uyumsuz: {op_point_data['R']} != {float(val_row['R'])}"
    )
    assert op_point_data["F1"] == float(val_row["F1"]), (
        f"F1 uyumsuz: {op_point_data['F1']} != {float(val_row['F1'])}"
    )

    payload: dict[str, Any] = {
        "model": "yolo11s",
        "weights": "runs/detect/g_y11s_1024_full/weights/last.pt",
        "split": "val",
        "imgsz": 1024,
        "iou_threshold": IOU_THRESHOLD,
        "operating_conf": OPERATING_CONF,
        "operating_point": {
            "conf": OPERATING_CONF,
            "TP": op_point_data["TP"],
            "FP": op_point_data["FP"],
            "FN": op_point_data["FN"],
            "P": op_point_data["P"],
            "R": op_point_data["R"],
            "F1": op_point_data["F1"],
        },
        "points": points,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"[OK] {out_path} yazıldı ({out_path.stat().st_size} bayt, {len(points)} nokta). "
        f"conf=0.395 -> P={op_point_data['P']:.4f}, R={op_point_data['R']:.4f}, F1={op_point_data['F1']:.4f} "
        f"(TP={op_point_data['TP']}, FP={op_point_data['FP']}, FN={op_point_data['FN']})"
    )
    return payload


if __name__ == "__main__":
    generate_pr_curve_val()
