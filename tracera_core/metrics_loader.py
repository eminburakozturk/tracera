"""results/*.csv dosyalarından CONTRACT §3 /metrics özet JSON'unu üretir."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from tracera_core.config import RESULTS_DIR


def _clean_value(val: Any) -> Any:
    if pd.isna(val):
        return None
    if isinstance(val, (float, int)) and not isinstance(val, bool):
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        if isinstance(val, int):
            return int(val)
        return round(f, 6)
    if hasattr(val, "item"):
        return _clean_value(val.item())
    return val


def _read_csv_records(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.is_file():
        return []
    df = pd.read_csv(csv_path)
    records: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        records.append({str(k): _clean_value(v) for k, v in row.items()})
    return records


def _summarize_cv(results_dir: Path) -> dict[str, Any]:
    leakage_rows = _read_csv_records(results_dir / "cv_leakage.csv")
    resolution_rows = _read_csv_records(results_dir / "cv_resolution.csv")

    leakage_summary: dict[str, Any] = {}
    if leakage_rows:
        df_l = pd.DataFrame(leakage_rows)
        for schema in ("random", "grouped"):
            sub = df_l[df_l["şema"] == schema]
            if not sub.empty:
                leakage_summary[schema] = {
                    "n_folds": int(len(sub)),
                    "mAP50_mean": round(float(sub["mAP50"].mean()), 4),
                    "mAP50_std": round(float(sub["mAP50"].std(ddof=1)), 4),
                    "mAP50_95_mean": round(float(sub["mAP50-95"].mean()), 4),
                    "mAP50_95_std": round(float(sub["mAP50-95"].std(ddof=1)), 4),
                }

    resolution_summary: dict[str, Any] = {}
    if resolution_rows:
        df_r = pd.DataFrame(resolution_rows)
        for (imgsz, weight), sub in df_r.groupby(["imgsz", "ağırlık"], sort=False):
            key = f"{int(imgsz)}_{weight}"
            resolution_summary[key] = {
                "imgsz": int(imgsz),
                "weight": str(weight),
                "n_folds": int(len(sub)),
                "mAP50_mean": round(float(sub["mAP50"].mean()), 4),
                "mAP50_std": round(float(sub["mAP50"].std(ddof=1)), 4),
                "mAP50_95_mean": round(float(sub["mAP50-95"].mean()), 4),
                "mAP50_95_std": round(float(sub["mAP50-95"].std(ddof=1)), 4),
            }

    return {
        "leakage_rows": leakage_rows,
        "leakage_summary": leakage_summary,
        "resolution_rows": resolution_rows,
        "resolution_summary": resolution_summary,
    }


def _summarize_aoi(results_dir: Path) -> dict[str, Any]:
    rows = _read_csv_records(results_dir / "aoi_transfer.csv")
    summary: dict[str, Any] = {}
    if rows:
        df = pd.DataFrame(rows)
        df_test = df[df["split"] == "test"]
        for init_name in ("coco", "pku"):
            sub = df_test[df_test["baslangic"] == init_name]
            if not sub.empty:
                summary[init_name] = {
                    "n_seeds": int(len(sub)),
                    "mAP50_mean": round(float(sub["mAP50"].mean()), 4),
                    "mAP50_std": round(float(sub["mAP50"].std(ddof=1)), 4),
                    "mAP50_95_mean": round(float(sub["mAP50-95"].mean()), 4),
                    "mAP50_95_std": round(float(sub["mAP50-95"].std(ddof=1)), 4),
                    "Bad_qiaojiao_mAP50_mean": round(float(sub["Bad_qiaojiao_mAP50"].mean()), 4),
                    "Bad_podu_mAP50_mean": round(float(sub["Bad_podu_mAP50"].mean()), 4),
                }
    return {
        "rows": rows,
        "test_summary": summary,
    }


def build_metrics_payload(results_dir: Path | None = None) -> dict[str, Any]:
    """CONTRACT §3 /metrics ve web/public/metrics.json için tek kaynak özet sözlüğü üretir."""
    target_dir = results_dir or RESULTS_DIR
    xai_rows = _read_csv_records(target_dir / "gradcam_sanity.csv")
    xai_1024 = [r for r in xai_rows if r.get("model") == "g_y11s_1024_full"]

    return {
        "operating_point": _read_csv_records(target_dir / "operating_point.csv"),
        "edge_benchmark": _read_csv_records(target_dir / "edge_benchmark.csv"),
        "cv": _summarize_cv(target_dir),
        "xai": xai_1024 if xai_1024 else xai_rows,
        "aoi": _summarize_aoi(target_dir),
        "final_val": _read_csv_records(target_dir / "final_val.csv"),
        "final_test": _read_csv_records(target_dir / "final_test.csv"),
        "web_spike": _read_csv_records(target_dir / "web_spike.csv"),
    }
