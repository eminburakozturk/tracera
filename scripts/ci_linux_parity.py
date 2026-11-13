"""CI Linux OpenVINO INT8 parite kontrol scripti (yalniz bilgilendirme / raporlama)."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_core_config():
    """tracera_core/__init__.py uzerinden agir/fastapi bagimliliklarini tetiklemeden config.py yukler."""
    cfg_path = REPO_ROOT / "tracera_core" / "config.py"
    spec = importlib.util.spec_from_file_location("_tracera_core_config_only", cfg_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"tracera_core/config.py yuklenemedi: {cfg_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_CFG = _load_core_config()
DEFAULT_CONF: float = _CFG.DEFAULT_CONF
DEFAULT_IOU_NMS: float = _CFG.DEFAULT_IOU_NMS
IMGSZ: int = _CFG.IMGSZ
OPENVINO_MODEL_PATH: Path = _CFG.OPENVINO_MODEL_PATH
ensure_weights_from_hf = _CFG.ensure_weights_from_hf

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
        "ref_count_0395",
        "linux_count_0395",
        "count_diff_0395",
        "matched",
        "only_linux",
        "only_ref",
        "mean_iou",
        "mean_abs_dconf",
        "max_abs_dconf",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_md(
    summary_path: Path | None,
    *,
    tested_images: int,
    total_samples: int,
    skipped_images: list[str],
    mean_iou: float,
    mean_dconf: float,
    max_dconf: float,
    tot_ref_0395: int,
    tot_cur_0395: int,
    status_note: str,
) -> None:
    if summary_path is None:
        return
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    diff_0395 = tot_cur_0395 - tot_ref_0395
    skip_note = ", ".join(skipped_images) if skipped_images else "yok"
    lines = [
        "## Linux OpenVINO INT8 Parity (Informational)",
        "",
        f"- **Durum:** `{status_note}`",
        f"- **Görüntü sayısı:** {tested_images} / {total_samples} (atlanan: {skip_note})",
        f"- **Ort. |Δconf| (0.25):** {mean_dconf:.6f}",
        f"- **Maks. |Δconf| (0.25):** {max_dconf:.6f}",
        f"- **Ort. IoU (0.25):** {mean_iou:.4f}",
        f"- **Tespit sayısı farkı (0.395'te):** {tot_ref_0395} → {tot_cur_0395} ({diff_0395:+d})",
        "",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Linux OpenVINO INT8 parite kontrolu (bilgilendirme)")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "linux_parity.csv",
        help="Cikti CSV yolu",
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=None,
        help="Opsiyonel Markdown ozet cikti yolu ($GITHUB_STEP_SUMMARY icin)",
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
                    "ref_count_0395": 0,
                    "linux_count_0395": 0,
                    "count_diff_0395": 0,
                    "matched": 0,
                    "only_linux": 0,
                    "only_ref": 0,
                    "mean_iou": "",
                    "mean_abs_dconf": "",
                    "max_abs_dconf": "",
                }
            ],
        )
        write_summary_md(
            args.summary_md,
            tested_images=0,
            total_samples=0,
            skipped_images=[],
            mean_iou=0.0,
            mean_dconf=0.0,
            max_dconf=0.0,
            tot_ref_0395=0,
            tot_cur_0395=0,
            status_note="skipped_no_weights",
        )
        return 0

    from ultralytics import YOLO  # lazy import so skipped environments don't need ultralytics

    ov = YOLO(str(OPENVINO_MODEL_PATH), task="detect")
    samples: list[str] = json.loads(SAMPLES_JSON.read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    all_ious: list[float] = []
    all_dconfs: list[float] = []
    skipped_images: list[str] = []
    tested_images = 0
    tot_ref = 0
    tot_cur = 0
    tot_ref_0395 = 0
    tot_cur_0395 = 0
    tot_matched = 0
    tot_only_cur = 0
    tot_only_ref = 0

    for fname in samples:
        stem = Path(fname).stem
        ref_path = REF_DIR / f"{stem}__openvino-int8_c025.json"
        ref_0395_path = REF_DIR / f"{stem}__openvino-int8.json"
        img_path = resolve_val_image(fname)
        if not ref_path.is_file() or img_path is None:
            skipped_images.append(fname)
            rows.append(
                {
                    "image": fname,
                    "status": "skipped_missing_file",
                    "ref_count_025": "",
                    "linux_count_025": "",
                    "count_diff": "",
                    "ref_count_0395": "",
                    "linux_count_0395": "",
                    "count_diff_0395": "",
                    "matched": "",
                    "only_linux": "",
                    "only_ref": "",
                    "mean_iou": "",
                    "mean_abs_dconf": "",
                    "max_abs_dconf": "",
                }
            )
            continue

        tested_images += 1
        ref_data = json.loads(ref_path.read_text(encoding="utf-8"))
        ref_dets = ref_data.get("detections", [])

        if ref_0395_path.is_file():
            ref_0395_dets = json.loads(ref_0395_path.read_text(encoding="utf-8")).get("detections", [])
        else:
            ref_0395_dets = [d for d in ref_dets if float(d["conf"]) >= DEFAULT_CONF]

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

        cur_0395_dets = [d for d in cur_dets if float(d["conf"]) >= DEFAULT_CONF]

        matches, only_cur, only_ref = match_detections(ref_dets, cur_dets)
        mean_iou = sum(m[0] for m in matches) / len(matches) if matches else 0.0
        mean_dconf = sum(m[1] for m in matches) / len(matches) if matches else 0.0
        max_dconf = max((m[1] for m in matches), default=0.0)

        all_ious.extend(m[0] for m in matches)
        all_dconfs.extend(m[1] for m in matches)
        tot_ref += len(ref_dets)
        tot_cur += len(cur_dets)
        tot_ref_0395 += len(ref_0395_dets)
        tot_cur_0395 += len(cur_0395_dets)
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
                "ref_count_0395": len(ref_0395_dets),
                "linux_count_0395": len(cur_0395_dets),
                "count_diff_0395": len(cur_0395_dets) - len(ref_0395_dets),
                "matched": len(matches),
                "only_linux": only_cur,
                "only_ref": only_ref,
                "mean_iou": f"{mean_iou:.6f}",
                "mean_abs_dconf": f"{mean_dconf:.6f}",
                "max_abs_dconf": f"{max_dconf:.6f}",
            }
        )

    sum_iou = sum(all_ious) / len(all_ious) if all_ious else 0.0
    sum_dconf = sum(all_dconfs) / len(all_dconfs) if all_dconfs else 0.0
    sum_max_dconf = max(all_dconfs, default=0.0)
    summary_status = f"ok_{tested_images}_of_{len(samples)}" if tot_ref > 0 else "empty"
    rows.append(
        {
            "image": "SUMMARY",
            "status": summary_status,
            "ref_count_025": tot_ref,
            "linux_count_025": tot_cur,
            "count_diff": tot_cur - tot_ref,
            "ref_count_0395": tot_ref_0395,
            "linux_count_0395": tot_cur_0395,
            "count_diff_0395": tot_cur_0395 - tot_ref_0395,
            "matched": tot_matched,
            "only_linux": tot_only_cur,
            "only_ref": tot_only_ref,
            "mean_iou": f"{sum_iou:.6f}",
            "mean_abs_dconf": f"{sum_dconf:.6f}",
            "max_abs_dconf": f"{sum_max_dconf:.6f}",
        }
    )

    write_csv(args.out, rows)
    write_summary_md(
        args.summary_md,
        tested_images=tested_images,
        total_samples=len(samples),
        skipped_images=skipped_images,
        mean_iou=sum_iou,
        mean_dconf=sum_dconf,
        max_dconf=sum_max_dconf,
        tot_ref_0395=tot_ref_0395,
        tot_cur_0395=tot_cur_0395,
        status_note=summary_status,
    )
    print(
        f"[linux-parity] images={tested_images}/{len(samples)} "
        f"ref_025={tot_ref} linux_025={tot_cur} diff_025={tot_cur - tot_ref:+d} "
        f"diff_0395={tot_cur_0395 - tot_ref_0395:+d} matched={tot_matched} "
        f"mean_iou={sum_iou:.4f} mean_|dconf|={sum_dconf:.6f} max_|dconf|={sum_max_dconf:.6f} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
