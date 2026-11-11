"""Nihai model değerlendirme scripti (val / test).

FP32, batch=1 ile çalışır; genel ve sınıf bazlı P, R, mAP50, mAP50-95 ile görüntü/kutu
sayılarını CSV'ye yazar ve test değerlendirmesinde confusion matrix + PR eğrisini
results/ altına kaydeder.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent

CLASSES: list[str] = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]


def resolve_data_yaml_and_counts(data_yaml_path: Path, split: str, tag: str) -> tuple[Path, int, int]:
    """data.yaml içindeki 'path' alanını mutlak yola çevirir ve ilgili split'teki görüntü/kutu sayısını hesaplar."""
    if not data_yaml_path.is_absolute():
        data_yaml_path = (REPO_ROOT / data_yaml_path).resolve()

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_path = Path(cfg.get("path", "data/yolo/grouped"))
    abs_dataset_path = raw_path.resolve() if raw_path.is_absolute() else (REPO_ROOT / raw_path).resolve()
    cfg["path"] = abs_dataset_path.as_posix()

    runs_dir = REPO_ROOT / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_yaml = runs_dir / f"data_eval_{tag}.yaml"
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    split_rel = str(cfg.get(split, f"{split}/images"))
    img_dir = (abs_dataset_path / split_rel).resolve()
    lbl_dir = img_dir.parent / "labels"

    img_files = [
        p for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    n_images = len(img_files)
    n_boxes = 0
    for img_p in img_files:
        lbl_p = lbl_dir / f"{img_p.stem}.txt"
        if lbl_p.is_file():
            lines = [ln.strip() for ln in lbl_p.read_text(encoding="utf-8").splitlines() if ln.strip()]
            n_boxes += len(lines)

    return out_yaml, n_images, n_boxes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tracera YOLO Değerlendirme Scripti (FP32, batch=1)")
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Değerlendirilecek model ağırlık dosyası (.pt)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("configs/data_grouped.yaml"),
        help="Veri konfigürasyon dosyası (varsayılan: configs/data_grouped.yaml)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "test"],
        default="val",
        help="Değerlendirilecek veri bölmesi: val veya test (varsayılan: val)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="Görüntü boyutu (varsayılan: 1024)",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Çıktı CSV yolu (varsayılan: results/final_<split>.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    weights_path = args.weights if args.weights.is_absolute() else (REPO_ROOT / args.weights).resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"Ağırlık dosyası bulunamadı: {weights_path}")

    weight_name = weights_path.name
    if args.split == "test":
        assert weight_name == "last.pt", (
            f"Test seti protokol gereği SADECE last.pt ile değerlendirilebilir! Verilen: {weight_name}"
        )

    out_csv = (
        args.out_csv
        if args.out_csv is not None
        else REPO_ROOT / "results" / f"final_{args.split}.csv"
    )
    if not out_csv.is_absolute():
        out_csv = (REPO_ROOT / out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.split == "test" and out_csv.is_file():
        raise RuntimeError(
            f"{out_csv} zaten mevcut! Test seti yalnızca BİR KEZ değerlendirilebilir."
        )

    tag = f"{args.split}_{weights_path.stem}_{args.imgsz}"
    resolved_yaml, n_images, n_boxes = resolve_data_yaml_and_counts(args.data, args.split, tag)
    project_dir = (REPO_ROOT / "runs" / "eval").resolve()

    print(
        f"\n>>> DEĞERLENDİRME BAŞLIYOR: split={args.split} | weights={weight_name} | "
        f"imgsz={args.imgsz} | batch=1 | FP32 | n_images={n_images} | n_boxes={n_boxes}"
    )

    model = YOLO(str(weights_path))
    metrics = model.val(
        data=str(resolved_yaml),
        split=args.split,
        imgsz=args.imgsz,
        batch=1,
        half=False,
        plots=True,
        workers=4,
        project=str(project_dir),
        name=tag,
        exist_ok=True,
    )

    mp = round(float(metrics.box.mp), 4)
    mr = round(float(metrics.box.mr), 4)
    map50 = round(float(metrics.box.map50), 4)
    map50_95 = round(float(metrics.box.map), 4)

    row: dict[str, object] = {
        "ağırlık": weight_name,
        "split": args.split,
        "imgsz": args.imgsz,
        "görüntü_sayısı": n_images,
        "kutu_sayısı": n_boxes,
        "P": mp,
        "R": mr,
        "mAP50": map50,
        "mAP50-95": map50_95,
    }

    names = metrics.names
    print("\n" + "=" * 68)
    print(f"SONUÇLAR — split={args.split} | ağırlık={weight_name} (n={n_images} görüntü, {n_boxes} kutu)")
    print("=" * 68)
    print(f"Genel P: {mp:.4f} | R: {mr:.4f} | mAP50: {map50:.4f} | mAP50-95: {map50_95:.4f}")
    print("-" * 68)
    print(f"{'Sınıf':<20} {'Precision':>10} {'Recall':>10} {'mAP50':>10} {'mAP50-95':>10}")
    print("-" * 68)

    cls_metrics: dict[str, tuple[float, float, float, float]] = {}
    for i, cls_idx in enumerate(metrics.box.ap_class_index):
        cls_name = names[int(cls_idx)]
        p_val, r_val, ap50_val, ap_val = metrics.box.class_result(i)
        cls_metrics[cls_name] = (
            round(float(p_val), 4),
            round(float(r_val), 4),
            round(float(ap50_val), 4),
            round(float(ap_val), 4),
        )
        print(f"{cls_name:<20} {p_val:>10.4f} {r_val:>10.4f} {ap50_val:>10.4f} {ap_val:>10.4f}")
    print("=" * 68)

    for cls_name in CLASSES:
        p_v, r_v, ap50_v, ap_v = cls_metrics.get(
            cls_name, (float("nan"), float("nan"), float("nan"), float("nan"))
        )
        row[f"{cls_name}_P"] = p_v
        row[f"{cls_name}_R"] = r_v
        row[f"{cls_name}_mAP50"] = ap50_v
        row[f"{cls_name}_mAP50-95"] = ap_v

    rows: list[dict[str, object]] = []
    if out_csv.is_file():
        df_old = pd.read_csv(out_csv)
        rows = [
            r for r in df_old.to_dict(orient="records")
            if not (str(r.get("ağırlık")) == weight_name and str(r.get("split")) == args.split)
        ]
    rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[KAYDEDİLDİ] {out_csv}")

    save_dir = Path(metrics.save_dir)
    if args.split == "test":
        cm_src = save_dir / "confusion_matrix.png"
        pr_src = save_dir / "BoxPR_curve.png"
        if not pr_src.is_file():
            pr_src = save_dir / "PR_curve.png"

        cm_dst = REPO_ROOT / "results" / "confusion_matrix_test.png"
        pr_dst = REPO_ROOT / "results" / "pr_curve_test.png"

        if cm_src.is_file():
            shutil.copy2(cm_src, cm_dst)
            print(f"[KAYDEDİLDİ] {cm_dst}")
        else:
            raise FileNotFoundError(f"Confusion matrix bulunamadı: {cm_src}")

        if pr_src.is_file():
            shutil.copy2(pr_src, pr_dst)
            print(f"[KAYDEDİLDİ] {pr_dst}")
        else:
            raise FileNotFoundError(f"PR eğrisi bulunamadı: {pr_src}")


if __name__ == "__main__":
    main()
