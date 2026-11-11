"""PKU-Market-PCB veri seti üzerinde Ultralytics YOLO eğitimi ve doğrulama (val) scripti."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_data_yaml(data_yaml_path: Path, run_name: str) -> Path:
    """data.yaml içindeki 'path' alanını repo köküne göre mutlak yola çevirip runs/ altına yazar."""
    if not data_yaml_path.is_absolute():
        data_yaml_path = (REPO_ROOT / data_yaml_path).resolve()

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_path = Path(cfg.get("path", "data/yolo/random"))
    if not raw_path.is_absolute():
        abs_dataset_path = (REPO_ROOT / raw_path).resolve()
    else:
        abs_dataset_path = raw_path.resolve()

    cfg["path"] = abs_dataset_path.as_posix()

    runs_dir = REPO_ROOT / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_yaml = runs_dir / f"data_resolved_{run_name}.yaml"

    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    return out_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tracera PCB YOLO Eğitim Scripti")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Veri konfigürasyon dosyası (varsayılan: configs/data.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11s.pt",
        help="Başlangıç model ağırlığı (varsayılan: yolo11s.pt)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Görüntü boyutu (varsayılan: 640)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Eğitim epoch sayısı (varsayılan: 100)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch boyutu (varsayılan: 16)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="baseline_y11s_640",
        help="Deney / çalışma adı (varsayılan: baseline_y11s_640)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Rastgelelik tohumu (varsayılan: 42)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    resolved_yaml = resolve_data_yaml(args.data, args.name)
    project_dir = (REPO_ROOT / "runs" / "detect").resolve()

    print(f"Çözümlenmiş veri dosyası: {resolved_yaml}")
    print(f"Model: {args.model} | imgsz: {args.imgsz} | epochs: {args.epochs} | batch: {args.batch}")

    # Başlangıç ağırlığını weights/ altında sakla veya doğrudan yükle
    weights_dir = REPO_ROOT / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    model_arg = args.model
    if not Path(model_arg).is_absolute() and not Path(model_arg).exists():
        candidate = weights_dir / model_arg
        if candidate.exists():
            model_arg = str(candidate)

    model = YOLO(model_arg)

    model.train(
        data=str(resolved_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        project=str(project_dir),
        name=args.name,
        exist_ok=True,
        seed=args.seed,
        deterministic=True,
        cache="ram",
        workers=4,
        patience=30,
    )

    best_pt = project_dir / args.name / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Eğitim sonunda best.pt bulunamadı: {best_pt}")

    print(f"\n=== VAL SETİ DEĞERLENDİRMESİ ({best_pt}) ===")
    best_model = YOLO(str(best_pt))
    metrics = best_model.val(
        data=str(resolved_yaml),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        workers=4,
    )

    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    names = metrics.names

    print("\n" + "=" * 60)
    print(f"VAL SONUÇLARI — {args.name}")
    print("=" * 60)
    print(f"Genel mAP50    : {map50:.4f}")
    print(f"Genel mAP50-95 : {map50_95:.4f}")
    print("-" * 60)
    print(f"{'Sınıf':<20} {'Precision':>10} {'Recall':>10} {'mAP50':>10} {'mAP50-95':>10}")
    print("-" * 60)

    for i, cls_idx in enumerate(metrics.box.ap_class_index):
        cls_name = names[int(cls_idx)]
        p_val, r_val, ap50_val, ap_val = metrics.box.class_result(i)
        print(f"{cls_name:<20} {p_val:>10.4f} {r_val:>10.4f} {ap50_val:>10.4f} {ap_val:>10.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
