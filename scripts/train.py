"""PKU-Market-PCB veri seti üzerinde Ultralytics YOLO eğitimi ve doğrulama (val) scripti.

Not: Early stopping varsayılan kapalı (patience=0).
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import pandas as pd
import torch
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


def resolve_data_yaml(data_yaml_path: Path, run_name: str) -> tuple[Path, str]:
    """data.yaml içindeki 'path' alanını repo köküne göre mutlak yola çevirip runs/ altına yazar."""
    if not data_yaml_path.is_absolute():
        data_yaml_path = (REPO_ROOT / data_yaml_path).resolve()

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_path = Path(cfg.get("path", "data/yolo/random"))
    split_scheme = "grouped" if "grouped" in str(raw_path).lower() else "random"

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

    return out_yaml, split_scheme


def get_best_epoch(run_dir: Path) -> int:
    """Eğitim results.csv dosyasından en iyi fitness epoch numarasını bulur."""
    res_csv = run_dir / "results.csv"
    if not res_csv.is_file():
        return -1
    df = pd.read_csv(res_csv)
    df.columns = df.columns.str.strip()
    fit = 0.1 * df["metrics/mAP50(B)"] + 0.9 * df["metrics/mAP50-95(B)"]
    return int(df.loc[fit.idxmax(), "epoch"])


def append_experiment_row(row: dict[str, object]) -> None:
    """Deney sonucunu results/experiments.csv dosyasına ekler veya günceller."""
    exp_csv = REPO_ROOT / "results" / "experiments.csv"
    exp_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    if exp_csv.is_file():
        df_old = pd.read_csv(exp_csv)
        rows = [r for r in df_old.to_dict(orient="records") if str(r.get("deney")) != str(row["deney"])]

    rows.append(row)
    df_new = pd.DataFrame(rows)
    df_new.to_csv(exp_csv, index=False, encoding="utf-8")
    print(f"[KAYDEDİLDİ] {exp_csv}: {row['deney']}")


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
        "--patience",
        type=int,
        default=0,
        help="Erken durdurma sabrı — early stopping varsayılan kapalı (varsayılan: 0)",
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

    resolved_yaml, split_scheme = resolve_data_yaml(args.data, args.name)
    project_dir = (REPO_ROOT / "runs" / "detect").resolve()

    weights_dir = REPO_ROOT / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    model_arg = args.model
    if not Path(model_arg).is_absolute() and not Path(model_arg).exists():
        candidate = weights_dir / model_arg
        if candidate.exists():
            model_arg = str(candidate)

    actual_batch = args.batch
    while True:
        print(
            f"Çözümlenmiş veri dosyası: {resolved_yaml} (split={split_scheme})\n"
            f"Model: {args.model} | imgsz: {args.imgsz} | epochs: {args.epochs} | batch: {actual_batch}"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        try:
            model = YOLO(model_arg)
            model.train(
                data=str(resolved_yaml),
                imgsz=args.imgsz,
                epochs=args.epochs,
                batch=actual_batch,
                project=str(project_dir),
                name=args.name,
                exist_ok=True,
                seed=args.seed,
                deterministic=True,
                cache="ram",
                workers=4,
                patience=args.patience,
            )
            train_duration_min = round((time.perf_counter() - t0) / 60.0, 2)
            peak_vram_gb = (
                round(torch.cuda.max_memory_reserved() / (1024**3), 2)
                if torch.cuda.is_available()
                else 0.0
            )
            break
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" in str(exc).lower() and actual_batch > 1:
                new_batch = max(1, actual_batch // 2)
                print(f"[UYARI] CUDA OOM (batch={actual_batch}). batch={new_batch} ile tekrar deneniyor...")
                actual_batch = new_batch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                raise

    run_dir = project_dir / args.name
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Eğitim sonunda best.pt bulunamadı: {best_pt}")

    best_epoch = get_best_epoch(run_dir)

    print(f"\n=== VAL SETİ DEĞERLENDİRMESİ ({best_pt}) ===")
    best_model = YOLO(str(best_pt))
    metrics = best_model.val(
        data=str(resolved_yaml),
        split="val",
        imgsz=args.imgsz,
        batch=actual_batch,
        workers=4,
        project=str(project_dir),
        name=f"{args.name}_val",
        exist_ok=True,
    )

    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    names = metrics.names

    print("\n" + "=" * 60)
    print(f"VAL SONUÇLARI — {args.name}")
    print("=" * 60)
    print(f"Genel mAP50    : {map50:.4f}")
    print(f"Genel mAP50-95 : {map50_95:.4f}")
    print(f"Best Epoch     : {best_epoch} | Süre: {train_duration_min:.2f} dk | Tepe VRAM: {peak_vram_gb:.2f} GB")
    print("-" * 60)
    print(f"{'Sınıf':<20} {'Precision':>10} {'Recall':>10} {'mAP50':>10} {'mAP50-95':>10}")
    print("-" * 60)

    class_ap50_95: dict[str, float] = {cls_name: float("nan") for cls_name in CLASSES}
    for i, cls_idx in enumerate(metrics.box.ap_class_index):
        cls_name = names[int(cls_idx)]
        p_val, r_val, ap50_val, ap_val = metrics.box.class_result(i)
        class_ap50_95[cls_name] = round(float(ap_val), 4)
        print(f"{cls_name:<20} {p_val:>10.4f} {r_val:>10.4f} {ap50_val:>10.4f} {ap_val:>10.4f}")
    print("=" * 60)

    exp_row: dict[str, object] = {
        "deney": args.name,
        "split": split_scheme,
        "model": Path(args.model).stem,
        "imgsz": args.imgsz,
        "batch": actual_batch,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "mAP50": round(map50, 4),
        "mAP50-95": round(map50_95, 4),
    }
    for cls_name in CLASSES:
        exp_row[cls_name] = class_ap50_95[cls_name]
    exp_row["eğitim_süresi_dk"] = train_duration_min
    exp_row["gpu_bellek_tepe_gb"] = peak_vram_gb

    append_experiment_row(exp_row)


if __name__ == "__main__":
    main()
