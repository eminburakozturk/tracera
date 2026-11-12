"""PCB-AoI (KubeEdge Ianvs) transfer öğrenme deneyi:
COCO (yolo11s.pt) vs PKU ön eğitimli (runs/detect/g_y11s_1024_full/weights/last.pt) modelleri
configs/data_aoi.yaml üzerinde 3 farklı seed (0, 1, 2) ile 100 epoch eğitilir,
her eğitimin last.pt'si ile hem VAL (bilgi amaçlı) hem TEST seti değerlendirilir.
Sonuçlar results/aoi_transfer.csv'ye kaydedilir.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent

CLASSES = ["Bad_qiaojiao", "Bad_podu"]

EXPERIMENTS = [
    {
        "baslangic": "coco",
        "weights": "yolo11s.pt",
        "label": "COCO (yolo11s.pt)",
    },
    {
        "baslangic": "pku",
        "weights": "runs/detect/g_y11s_1024_full/weights/last.pt",
        "label": "PKU (g_y11s_1024_full)",
    },
]


def resolve_data_yaml(data_yaml_path: Path, run_name: str) -> Path:
    """data_aoi.yaml içindeki path alanını repo köküne göre mutlak yola çevirip runs/ altına yazar."""
    if not data_yaml_path.is_absolute():
        data_yaml_path = (REPO_ROOT / data_yaml_path).resolve()

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_path = Path(cfg.get("path", "data/yolo/aoi"))
    if not raw_path.is_absolute():
        abs_dataset_path = (REPO_ROOT / raw_path).resolve()
    else:
        abs_dataset_path = raw_path.resolve()

    cfg["path"] = abs_dataset_path.as_posix()

    runs_dir = REPO_ROOT / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_yaml = runs_dir / f"data_resolved_aoi_{run_name}.yaml"

    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    return out_yaml


def evaluate_model(
    weights_path: Path,
    data_yaml_path: Path,
    split: str,
    imgsz: int = 640,
    project_dir: Path | None = None,
    run_name: str = "eval",
) -> dict[str, float]:
    """Modelin belirtilen split (val veya test) üzerindeki FP32 mAP sonuçlarını döndürür."""
    model = YOLO(str(weights_path))
    val_args = {
        "data": str(data_yaml_path),
        "split": split,
        "imgsz": imgsz,
        "batch": 1,
        "device": "0",
        "workers": 2,
        "verbose": False,
        "half": False,  # FP32
    }
    if project_dir is not None:
        val_args["project"] = str(project_dir)
        val_args["name"] = run_name
        val_args["exist_ok"] = True

    metrics = model.val(**val_args)

    res: dict[str, float] = {
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50-95": round(float(metrics.box.map), 4),
    }

    # Sınıf bazlı metrikler
    for cls_name in CLASSES:
        res[f"{cls_name}_mAP50"] = float("nan")
        res[f"{cls_name}_mAP50-95"] = float("nan")

    for i, cls_idx in enumerate(metrics.box.ap_class_index):
        c_name = metrics.names[int(cls_idx)]
        res[f"{c_name}_mAP50"] = round(float(metrics.box.ap50[i]), 4)
        res[f"{c_name}_mAP50-95"] = round(float(metrics.box.ap[i]), 4)

    return res


def run_aoi_transfer(
    seeds: list[int],
    epochs: int = 100,
    batch: int = 16,
    imgsz: int = 640,
    patience: int = 0,
    data_yaml: Path = Path("configs/data_aoi.yaml"),
    out_csv: Path = Path("results/aoi_transfer.csv"),
) -> pd.DataFrame:
    """6 eğitimi yürütür ve val + test metriklerini kaydeder."""
    project_dir = (REPO_ROOT / "runs" / "aoi_transfer").resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []

    for exp in EXPERIMENTS:
        base_key = exp["baslangic"]
        base_weights = exp["weights"]
        if not Path(base_weights).is_absolute():
            base_weights = (REPO_ROOT / base_weights).resolve()

        if not Path(base_weights).is_file():
            # yolo11s.pt weights klasöründe olabilir
            alt = REPO_ROOT / "weights" / Path(base_weights).name
            if alt.is_file():
                base_weights = alt
            else:
                raise FileNotFoundError(f"Ağırlık bulunamadı: {base_weights}")

        for seed in seeds:
            run_name = f"aoi_{base_key}_seed{seed}"
            print("\n" + "=" * 72)
            print(f">>> EĞİTİM BAŞLIYOR: {run_name} (Başlangıç: {exp['label']}, Seed: {seed})")
            print("=" * 72)

            resolved_yaml = resolve_data_yaml(data_yaml, run_name)

            model = YOLO(str(base_weights))
            t0 = time.perf_counter()
            model.train(
                data=str(resolved_yaml),
                imgsz=imgsz,
                epochs=epochs,
                batch=batch,
                patience=patience,
                seed=seed,
                deterministic=True,
                cache="ram",
                workers=4,
                project=str(project_dir),
                name=run_name,
                exist_ok=True,
                device="0",
            )
            train_duration_min = round((time.perf_counter() - t0) / 60.0, 2)

            run_dir = project_dir / run_name
            last_pt = run_dir / "weights" / "last.pt"
            best_pt = run_dir / "weights" / "best.pt"

            target_weights = last_pt if last_pt.is_file() else best_pt

            # 1. VAL Değerlendirmesi
            print(f">>> Val Değerlendirmesi ({run_name}) ...")
            val_metrics = evaluate_model(
                weights_path=target_weights,
                data_yaml_path=resolved_yaml,
                split="val",
                imgsz=imgsz,
                project_dir=project_dir,
                run_name=f"{run_name}_val",
            )

            row_val = {
                "baslangic": base_key,
                "model_kaynak": exp["label"],
                "seed": seed,
                "split": "val",
                "weights": "last.pt",
                "epochs": epochs,
                "batch": batch,
                "imgsz": imgsz,
                "train_sure_dk": train_duration_min,
                **val_metrics,
            }
            rows.append(row_val)

            # 2. TEST Değerlendirmesi
            print(f">>> TEST Değerlendirmesi ({run_name}) ...")
            test_metrics = evaluate_model(
                weights_path=target_weights,
                data_yaml_path=resolved_yaml,
                split="test",
                imgsz=imgsz,
                project_dir=project_dir,
                run_name=f"{run_name}_test",
            )

            row_test = {
                "baslangic": base_key,
                "model_kaynak": exp["label"],
                "seed": seed,
                "split": "test",
                "weights": "last.pt",
                "epochs": epochs,
                "batch": batch,
                "imgsz": imgsz,
                "train_sure_dk": train_duration_min,
                **test_metrics,
            }
            rows.append(row_test)

    df_out = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\n[KAYDEDİLDİ] Transfer öğrenme sonuçları: {out_csv} (Toplam {len(df_out)} satır)")

    return df_out


def print_summary(df: pd.DataFrame) -> None:
    """Başlangıç başına ortalama ± std ve seed eşleştirmeli fark tablosunu yazdırır."""
    print("\n" + "=" * 80)
    print("PCB-AoI TRANSFER ÖĞRENME SONUÇLARI (VAL & TEST)")
    print("=" * 80)

    for split in ["val", "test"]:
        sub = df[df["split"] == split].copy()
        print(f"\n--- [SPLIT: {split.upper()}] ---")
        disp_cols = [
            "baslangic",
            "seed",
            "mAP50",
            "mAP50-95",
            "Bad_qiaojiao_mAP50",
            "Bad_qiaojiao_mAP50-95",
            "Bad_podu_mAP50",
            "Bad_podu_mAP50-95",
        ]
        print(sub[disp_cols].to_string(index=False))

        # Ortalama ± std
        print(f"\nOrtalama ± Std ({split.upper()}):")
        for b_name in ["coco", "pku"]:
            b_sub = sub[sub["baslangic"] == b_name]
            m50_mean, m50_std = b_sub["mAP50"].mean(), b_sub["mAP50"].std()
            m95_mean, m95_std = b_sub["mAP50-95"].mean(), b_sub["mAP50-95"].std()
            q50_mean = b_sub["Bad_qiaojiao_mAP50"].mean()
            p50_mean = b_sub["Bad_podu_mAP50"].mean()
            print(
                f"  {b_name.upper():<6}: mAP50 = {m50_mean:.4f} ± {m50_std:.4f} | "
                f"mAP50-95 = {m95_mean:.4f} ± {m95_std:.4f} | "
                f"qiaojiao_mAP50 = {q50_mean:.4f} | podu_mAP50 = {p50_mean:.4f}"
            )

        # Seed eşleştirmeli fark (PKU - COCO)
        print(f"\nSeed Eşleştirmeli Fark (PKU - COCO) [{split.upper()}]:")
        coco_sub = sub[sub["baslangic"] == "coco"].set_index("seed")
        pku_sub = sub[sub["baslangic"] == "pku"].set_index("seed")

        diffs_50 = []
        diffs_95 = []
        for s in sorted(sub["seed"].unique()):
            diff_50 = pku_sub.loc[s, "mAP50"] - coco_sub.loc[s, "mAP50"]
            diff_95 = pku_sub.loc[s, "mAP50-95"] - coco_sub.loc[s, "mAP50-95"]
            diffs_50.append(diff_50)
            diffs_95.append(diff_95)
            print(
                f"  Seed {s}: ΔmAP50 = {diff_50:+.4f} (PKU {pku_sub.loc[s, 'mAP50']:.4f} vs COCO {coco_sub.loc[s, 'mAP50']:.4f}) | "
                f"ΔmAP50-95 = {diff_95:+.4f} (PKU {pku_sub.loc[s, 'mAP50-95']:.4f} vs COCO {coco_sub.loc[s, 'mAP50-95']:.4f})"
            )

        print(
            f"  Ortalama Fark: ΔmAP50 = {np.mean(diffs_50):+.4f} ± {np.std(diffs_50):.4f} | "
            f"ΔmAP50-95 = {np.mean(diffs_95):+.4f} ± {np.std(diffs_95):.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="PCB-AoI Transfer Öğrenme Deneyi")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="Tohum listesi (varsayılan: 0 1 2)")
    parser.add_argument("--epochs", type=int, default=100, help="Epoch sayısı (varsayılan: 100)")
    parser.add_argument("--batch", type=int, default=16, help="Batch boyutu (varsayılan: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Görüntü boyutu (varsayılan: 640)")
    parser.add_argument("--patience", type=int, default=0, help="Patience (varsayılan: 0)")
    parser.add_argument("--data", type=Path, default=Path("configs/data_aoi.yaml"), help="Veri YAML yolu")
    parser.add_argument("--out", type=Path, default=Path("results/aoi_transfer.csv"), help="Çıktı CSV yolu")
    args = parser.parse_args()

    df = run_aoi_transfer(
        seeds=args.seeds,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        data_yaml=args.data,
        out_csv=args.out,
    )
    print_summary(df)


if __name__ == "__main__":
    main()
