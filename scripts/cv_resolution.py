"""640 vs 1024 çözünürlük karşılaştırması — 5 katlı GroupKFold CV (patience=0, early stopping kapalı).

Her kat için hem last.pt hem best.pt FP32 val sonuçları results/cv_resolution.csv dosyasına kaydedilir.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from ultralytics import YOLO

from cv_leakage import CLASSES, get_model_path, load_cv_pool, write_fold_files

REPO_ROOT = Path(__file__).resolve().parent.parent

CONFIGS: list[tuple[int, int]] = [
    (640, 16),
    (1024, 8),
]


def build_and_verify_grouped_folds(
    df_pool: pd.DataFrame,
    leakage_csv: Path,
) -> list[tuple[int, np.ndarray, np.ndarray, str]]:
    """GroupKFold katlarını üretir ve cv_leakage.csv'deki val_kartları ile birebir eşleştiğini doğrular."""
    df_leak = pd.read_csv(leakage_csv, dtype={"val_kartları": str})
    df_leak_grouped = df_leak[df_leak["şema"] == "grouped"].set_index("kat")
    assert len(df_leak_grouped) == 5, "cv_leakage.csv içinde 5 grouped kat bulunamadı!"

    gkf = GroupKFold(n_splits=5)
    folds: list[tuple[int, np.ndarray, np.ndarray, str]] = []

    print("\n=== GROUPKFold KATLARI VE cv_leakage.csv EŞLEŞME KONTROLÜ ===")
    for k, (tr_idx, va_idx) in enumerate(
        gkf.split(df_pool, df_pool["sınıf"], groups=df_pool["kart_no"]), start=1
    ):
        train_cards = set(df_pool.iloc[tr_idx]["kart_no"].unique())
        val_cards = sorted(df_pool.iloc[va_idx]["kart_no"].unique().tolist())
        assert train_cards.isdisjoint(set(val_cards)), f"Kat {k} train/val kart kesişimi boş değil!"

        val_cards_str = ";".join(val_cards)
        expected_str = str(df_leak_grouped.loc[k, "val_kartları"])
        assert val_cards_str == expected_str, (
            f"Kat {k} val_kartları uyuşmazlığı: {val_cards_str} != {expected_str}"
        )
        print(
            f"Kat {k}: val_kartları={val_cards_str} "
            f"(n_train={len(tr_idx)}, n_val={len(va_idx)}) == cv_leakage.csv [OK]"
        )
        folds.append((k, tr_idx, va_idx, val_cards_str))

    return folds


def eval_weight_fp32(
    weight_path: Path,
    yaml_path: Path,
    imgsz: int,
    batch: int,
    project_dir: Path,
    val_run_name: str,
) -> tuple[float, float, dict[str, float]]:
    """Verilen ağırlık (last.pt / best.pt) ile val split üzerinde FP32 değerlendirme yapar."""
    model = YOLO(str(weight_path))
    metrics = model.val(
        data=str(yaml_path),
        split="val",
        imgsz=imgsz,
        batch=batch,
        workers=4,
        project=str(project_dir),
        name=val_run_name,
        exist_ok=True,
    )
    map50 = round(float(metrics.box.map50), 4)
    map50_95 = round(float(metrics.box.map), 4)

    class_ap50_95: dict[str, float] = {cls_name: float("nan") for cls_name in CLASSES}
    for i, cls_idx in enumerate(metrics.box.ap_class_index):
        cls_name = CLASSES[int(cls_idx)]
        _, _, _, ap_val = metrics.box.class_result(i)
        class_ap50_95[cls_name] = round(float(ap_val), 4)

    del model, metrics
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return map50, map50_95, class_ap50_95


def print_summary(df_res: pd.DataFrame) -> None:
    """Her imgsz için last.pt (ve best.pt) ortalama ± std ve kat bazında 1024 - 640 farkını yazdırır."""
    metric_cols = ["mAP50", "mAP50-95"] + CLASSES

    for weight_type in ["last", "best"]:
        print("\n" + "=" * 82)
        print(f"5 KATLI GROUPED CV ÇÖZÜNÜRLÜK ÖZETİ — Ağırlık: {weight_type}.pt (Ortalama ± Std)")
        print("=" * 82)
        for imgsz in [640, 1024]:
            sub = df_res[(df_res["imgsz"] == imgsz) & (df_res["ağırlık"] == weight_type)]
            if sub.empty:
                continue
            print(f"\nimgsz={imgsz} ({weight_type}.pt, n={len(sub)} kat, ort. süre={sub['süre_dk'].mean():.2f} dk)")
            print("-" * 54)
            for col in metric_cols:
                print(f"  {col:<18}: {sub[col].mean():.4f} ± {sub[col].std(ddof=1):.4f}")

        sub_640 = (
            df_res[(df_res["imgsz"] == 640) & (df_res["ağırlık"] == weight_type)]
            .set_index("kat")
            .sort_index()
        )
        sub_1024 = (
            df_res[(df_res["imgsz"] == 1024) & (df_res["ağırlık"] == weight_type)]
            .set_index("kat")
            .sort_index()
        )
        if len(sub_640) == 5 and len(sub_1024) == 5:
            print(f"\n--- KAT BAZINDA FARK (1024 - 640) [{weight_type}.pt] ---")
            diff_rows: list[dict[str, object]] = []
            for k in range(1, 6):
                r640 = sub_640.loc[k]
                r1024 = sub_1024.loc[k]
                diff_rows.append(
                    {
                        "kat": k,
                        "val_kartları": r640["val_kartları"],
                        "640_mAP50": f"{r640['mAP50']:.4f}",
                        "1024_mAP50": f"{r1024['mAP50']:.4f}",
                        "Δ_mAP50": f"{r1024['mAP50'] - r640['mAP50']:+.4f}",
                        "640_mAP50-95": f"{r640['mAP50-95']:.4f}",
                        "1024_mAP50-95": f"{r1024['mAP50-95']:.4f}",
                        "Δ_mAP50-95": f"{r1024['mAP50-95'] - r640['mAP50-95']:+.4f}",
                    }
                )
            print(pd.DataFrame(diff_rows).to_string(index=False))
    print("=" * 82)


def main() -> None:
    meta_csv = REPO_ROOT / "data" / "yolo" / "meta.csv"
    leakage_csv = REPO_ROOT / "results" / "cv_leakage.csv"
    cv_dir = REPO_ROOT / "data" / "yolo" / "cv"
    results_csv = REPO_ROOT / "results" / "cv_resolution.csv"
    project_dir = (REPO_ROOT / "runs" / "cv_res").resolve()
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    df_pool = load_cv_pool(meta_csv)
    folds = build_and_verify_grouped_folds(df_pool, leakage_csv)

    rows: list[dict[str, object]] = []
    completed: set[tuple[int, int]] = set()
    if results_csv.is_file():
        df_existing = pd.read_csv(results_csv, dtype={"val_kartları": str})
        rows = df_existing.to_dict(orient="records")
        for (im_sz, k), grp in df_existing.groupby(["imgsz", "kat"]):
            if set(grp["ağırlık"].tolist()) == {"last", "best"}:
                completed.add((int(im_sz), int(k)))
        print(f"\nMevcut {results_csv.name} bulundu: {len(completed)}/10 eğitim tamamlanmış.")

    model_path = get_model_path()

    for imgsz, batch in CONFIGS:
        for k, tr_idx, va_idx, val_cards_str in folds:
            run_name = f"grouped_{imgsz}_fold{k}"
            df_tr = df_pool.iloc[tr_idx]
            df_va = df_pool.iloc[va_idx]

            yaml_path = write_fold_files(
                cv_dir=cv_dir,
                schema="grouped",
                fold_idx=k,
                train_paths=df_tr["abs_path"].tolist(),
                val_paths=df_va["abs_path"].tolist(),
            )

            if (imgsz, k) in completed:
                print(f"[ATLANDI] {run_name} zaten {results_csv.name} içinde mevcut.")
                continue

            # Yarım kalmış satır varsa temizle
            rows = [r for r in rows if not (int(r["imgsz"]) == imgsz and int(r["kat"]) == k)]

            print(
                f"\n>>> EĞİTİM BAŞLIYOR: {run_name} "
                f"(imgsz={imgsz}, batch={batch}, epochs=80, patience=0, val_kartları={val_cards_str})"
            )
            t0 = time.perf_counter()
            model = YOLO(model_path)
            model.train(
                data=str(yaml_path),
                imgsz=imgsz,
                epochs=80,
                batch=batch,
                seed=42,
                deterministic=True,
                cache="ram",
                workers=4,
                patience=0,
                project=str(project_dir),
                name=run_name,
                exist_ok=True,
            )
            duration_min = round((time.perf_counter() - t0) / 60.0, 2)

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            run_dir = project_dir / run_name
            run_res_csv = run_dir / "results.csv"
            assert run_res_csv.is_file(), f"results.csv bulunamadı: {run_res_csv}"
            df_epochs = pd.read_csv(run_res_csv)
            assert len(df_epochs) == 80, (
                f"{run_name} 80 epoch tamamlanmadı! Satır sayısı: {len(df_epochs)}"
            )

            for weight_type in ("last", "best"):
                w_path = run_dir / "weights" / f"{weight_type}.pt"
                assert w_path.is_file(), f"Ağırlık bulunamadı: {w_path}"
                print(f"\n>>> VAL DEĞERLENDİRMESİ (FP32) [{weight_type}.pt]: {run_name}")
                map50, map50_95, cls_scores = eval_weight_fp32(
                    weight_path=w_path,
                    yaml_path=yaml_path,
                    imgsz=imgsz,
                    batch=batch,
                    project_dir=project_dir,
                    val_run_name=f"{run_name}_val_{weight_type}",
                )

                row: dict[str, object] = {
                    "imgsz": imgsz,
                    "kat": k,
                    "val_kartları": val_cards_str,
                    "ağırlık": weight_type,
                    "mAP50": map50,
                    "mAP50-95": map50_95,
                }
                for cls_name in CLASSES:
                    row[cls_name] = cls_scores[cls_name]
                row["süre_dk"] = duration_min
                rows.append(row)
                print(
                    f"[SONUÇ] {run_name} ({weight_type}.pt) -> "
                    f"mAP50={map50:.4f}, mAP50-95={map50_95:.4f}"
                )

            pd.DataFrame(rows).to_csv(results_csv, index=False, encoding="utf-8")
            print(f"[KAYDEDİLDİ] {results_csv.name} ({len(rows)}/20 satır)")

    df_final = pd.read_csv(results_csv, dtype={"val_kartları": str})
    assert len(df_final) == 20, f"Beklenen 20 satır, bulunan: {len(df_final)}"
    print("\n=== TÜM SONUÇLAR TABLOSU (results/cv_resolution.csv) ===")
    print(df_final.to_string(index=False))
    print_summary(df_final)


if __name__ == "__main__":
    main()
