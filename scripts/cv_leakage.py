"""Rastgele (StratifiedKFold) vs Kart Bazlı (GroupKFold) 5 katlı CV ile veri sızıntısı karşılaştırması."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import GroupKFold, StratifiedKFold
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


def load_cv_pool(meta_csv: Path) -> pd.DataFrame:
    """meta.csv'den sadece train ve val split'indeki 623 görüntüyü yükler (test hariç)."""
    df = pd.read_csv(meta_csv, dtype={"kart_no": str})
    df_pool = df[df["split"].isin(["train", "val"])].copy()
    assert "test" not in df_pool["split"].values, "Test seti CV havuzuna dahil edilemez!"
    assert len(df_pool) == 623, f"Beklenen 623 görüntü, bulunan: {len(df_pool)}"

    # Her görüntünün mevcut mutlak yolunu oluştur ve doğrula
    img_paths: list[str] = []
    for _, row in df_pool.iterrows():
        p = (REPO_ROOT / "data" / "yolo" / "random" / str(row["split"]) / "images" / str(row["dosya"])).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Görüntü bulunamadı: {p}")
        img_paths.append(p.as_posix())

    df_pool["abs_path"] = img_paths
    df_pool = df_pool.sort_values(by=["sınıf", "dosya"]).reset_index(drop=True)
    return df_pool


def write_fold_files(
    cv_dir: Path,
    schema: str,
    fold_idx: int,
    train_paths: list[str],
    val_paths: list[str],
) -> Path:
    """Kat için train.txt, val.txt ve yaml dosyalarını data/yolo/cv/ altına yazar."""
    cv_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{schema}_fold{fold_idx}"

    train_txt = (cv_dir / f"{prefix}_train.txt").resolve()
    val_txt = (cv_dir / f"{prefix}_val.txt").resolve()
    yaml_path = (cv_dir / f"{prefix}.yaml").resolve()

    train_txt.write_text("\n".join(train_paths) + "\n", encoding="utf-8")
    val_txt.write_text("\n".join(val_paths) + "\n", encoding="utf-8")

    cfg = {
        "path": cv_dir.resolve().as_posix(),
        "train": train_txt.as_posix(),
        "val": val_txt.as_posix(),
        "nc": len(CLASSES),
        "names": {i: name for i, name in enumerate(CLASSES)},
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    return yaml_path


def get_model_path() -> str:
    candidate = REPO_ROOT / "weights" / "yolo11s.pt"
    if candidate.is_file():
        return str(candidate.resolve())
    return "yolo11s.pt"


def build_folds(df_pool: pd.DataFrame) -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    """Şema A (random StratifiedKFold) ve Şema B (grouped GroupKFold) katlarını üretir."""
    folds: list[tuple[str, int, np.ndarray, np.ndarray]] = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for k, (tr_idx, va_idx) in enumerate(skf.split(df_pool, df_pool["sınıf"]), start=1):
        folds.append(("random", k, tr_idx, va_idx))

    gkf = GroupKFold(n_splits=5)
    print("\n=== GROUPED CV VAL KARTLARI VE KESIŞIM KONTROLÜ ===")
    for k, (tr_idx, va_idx) in enumerate(
        gkf.split(df_pool, df_pool["sınıf"], groups=df_pool["kart_no"]), start=1
    ):
        train_cards = sorted(df_pool.iloc[tr_idx]["kart_no"].unique().tolist())
        val_cards = sorted(df_pool.iloc[va_idx]["kart_no"].unique().tolist())
        overlap = set(train_cards).intersection(set(val_cards))
        assert not overlap, f"Grouped fold {k} içinde kart sızıntısı var: {overlap}"
        print(
            f"grouped_fold{k}: val_kartları={val_cards} "
            f"(n_train={len(tr_idx)}, n_val={len(va_idx)}) | kesişim={overlap or 'BOŞ (OK)'}"
        )
        folds.append(("grouped", k, tr_idx, va_idx))

    return folds


def print_summary(df_res: pd.DataFrame) -> None:
    """Her şema için ortalama ± std özet tablosunu yazdırır."""
    metric_cols = ["mAP50", "mAP50-95"] + CLASSES
    print("\n" + "=" * 78)
    print("5 KATLI CV SIZINTI KARŞILAŞTIRMASI ÖZETİ (Ortalama ± Std)")
    print("=" * 78)
    for schema in ["random", "grouped"]:
        sub = df_res[df_res["şema"] == schema]
        if sub.empty:
            continue
        print(f"\nŞema: {schema.upper()} (n={len(sub)} kat)")
        print("-" * 50)
        for col in metric_cols:
            mean_v = sub[col].mean()
            std_v = sub[col].std(ddof=1)
            print(f"  {col:<18}: {mean_v:.4f} ± {std_v:.4f}")
    print("=" * 78)


def main() -> None:
    meta_csv = REPO_ROOT / "data" / "yolo" / "meta.csv"
    cv_dir = REPO_ROOT / "data" / "yolo" / "cv"
    results_csv = REPO_ROOT / "results" / "cv_leakage.csv"
    project_dir = (REPO_ROOT / "runs" / "cv").resolve()
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    df_pool = load_cv_pool(meta_csv)
    folds = build_folds(df_pool)

    completed: set[tuple[str, int]] = set()
    rows: list[dict[str, object]] = []
    if results_csv.is_file():
        df_existing = pd.read_csv(results_csv, dtype={"val_kartları": str})
        rows = df_existing.to_dict(orient="records")
        for r in rows:
            completed.add((str(r["şema"]), int(r["kat"])))
        print(f"\nMevcut {results_csv.name} bulundu: {len(completed)}/10 kat tamamlanmış.")

    model_path = get_model_path()

    for schema, k, tr_idx, va_idx in folds:
        run_name = f"{schema}_fold{k}"
        df_tr = df_pool.iloc[tr_idx]
        df_va = df_pool.iloc[va_idx]

        train_cards = set(df_tr["kart_no"].unique())
        val_cards = sorted(df_va["kart_no"].unique().tolist())
        if schema == "grouped":
            assert train_cards.isdisjoint(set(val_cards)), (
                f"Grouped kat {k} train/val kart kesişimi boş değil!"
            )

        yaml_path = write_fold_files(
            cv_dir=cv_dir,
            schema=schema,
            fold_idx=k,
            train_paths=df_tr["abs_path"].tolist(),
            val_paths=df_va["abs_path"].tolist(),
        )

        if (schema, k) in completed:
            print(f"[ATLANDI] {run_name} zaten {results_csv.name} içinde mevcut.")
            continue

        print(f"\n>>> EĞİTİM BAŞLIYOR: {run_name} (n_train={len(df_tr)}, n_val={len(df_va)}, val_kartları={val_cards})")
        model = YOLO(model_path)
        model.train(
            data=str(yaml_path),
            imgsz=640,
            epochs=60,
            batch=16,
            seed=42,
            deterministic=True,
            cache="ram",
            workers=4,
            patience=30,
            project=str(project_dir),
            name=run_name,
            exist_ok=True,
        )

        best_pt = project_dir / run_name / "weights" / "best.pt"
        if not best_pt.is_file():
            raise FileNotFoundError(f"best.pt bulunamadı: {best_pt}")

        print(f"\n>>> VAL DEĞERLENDİRMESİ (FP32): {run_name}")
        best_model = YOLO(str(best_pt))
        metrics = best_model.val(
            data=str(yaml_path),
            split="val",
            imgsz=640,
            batch=16,
            workers=4,
            project=str(project_dir),
            name=f"{run_name}_val",
            exist_ok=True,
        )

        class_ap50_95: dict[str, float] = {cls_name: float("nan") for cls_name in CLASSES}
        for i, cls_idx in enumerate(metrics.box.ap_class_index):
            cls_name = CLASSES[int(cls_idx)]
            _, _, _, ap_val = metrics.box.class_result(i)
            class_ap50_95[cls_name] = round(float(ap_val), 4)

        row: dict[str, object] = {
            "şema": schema,
            "kat": k,
            "val_kartları": ";".join(val_cards),
            "n_train": len(df_tr),
            "n_val": len(df_va),
            "mAP50": round(float(metrics.box.map50), 4),
            "mAP50-95": round(float(metrics.box.map), 4),
        }
        for cls_name in CLASSES:
            row[cls_name] = class_ap50_95[cls_name]

        rows.append(row)
        df_out = pd.DataFrame(rows)
        df_out.to_csv(results_csv, index=False, encoding="utf-8")
        print(f"[KAYDEDİLDİ] {run_name} -> mAP50={row['mAP50']:.4f}, mAP50-95={row['mAP50-95']:.4f}")

        del model, best_model, metrics
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df_final = pd.read_csv(results_csv, dtype={"val_kartları": str})
    print("\n=== TÜM KATLAR TABLOSU (results/cv_leakage.csv) ===")
    print(df_final.to_string(index=False))
    print_summary(df_final)


if __name__ == "__main__":
    main()
