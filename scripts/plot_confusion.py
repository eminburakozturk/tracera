"""Val setinde F1-maksimum güven eşiğini (operating point) bulur ve test confusion matrix'ini yeniden çizer.

- final_test.csv dosyasına DOKUNMAZ ve mAP hesaplamaz.
- Mevcut confusion_matrix_test.png'nin hangi conf eşiğiyle çizildiğini birebir doğrular.
- results/operating_point.csv ve iki panelli (Sayı + Satır-Normalize) results/confusion_matrix_test.png üretir.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.metrics import ConfusionMatrix

REPO_ROOT = Path(__file__).resolve().parent.parent

CLASSES: list[str] = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]


@torch.no_grad()
def collect_validator_batches(
    weights_path: Path,
    data_yaml: Path,
    split: str,
    imgsz: int = 1024,
) -> tuple[list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]], float]:
    """DetectionValidator'ın kendi dataloader, preprocess ve postprocess adımlarını kullanarak
    mAP hesaplamadan (predn, pbatch) çiftlerini ve validator'ın varsayılan args.conf değerini döndürür.
    """
    args = get_cfg(
        overrides={
            "model": str(weights_path),
            "data": str(data_yaml),
            "split": split,
            "imgsz": imgsz,
            "batch": 1,
            "rect": True,
            "half": False,
            "plots": False,
            "workers": 0,
            "verbose": False,
        }
    )
    validator = DetectionValidator(args=args)
    validator.training = False
    validator.data = check_det_dataset(args.data)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = AutoBackend(
        model=str(weights_path),
        device=device,
        dnn=False,
        data=args.data,
        fp16=False,
    )
    model.eval()
    validator.device = device
    validator.stride = model.stride
    validator.names = model.names
    val_path = validator.data.get(split)
    dataloader = validator.get_dataloader(val_path, batch_size=1)
    validator.dataloader = dataloader
    validator.init_metrics(model)

    default_conf = float(validator.args.conf or 0.001)
    records: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]] = []

    for batch in dataloader:
        batch = validator.preprocess(batch)
        preds = model(batch["img"])
        preds = validator.postprocess(preds)
        for si, pred in enumerate(preds):
            pbatch = validator._prepare_batch(si, batch)
            predn = validator._prepare_pred(pred)
            records.append(
                (
                    {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in predn.items()},
                    {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in pbatch.items()},
                )
            )

    return records, default_conf


def build_confusion_matrix(
    records: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]],
    conf_thres: float,
    iou_thres: float = 0.5,
) -> np.ndarray:
    """Ultralytics ConfusionMatrix.process_batch ile verilen conf ve iou_thres için (7x7) matrisi üretir."""
    cm = ConfusionMatrix(names={i: c for i, c in enumerate(CLASSES)}, task="detect")
    for predn, pbatch in records:
        cm.process_batch(predn, pbatch, conf=conf_thres, iou_thres=iou_thres)
    return cm.matrix.copy()


def metrics_from_matrix(matrix: np.ndarray) -> dict[str, object]:
    """(nc+1, nc+1) matristen (satır=Tahmin, sütun=Gerçek) genel ve sınıf bazlı TP, FP, FN, P, R, F1 hesaplar."""
    nc = len(CLASSES)
    tp_per_cls = np.diag(matrix)[:nc]
    fp_per_cls = matrix[:nc, :].sum(axis=1) - tp_per_cls
    bg_fp_per_cls = matrix[:nc, nc]
    fn_per_cls = matrix[:, :nc].sum(axis=0) - tp_per_cls
    bg_fn_per_cls = matrix[nc, :nc]

    p_per_cls = np.divide(
        tp_per_cls,
        tp_per_cls + fp_per_cls,
        out=np.zeros_like(tp_per_cls, dtype=np.float64),
        where=(tp_per_cls + fp_per_cls) > 0,
    )
    r_per_cls = np.divide(
        tp_per_cls,
        tp_per_cls + fn_per_cls,
        out=np.zeros_like(tp_per_cls, dtype=np.float64),
        where=(tp_per_cls + fn_per_cls) > 0,
    )
    f1_per_cls = np.divide(
        2.0 * p_per_cls * r_per_cls,
        p_per_cls + r_per_cls,
        out=np.zeros_like(p_per_cls, dtype=np.float64),
        where=(p_per_cls + r_per_cls) > 0,
    )

    tp_total = int(tp_per_cls.sum())
    fp_total = int(fp_per_cls.sum())
    bg_fp_total = int(bg_fp_per_cls.sum())
    cross_cls_fp = int(matrix[:nc, :nc].sum() - tp_per_cls.sum())
    fn_total = int(fn_per_cls.sum())
    bg_fn_total = int(bg_fn_per_cls.sum())

    p_micro = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    r_micro = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1_micro = 2.0 * p_micro * r_micro / (p_micro + r_micro) if (p_micro + r_micro) > 0 else 0.0

    p_macro = float(p_per_cls.mean())
    r_macro = float(r_per_cls.mean())
    f1_macro = float(f1_per_cls.mean())

    out: dict[str, object] = {
        "TP": tp_total,
        "FP": fp_total,
        "background_FP": bg_fp_total,
        "cross_class_FP": cross_cls_fp,
        "FN": fn_total,
        "background_FN": bg_fn_total,
        "P": round(p_micro, 4),
        "R": round(r_micro, 4),
        "F1": round(f1_micro, 4),
        "P_macro": round(p_macro, 4),
        "R_macro": round(r_macro, 4),
        "F1_macro": round(f1_macro, 4),
    }

    for i, cls_name in enumerate(CLASSES):
        out[f"{cls_name}_TP"] = int(tp_per_cls[i])
        out[f"{cls_name}_FP"] = int(fp_per_cls[i])
        out[f"{cls_name}_bg_FP"] = int(bg_fp_per_cls[i])
        out[f"{cls_name}_FN"] = int(fn_per_cls[i])
        out[f"{cls_name}_P"] = round(float(p_per_cls[i]), 4)
        out[f"{cls_name}_R"] = round(float(r_per_cls[i]), 4)
        out[f"{cls_name}_F1"] = round(float(f1_per_cls[i]), 4)

    return out


def plot_two_panel_confusion_matrix(
    matrix: np.ndarray,
    conf_thres: float,
    test_metrics: dict[str, object],
    save_path: Path,
) -> None:
    """Sayı ve Satır-Normalize iki panelli confusion matrix figürünü çizer ve kaydeder."""
    labels = CLASSES + ["background"]
    n = len(labels)

    row_sums = matrix.sum(axis=1, keepdims=True)
    norm_matrix = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_sums > 0,
    )

    fig, axes = plt.subplots(1, 2, figsize=(18, 7.8))

    # Panel 1: Ham Sayılar (Counts)
    ax0 = axes[0]
    im0 = ax0.imshow(matrix, cmap="Blues", interpolation="nearest")
    ax0.set_title(
        f"Test Confusion Matrix — Sayı (Adet)\n(conf = {conf_thres:.3f}, IoU = 0.50)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax0.set_xticks(range(n))
    ax0.set_yticks(range(n))
    ax0.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax0.set_yticklabels(labels, fontsize=10)
    ax0.set_xlabel("Gerçek (True)", fontsize=11, fontweight="bold")
    ax0.set_ylabel("Tahmin (Predicted)", fontsize=11, fontweight="bold")

    thresh0 = matrix.max() / 2.0 if matrix.max() > 0 else 1.0
    for i in range(n):
        for j in range(n):
            val = int(round(matrix[i, j]))
            if val > 0 or (i == j and i < len(CLASSES)):
                color = "white" if matrix[i, j] > thresh0 else "#1a1a1a"
                ax0.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=10, fontweight="bold")
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    # Panel 2: Satır-Normalize (Predicted Row Normalized)
    ax1 = axes[1]
    im1 = ax1.imshow(norm_matrix, cmap="Blues", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax1.set_title(
        f"Test Confusion Matrix — Satır-Normalize (Tahmin Satırı = Precision)\n"
        f"(Genel P={test_metrics['P']:.4f}, R={test_metrics['R']:.4f}, F1={test_metrics['F1']:.4f})",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax1.set_yticklabels(labels, fontsize=10)
    ax1.set_xlabel("Gerçek (True)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Tahmin (Predicted)", fontsize=11, fontweight="bold")

    for i in range(n):
        for j in range(n):
            val = norm_matrix[i, j]
            if val >= 0.005 or (i == j and i < len(CLASSES)):
                color = "white" if val > 0.5 else "#1a1a1a"
                ax1.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=10, fontweight="bold")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"PKU-Market-PCB Test Seti (n=93 Görüntü, 469 GT Kutu) — Val F1-Maks Eşiği (conf={conf_thres:.3f})",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    final_test_csv = REPO_ROOT / "results" / "final_test.csv"
    assert final_test_csv.is_file(), f"final_test.csv bulunamadı: {final_test_csv}"
    hash_before = hashlib.sha256(final_test_csv.read_bytes()).hexdigest()

    weights_path = REPO_ROOT / "runs" / "detect" / "g_y11s_1024_full" / "weights" / "last.pt"
    resolved_val_yaml = REPO_ROOT / "runs" / "data_eval_val_last_1024.yaml"
    resolved_test_yaml = REPO_ROOT / "runs" / "data_eval_test_last_1024.yaml"
    op_csv = REPO_ROOT / "results" / "operating_point.csv"
    cm_png = REPO_ROOT / "results" / "confusion_matrix_test.png"

    print("1) Val ve Test üzerinde DetectionValidator ile birebir aynı tahminler (FP32, batch=1) alınıyor...")
    val_records, val_default_conf = collect_validator_batches(weights_path, resolved_val_yaml, "val", imgsz=1024)
    test_records, test_default_conf = collect_validator_batches(weights_path, resolved_test_yaml, "test", imgsz=1024)
    print(f"   Val : {len(val_records)} görüntü | Test: {len(test_records)} görüntü")
    print(f"   DetectionValidator varsayılan args.conf = {test_default_conf}")

    # Mevcut yanlış confusion_matrix_test.png'nin hangi conf ile çizildiğini birebir doğrula
    cm_old = build_confusion_matrix(test_records, conf_thres=test_default_conf, iou_thres=0.5)
    bg_fp_old = [int(x) for x in cm_old[:6, 6]]
    tp_old = [int(cm_old[i, i]) for i in range(6)]
    print("\n=== MEVCUT (ESKİ) MATRİSİN CONF EŞİĞİ TEŞHİSİ ===")
    print(
        f"Ultralytics DetectionValidator.update_metrics() içinde:\n"
        f"  self.confusion_matrix.process_batch(predn, pbatch, conf=self.args.conf)\n"
        f"çağrılır. model.val() sırasında mAP hesabı için self.args.conf = {test_default_conf} olduğundan,\n"
        f"ConfusionMatrix.process_batch varsayılan conf=0.25 yerine conf={test_default_conf} ile çalışmıştır."
    )
    print(f"conf={test_default_conf} ile Test köşegen (TP): {tp_old} (Toplam = {sum(tp_old)})")
    print(f"conf={test_default_conf} ile Test background-FP sütunu: {bg_fp_old} (Toplam = {sum(bg_fp_old)})")
    assert tp_old == [74, 75, 73, 75, 68, 77], f"Beklenen TP [74, 75, 73, 75, 68, 77], bulunan: {tp_old}"

    # Val üzerinde F1'i maksimize eden güven eşiğini bul (tüm sınıflar ortak)
    thresholds = np.round(np.arange(0.05, 0.951, 0.005), 3)
    best_conf = 0.25
    best_val_f1 = -1.0
    best_val_metrics: dict[str, object] = {}

    for t in thresholds:
        mat_v = build_confusion_matrix(val_records, conf_thres=float(t), iou_thres=0.5)
        m_v = metrics_from_matrix(mat_v)
        f1_v = float(m_v["F1"])
        if f1_v > best_val_f1:
            best_val_f1 = f1_v
            best_conf = float(t)
            best_val_metrics = m_v

    print("\n=== VAL SETİNDE SEÇİLEN ÇALIŞMA NOKTASI (OPERATING POINT) ===")
    print(
        f"Optimal conf eşiği : {best_conf:.3f}\n"
        f"Val Micro P/R/F1   : P={best_val_metrics['P']:.4f} | R={best_val_metrics['R']:.4f} | F1={best_val_metrics['F1']:.4f}\n"
        f"Val Macro P/R/F1   : P={best_val_metrics['P_macro']:.4f} | R={best_val_metrics['R_macro']:.4f} | F1={best_val_metrics['F1_macro']:.4f}\n"
        f"Val TP/FP/FN       : TP={best_val_metrics['TP']}, FP={best_val_metrics['FP']} (bg_FP={best_val_metrics['background_FP']}), FN={best_val_metrics['FN']}"
    )

    # Aynı eşikle Test confusion matrix'ini oluştur
    test_mat = build_confusion_matrix(test_records, conf_thres=best_conf, iou_thres=0.5)
    test_metrics = metrics_from_matrix(test_mat)

    print("\n=== TEST SETİ ÇALIŞMA NOKTASI SONUÇLARI (conf = {:.3f}, IoU = 0.50) ===".format(best_conf))
    print(
        f"Test Micro P/R/F1  : P={test_metrics['P']:.4f} | R={test_metrics['R']:.4f} | F1={test_metrics['F1']:.4f}\n"
        f"Test Macro P/R/F1  : P={test_metrics['P_macro']:.4f} | R={test_metrics['R_macro']:.4f} | F1={test_metrics['F1_macro']:.4f}\n"
        f"Test TP/FP/FN      : TP={test_metrics['TP']}, FP={test_metrics['FP']} (bg_FP={test_metrics['background_FP']}, cross_FP={test_metrics['cross_class_FP']}), FN={test_metrics['FN']}"
    )

    expected_p = test_metrics["TP"] / (test_metrics["TP"] + test_metrics["FP"])
    assert abs(expected_p - float(test_metrics["P"])) < 1e-4, "Precision ile TP/FP tutarsız!"

    print("\n--- TEST SINIF BAZLI TP / FP / FN / P / R / F1 ---")
    print(f"{'Sınıf':<18} {'TP':>5} {'FP':>5} {'bg_FP':>6} {'FN':>5} {'P':>8} {'R':>8} {'F1':>8}")
    for c in CLASSES:
        print(
            f"{c:<18} {test_metrics[f'{c}_TP']:>5} {test_metrics[f'{c}_FP']:>5} "
            f"{test_metrics[f'{c}_bg_FP']:>6} {test_metrics[f'{c}_FN']:>5} "
            f"{test_metrics[f'{c}_P']:>8.4f} {test_metrics[f'{c}_R']:>8.4f} {test_metrics[f'{c}_F1']:>8.4f}"
        )

    rows = [
        {"split": "val", "conf_threshold": best_conf, "iou_threshold": 0.50, **best_val_metrics},
        {"split": "test", "conf_threshold": best_conf, "iou_threshold": 0.50, **test_metrics},
    ]
    df_op = pd.DataFrame(rows)
    df_op.to_csv(op_csv, index=False, encoding="utf-8")
    print(f"\n[KAYDEDİLDİ] {op_csv}")

    plot_two_panel_confusion_matrix(test_mat, best_conf, test_metrics, cm_png)
    print(f"[KAYDEDİLDİ] {cm_png}")

    hash_after = hashlib.sha256(final_test_csv.read_bytes()).hexdigest()
    assert hash_before == hash_after, "HATA: final_test.csv değiştirildi!"
    print("[DOĞRULANDI] final_test.csv dokunulmadı (SHA-256 eşleşti).")


if __name__ == "__main__":
    main()
