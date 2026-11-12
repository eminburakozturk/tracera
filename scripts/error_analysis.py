"""Test ve Val setlerindeki yanlış pozitiflerin (FP) detaylı dökümü ve görsel montajı.

- Model: runs/detect/g_y11s_1024_full/weights/last.pt (@1024)
- Sabit çalışma noktası: conf = 0.395, IoU = 0.50 (operating_point.csv ile birebir aynı eşleştirme)
- final_test.csv ve operating_point.csv dosyalarına DOKUNMAZ.
- Çıktılar:
  1) results/error_analysis.csv (test ve val üzerindeki tüm FP'lerin detaylı listesi)
  2) results/fp_montage_mouse_bite.png (test mouse_bite en yüksek güvenli 16 FP, 4x4 montaj)
  3) results/fp_montage_spur.png (test spur en yüksek güvenli 16 FP, 4x4 montaj)
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.ops import scale_boxes

REPO_ROOT = Path(__file__).resolve().parent.parent

CLASSES: list[str] = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]

CONF_THRESHOLD = 0.395
IOU_THRESHOLD = 0.50
IMGSZ = 1024


@torch.no_grad()
def collect_split_predictions(
    weights_path: Path,
    data_yaml: Path,
    split: str,
    imgsz: int = IMGSZ,
) -> list[dict[str, Any]]:
    """DetectionValidator ile (FP32, batch=1, rect=True) orijinal görüntü koordinatlarında
    tahminleri (predn) ve GT kutularını (pbatch) dosya adlarıyla birlikte toplar.
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

    items: list[dict[str, Any]] = []
    for batch in dataloader:
        batch = validator.preprocess(batch)
        preds = model(batch["img"])
        preds = validator.postprocess(preds)
        for si, pred in enumerate(preds):
            pbatch = validator._prepare_batch(si, batch)
            predn = validator._prepare_pred(pred)
            im_file = Path(str(pbatch["im_file"]))

            # Orijinal görüntü (ori_shape) piksel koordinatlarına ölçekle
            pred_bboxes_orig = scale_boxes(
                pbatch["imgsz"],
                predn["bboxes"].clone(),
                pbatch["ori_shape"],
                ratio_pad=pbatch["ratio_pad"],
            ).detach().cpu()
            gt_bboxes_orig = scale_boxes(
                pbatch["imgsz"],
                pbatch["bboxes"].clone(),
                pbatch["ori_shape"],
                ratio_pad=pbatch["ratio_pad"],
            ).detach().cpu()

            items.append(
                {
                    "im_file": im_file,
                    "dosya": im_file.name,
                    "stem": im_file.stem,
                    "kart_no": im_file.stem.split("_")[0],
                    "pred_bboxes": predn["bboxes"].detach().cpu(),
                    "pred_bboxes_orig": pred_bboxes_orig,
                    "pred_conf": predn["conf"].detach().cpu(),
                    "pred_cls": predn["cls"].detach().cpu().int(),
                    "gt_bboxes": pbatch["bboxes"].detach().cpu(),
                    "gt_bboxes_orig": gt_bboxes_orig,
                    "gt_cls": pbatch["cls"].detach().cpu().int(),
                }
            )
    return items


def analyze_split_detections(
    items: list[dict[str, Any]],
    split: str,
    conf_thres: float = CONF_THRESHOLD,
    iou_thres: float = IOU_THRESHOLD,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Ultralytics ConfusionMatrix.process_batch ile birebir aynı eşleştirme kuralını uygulayarak
    her tespiti TP veya FP olarak sınıflandırır ve her FP için en yakın GT IoU / sınıf bilgisini çıkarır.
    """
    tp_rows: list[dict[str, Any]] = []
    fp_rows: list[dict[str, Any]] = []

    for item in items:
        keep = item["pred_conf"] > conf_thres
        p_boxes = item["pred_bboxes"][keep]
        p_boxes_orig = item["pred_bboxes_orig"][keep]
        p_conf = item["pred_conf"][keep]
        p_cls = item["pred_cls"][keep]

        g_boxes = item["gt_bboxes"]
        g_boxes_orig = item["gt_bboxes_orig"]
        g_cls = item["gt_cls"]

        n_pred = p_boxes.shape[0]
        n_gt = g_boxes.shape[0]
        if n_pred == 0:
            continue

        if n_gt > 0:
            orig_iou = box_iou(g_boxes, p_boxes).cpu().numpy()  # (M, N)
        else:
            orig_iou = np.zeros((0, n_pred), dtype=np.float32)

        # ConfusionMatrix.process_batch ile birebir aynı açgözlü (greedy) eşleştirme
        iou_work = orig_iou.copy()
        gt_match = np.full(n_gt, -1, dtype=np.int64)
        if n_gt > 0:
            candidates = np.flatnonzero((iou_work > iou_thres).any(0))
            order = np.argsort(-p_conf.numpy()[candidates], kind="stable")
            for det_idx in candidates[order]:
                gt_idx = int(iou_work[:, det_idx].argmax())
                if iou_work[gt_idx, det_idx] > iou_thres:
                    gt_match[gt_idx] = int(det_idx)
                    iou_work[gt_idx] = 0.0

        # Hangi det_idx hangi gt_idx ile eşleşti?
        det_to_matched_gt: dict[int, int] = {}
        for g_i, d_i in enumerate(gt_match.tolist()):
            if d_i >= 0:
                det_to_matched_gt[d_i] = g_i

        for d_i in range(n_pred):
            pred_c_id = int(p_cls[d_i].item())
            pred_c_name = CLASSES[pred_c_id]
            conf_val = float(p_conf[d_i].item())
            box_xyxy = [float(x) for x in p_boxes_orig[d_i].tolist()]

            if n_gt > 0:
                nearest_gt_idx = int(orig_iou[:, d_i].argmax())
                nearest_gt_iou = float(orig_iou[nearest_gt_idx, d_i])
                nearest_gt_cls_name = CLASSES[int(g_cls[nearest_gt_idx].item())]
            else:
                nearest_gt_iou = 0.0
                nearest_gt_cls_name = "yok"

            matched_g_i = det_to_matched_gt.get(d_i, None)
            is_tp = (matched_g_i is not None) and (int(g_cls[matched_g_i].item()) == pred_c_id)

            rec = {
                "split": split,
                "dosya": item["dosya"],
                "stem": item["stem"],
                "im_file": str(item["im_file"]),
                "kart_no": item["kart_no"],
                "sınıf": pred_c_name,
                "güven": round(conf_val, 4),
                "güven_raw": conf_val,
                "en_yakın_gt_iou": round(nearest_gt_iou, 4),
                "en_yakın_gt_iou_raw": nearest_gt_iou,
                "en_yakın_gt_sınıfı": nearest_gt_cls_name,
                "x1": round(box_xyxy[0], 1),
                "y1": round(box_xyxy[1], 1),
                "x2": round(box_xyxy[2], 1),
                "y2": round(box_xyxy[3], 1),
                "gt_bboxes": g_boxes_orig.numpy(),
                "gt_cls": g_cls.numpy(),
            }

            if is_tp:
                tp_rows.append(rec)
            else:
                if matched_g_i is not None:
                    fp_tip = "cross_class_FP (IoU>0.5, farklı sınıf)"
                elif nearest_gt_iou > 0.5:
                    fp_tip = "duplicate_FP (IoU>0.5, aynı GT'ye 2. kutu)"
                elif nearest_gt_iou >= 0.1:
                    fp_tip = "konum_hatası (0.1<=IoU<=0.5)"
                else:
                    fp_tip = "arka_plan (IoU<0.1)"
                rec["fp_tipi"] = fp_tip
                fp_rows.append(rec)

    return tp_rows, fp_rows


def create_fp_montage(
    fp_rows: list[dict[str, Any]],
    target_class: str,
    save_path: Path,
    top_k: int = 16,
    zoom_factor: float = 4.0,
    tile_px: int = 320,
) -> None:
    """Belirtilen sınıf için en yüksek güvenli top_k (16) FP'yi kutu çevresi 4x yakınlaştırma ile
    4x4 montaj olarak çizer ve kaydeder.
    """
    cls_fps = [r for r in fp_rows if r["sınıf"] == target_class]
    cls_fps.sort(key=lambda r: -float(r["güven_raw"]))
    selected = cls_fps[:top_k]

    n_rows, n_cols = 4, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 16.5))
    img_cache: dict[str, np.ndarray] = {}

    for idx, ax in enumerate(axes.flatten()):
        if idx >= len(selected):
            ax.axis("off")
            continue

        r = selected[idx]
        im_path = r["im_file"]
        if im_path not in img_cache:
            bgr = cv2.imread(im_path)
            if bgr is None:
                ax.axis("off")
                continue
            img_cache[im_path] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        img_rgb = img_cache[im_path].copy()
        h_img, w_img = img_rgb.shape[:2]

        x1, y1, x2, y2 = float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)

        # Kutu çevresi 4x pencere (kutu boyutunun 4 katı kenar uzunluğu)
        crop_side = max(80.0, zoom_factor * max(bw, bh))
        half_side = 0.5 * crop_side

        # Görüntüdeki GT kutularını yeşil çerçeveyle, FP kutusunu kırmızı çerçeveyle çiz
        for g_box, g_c in zip(r["gt_bboxes"], r["gt_cls"]):
            gx1, gy1, gx2, gy2 = [int(round(v)) for v in g_box]
            cv2.rectangle(img_rgb, (gx1, gy1), (gx2, gy2), (0, 230, 100), thickness=2)

        cv2.rectangle(
            img_rgb,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            (255, 45, 45),
            thickness=2,
        )

        cx1 = max(0, int(round(cx - half_side)))
        cy1 = max(0, int(round(cy - half_side)))
        cx2 = min(w_img, int(round(cx + half_side)))
        cy2 = min(h_img, int(round(cy + half_side)))

        crop = img_rgb[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            crop = img_rgb
        crop_resized = cv2.resize(crop, (tile_px, tile_px), interpolation=cv2.INTER_LINEAR)

        ax.imshow(crop_resized)
        iou_val = float(r["en_yakın_gt_iou"])
        ax.set_title(
            f"#{idx + 1} {r['dosya']}\nconf={r['güven']:.3f} | GT IoU={iou_val:.2f}",
            fontsize=9.5,
            fontweight="bold",
        )
        ax.axis("off")

    fig.suptitle(
        f"Test Seti — '{target_class}' En Yüksek Güvenli {len(selected)} Yanlış Pozitif (FP) Montajı (4× Yakınlaştırma)\n"
        f"(Kırmızı: FP Tahmini, Yeşil: Varsa Yakındaki GT Kutusu | conf ≥ {CONF_THRESHOLD}, IoU = {IOU_THRESHOLD})",
        fontsize=13.5,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def print_split_summary_tables(
    split_name: str,
    tp_rows: list[dict[str, Any]],
    fp_rows: list[dict[str, Any]],
) -> None:
    print("\n" + "=" * 92)
    print(f"HATA ANALİZİ DÖKÜMÜ — SPLIT: {split_name.upper()} (Toplam TP={len(tp_rows)}, Toplam FP={len(fp_rows)})")
    print("=" * 92)

    # 1) Kart x Sınıf FP Tablosu
    df_fp = pd.DataFrame(fp_rows)
    print(f"\n[1] {split_name.upper()} — KART × SINIF FP SAYISI TABLOSU:")
    if len(df_fp) > 0:
        boards = sorted(df_fp["kart_no"].unique().tolist())
        pivot_data: list[dict[str, Any]] = []
        for b in boards:
            sub = df_fp[df_fp["kart_no"] == b]
            row_d: dict[str, Any] = {"kart_no": b}
            for c in CLASSES:
                row_d[c] = int((sub["sınıf"] == c).sum())
            row_d["TOPLAM"] = int(len(sub))
            pivot_data.append(row_d)
        tot_d: dict[str, Any] = {"kart_no": "TOPLAM"}
        for c in CLASSES:
            tot_d[c] = int((df_fp["sınıf"] == c).sum())
        tot_d["TOPLAM"] = int(len(df_fp))
        pivot_data.append(tot_d)
        print(pd.DataFrame(pivot_data).to_string(index=False))
    else:
        print("FP yok.")

    # 2) Güven Histogramı (0.395 - 1.0, 0.05 adım) TP ve FP ayrı
    # İki standart binleme: (a) 0.395'ten 0.05 adımlı ve (b) [0.395, 0.45), [0.45, 0.50), ..., [0.95, 1.00]
    edges_round = [0.395, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0001]
    tp_confs = np.array([r["güven_raw"] for r in tp_rows], dtype=np.float64)
    fp_confs = np.array([r["güven_raw"] for r in fp_rows], dtype=np.float64)

    hist_rows: list[dict[str, Any]] = []
    for i in range(len(edges_round) - 1):
        lo, hi = edges_round[i], edges_round[i + 1]
        hi_disp = 1.00 if hi > 1.0 else hi
        n_tp_bin = int(((tp_confs >= lo) & (tp_confs < hi)).sum()) if len(tp_confs) > 0 else 0
        n_fp_bin = int(((fp_confs >= lo) & (fp_confs < hi)).sum()) if len(fp_confs) > 0 else 0
        n_mb_bin = (
            int(
                sum(
                    1
                    for r in fp_rows
                    if r["sınıf"] == "mouse_bite" and lo <= r["güven_raw"] < hi
                )
            )
            if len(fp_rows) > 0
            else 0
        )
        n_sp_bin = (
            int(
                sum(
                    1
                    for r in fp_rows
                    if r["sınıf"] == "spur" and lo <= r["güven_raw"] < hi
                )
            )
            if len(fp_rows) > 0
            else 0
        )
        tot_bin = n_tp_bin + n_fp_bin
        fp_rate = (n_fp_bin / tot_bin * 100.0) if tot_bin > 0 else 0.0
        hist_rows.append(
            {
                "güven_aralığı": f"[{lo:.3f}, {hi_disp:.2f}{']' if hi > 1.0 else ')'}",
                "TP": n_tp_bin,
                "FP": n_fp_bin,
                "FP_mouse_bite": n_mb_bin,
                "FP_spur": n_sp_bin,
                "FP_oranı_%": round(fp_rate, 1),
            }
        )

    print(f"\n[2] {split_name.upper()} — GÜVEN HİSTOGRAMI (0.395–1.00, 0.05 ADIM, TP ve FP AYRI):")
    print(pd.DataFrame(hist_rows).to_string(index=False))

    # 3) En yakın GT ile IoU Dağılımı (IoU < 0.1 vs 0.1 <= IoU <= 0.5 vs IoU > 0.5)
    iou_summary_rows: list[dict[str, Any]] = []
    for c in CLASSES + ["TOPLAM"]:
        sub_r = fp_rows if c == "TOPLAM" else [r for r in fp_rows if r["sınıf"] == c]
        n_tot = len(sub_r)
        n_lt_01 = sum(1 for r in sub_r if r["en_yakın_gt_iou_raw"] < 0.1)
        n_01_05 = sum(1 for r in sub_r if 0.1 <= r["en_yakın_gt_iou_raw"] <= 0.5)
        n_gt_05 = sum(1 for r in sub_r if r["en_yakın_gt_iou_raw"] > 0.5)
        iou_summary_rows.append(
            {
                "sınıf": c,
                "toplam_FP": n_tot,
                "IoU<0.1 (Arka Plan / GT'ye Değmiyor)": n_lt_01,
                "0.1<=IoU<=0.5 (Konum Hatası)": n_01_05,
                "IoU>0.5 (Sınıf Karışımı / Çift Kutu)": n_gt_05,
            }
        )

    print(f"\n[3] {split_name.upper()} — FP EN YAKIN GT IoU KATEGORİSİ TABLOSU:")
    print(pd.DataFrame(iou_summary_rows).to_string(index=False))


def main() -> None:
    final_test_csv = REPO_ROOT / "results" / "final_test.csv"
    op_csv = REPO_ROOT / "results" / "operating_point.csv"
    hash_test_before = hashlib.sha256(final_test_csv.read_bytes()).hexdigest()
    hash_op_before = hashlib.sha256(op_csv.read_bytes()).hexdigest()

    weights_path = REPO_ROOT / "runs" / "detect" / "g_y11s_1024_full" / "weights" / "last.pt"
    val_yaml = REPO_ROOT / "runs" / "data_eval_val_last_1024.yaml"
    test_yaml = REPO_ROOT / "runs" / "data_eval_test_last_1024.yaml"

    print("1) Test ve Val tahminleri toplanıyor (last.pt @1024, conf=0.395, IoU=0.50)...")
    test_items = collect_split_predictions(weights_path, test_yaml, "test", imgsz=IMGSZ)
    val_items = collect_split_predictions(weights_path, val_yaml, "val", imgsz=IMGSZ)

    test_tp, test_fp = analyze_split_detections(test_items, split="test")
    val_tp, val_fp = analyze_split_detections(val_items, split="val")

    # operating_point.csv ile birebir tutarlılık kontrolü
    assert len(test_tp) == 416 and len(test_fp) == 142, (
        f"Test TP/FP sayısı operating_point.csv (416/142) ile uyuşmuyor: TP={len(test_tp)}, FP={len(test_fp)}"
    )
    assert len(val_tp) == 514 and len(val_fp) == 16, (
        f"Val TP/FP sayısı operating_point.csv (514/16) ile uyuşmuyor: TP={len(val_tp)}, FP={len(val_fp)}"
    )

    print_split_summary_tables("test", test_tp, test_fp)
    print_split_summary_tables("val", val_tp, val_fp)

    # CSV kaydı (test + val tüm FP'ler, güvene göre azalan sırada)
    csv_cols = [
        "split",
        "dosya",
        "kart_no",
        "sınıf",
        "güven",
        "en_yakın_gt_iou",
        "en_yakın_gt_sınıfı",
        "fp_tipi",
        "x1",
        "y1",
        "x2",
        "y2",
    ]
    all_fp_sorted = sorted(test_fp, key=lambda r: (r["kart_no"], r["sınıf"], -r["güven_raw"])) + sorted(
        val_fp, key=lambda r: (r["kart_no"], r["sınıf"], -r["güven_raw"])
    )
    df_out = pd.DataFrame([{k: r[k] for k in csv_cols} for r in all_fp_sorted])
    out_csv = REPO_ROOT / "results" / "error_analysis.csv"
    df_out.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\n[KAYDEDİLDİ] {out_csv} ({len(df_out)} satır: {len(test_fp)} test FP + {len(val_fp)} val FP)")

    # mouse_bite ve spur için en yüksek güvenli 16'şar FP montajı (Test)
    mb_png = REPO_ROOT / "results" / "fp_montage_mouse_bite.png"
    sp_png = REPO_ROOT / "results" / "fp_montage_spur.png"
    create_fp_montage(test_fp, target_class="mouse_bite", save_path=mb_png, top_k=16, zoom_factor=4.0)
    print(f"[KAYDEDİLDİ] {mb_png}")
    create_fp_montage(test_fp, target_class="spur", save_path=sp_png, top_k=16, zoom_factor=4.0)
    print(f"[KAYDEDİLDİ] {sp_png}")

    assert hashlib.sha256(final_test_csv.read_bytes()).hexdigest() == hash_test_before, "final_test.csv değişti!"
    assert hashlib.sha256(op_csv.read_bytes()).hexdigest() == hash_op_before, "operating_point.csv değişti!"
    print("[DOĞRULANDI] final_test.csv ve operating_point.csv değiştirilmedi.")


if __name__ == "__main__":
    main()
