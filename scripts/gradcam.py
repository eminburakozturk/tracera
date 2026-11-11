"""Tracera — Tespit bazlı Grad-CAM ve iki doğrulama testi (Pointing Game + Adebayo Model Randomizasyon).

ÖNEMLİ: Yalnızca CPU üzerinde (device='cpu') çalışır; GPU'ya dokunmaz.
"""

from __future__ import annotations

import argparse
import copy
import os

# GPU kullanımını tamamen devre dışı bırak (arka planda çalışan GPU eğitimine dokunmamak için)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from scipy.stats import spearmanr
from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class LetterboxTransform:
    img_lb: np.ndarray
    ratio: float
    left: int
    top: int
    unpad_w: int
    unpad_h: int
    orig_w: int
    orig_h: int


@dataclass(frozen=True)
class GTBox:
    class_id: int
    class_name: str
    xyxy_orig: tuple[float, float, float, float]
    xyxy_lb: tuple[float, float, float, float]


@dataclass(frozen=True)
class DetectionItem:
    img_path: Path
    stem: str
    class_id: int
    class_name: str
    conf: float
    xyxy_orig: tuple[float, float, float, float]
    xyxy_lb: tuple[float, float, float, float]
    matched_gt: GTBox | None
    iou_with_gt: float


def letterbox_image(img_bgr: np.ndarray, imgsz: int) -> LetterboxTransform:
    """Görüntüyü oranını koruyarak imgsz x imgsz boyutuna letterbox ile yerleştirir."""
    orig_h, orig_w = img_bgr.shape[:2]
    r = min(imgsz / orig_h, imgsz / orig_w)
    unpad_w = int(round(orig_w * r))
    unpad_h = int(round(orig_h * r))

    dw = (imgsz - unpad_w) / 2.0
    dh = (imgsz - unpad_h) / 2.0

    if (orig_w, orig_h) != (unpad_w, unpad_h):
        resized = cv2.resize(img_bgr, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = img_bgr.copy()

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return LetterboxTransform(
        img_lb=padded,
        ratio=r,
        left=left,
        top=top,
        unpad_w=unpad_w,
        unpad_h=unpad_h,
        orig_w=orig_w,
        orig_h=orig_h,
    )


def box_orig_to_lb(xyxy: tuple[float, float, float, float], lb: LetterboxTransform) -> tuple[float, float, float, float]:
    """Orijinal görüntü koordinatlarındaki [x1, y1, x2, y2] kutusunu letterbox koordinatlarına çevirir."""
    x1, y1, x2, y2 = xyxy
    return (
        x1 * lb.ratio + lb.left,
        y1 * lb.ratio + lb.top,
        x2 * lb.ratio + lb.left,
        y2 * lb.ratio + lb.top,
    )


def box_lb_to_orig(xyxy_lb: tuple[float, float, float, float], lb: LetterboxTransform) -> tuple[float, float, float, float]:
    """Letterbox koordinatlarındaki [x1, y1, x2, y2] kutusunu orijinal görüntü koordinatlarına çevirir."""
    x1, y1, x2, y2 = xyxy_lb
    ox1 = max(0.0, min((x1 - lb.left) / lb.ratio, float(lb.orig_w)))
    oy1 = max(0.0, min((y1 - lb.top) / lb.ratio, float(lb.orig_h)))
    ox2 = max(0.0, min((x2 - lb.left) / lb.ratio, float(lb.orig_w)))
    oy2 = max(0.0, min((y2 - lb.top) / lb.ratio, float(lb.orig_h)))
    return (ox1, oy1, ox2, oy2)


def unletterbox_cam(cam_lb: np.ndarray, lb: LetterboxTransform) -> np.ndarray:
    """imgsz x imgsz boyutundaki Grad-CAM haritasından letterbox dolgularını kesip orijinal boyuta ölçekler."""
    crop = cam_lb[lb.top : lb.top + lb.unpad_h, lb.left : lb.left + lb.unpad_w]
    if crop.size == 0:
        crop = cam_lb
    cam_orig = cv2.resize(crop, (lb.orig_w, lb.orig_h), interpolation=cv2.INTER_LINEAR)
    c_min, c_max = float(cam_orig.min()), float(cam_orig.max())
    if c_max > c_min:
        cam_orig = (cam_orig - c_min) / (c_max - c_min)
    else:
        cam_orig = np.zeros_like(cam_orig, dtype=np.float32)
    return cam_orig.astype(np.float32)


def compute_iou_single_vs_many(box: tuple[float, float, float, float], boxes: np.ndarray) -> np.ndarray:
    """Tek bir [x1, y1, x2, y2] kutusu ile (N, 4) boyutundaki kutular arasındaki IoU dizisini hesaplar."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = max(0.0, (box[2] - box[0]) * (box[3] - box[1]))
    area_b = np.maximum(0.0, (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


class YoloDetectionGradCAM:
    """Belirli bir katman üzerinde forward/backward hook ile tespit bazlı klasik Grad-CAM üretici."""

    def __init__(self, nn_model: nn.Module, layer_idx: int, imgsz: int) -> None:
        self.nn_model = nn_model.to("cpu").eval()
        self.layer_idx = layer_idx
        self.imgsz = imgsz
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        # Yalnızca hedef katman ve sonrasındaki parametrelerde gradyanı aktif tut (zincir kuralı gereği yeterli ve hızlı)
        for idx, submod in enumerate(self.nn_model.model):
            req = idx >= layer_idx
            for p in submod.parameters():
                p.requires_grad_(req)

        target_layer = self.nn_model.model[layer_idx]
        self._fwd_handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module: nn.Module, inputs: object, output: torch.Tensor) -> torch.Tensor:
        out_t = output[0] if isinstance(output, (tuple, list)) else output
        if not out_t.requires_grad:
            out_t = out_t.detach().requires_grad_(True)
        self.activations = out_t

        def _grad_hook(grad: torch.Tensor) -> None:
            self.gradients = grad.detach()

        out_t.register_hook(_grad_hook)
        return out_t

    def forward_image(self, x: torch.Tensor) -> tuple[torch.Tensor, np.ndarray]:
        """Görüntüyü ileri besler; ham çıktıyı (1, 4+nc, N) ve (N, 4) xyxy letterbox anchor kutularını döndürür."""
        self.activations = None
        self.gradients = None
        raw_out = self.nn_model(x)[0]  # (1, 4+nc, N)
        xywh = raw_out[0, :4, :].detach().cpu().numpy().T  # (N, 4)
        x1 = xywh[:, 0] - xywh[:, 2] / 2.0
        y1 = xywh[:, 1] - xywh[:, 3] / 2.0
        x2 = xywh[:, 0] + xywh[:, 2] / 2.0
        y2 = xywh[:, 1] + xywh[:, 3] / 2.0
        anchor_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        return raw_out, anchor_xyxy

    def find_best_anchor(self, target_xyxy_lb: tuple[float, float, float, float], anchor_xyxy: np.ndarray) -> int:
        """Hedef kutu ile IoU'su en yüksek olan NMS öncesi anchor indeksini bulur."""
        ious = compute_iou_single_vs_many(target_xyxy_lb, anchor_xyxy)
        max_iou = float(ious.max()) if len(ious) > 0 else 0.0
        if max_iou > 0.0:
            return int(np.argmax(ious))
        # Hiç örtüşme yoksa merkez noktası en yakın anchor'ı seç
        tcx = (target_xyxy_lb[0] + target_xyxy_lb[2]) / 2.0
        tcy = (target_xyxy_lb[1] + target_xyxy_lb[3]) / 2.0
        acx = (anchor_xyxy[:, 0] + anchor_xyxy[:, 2]) / 2.0
        acy = (anchor_xyxy[:, 1] + anchor_xyxy[:, 3]) / 2.0
        dist2 = (acx - tcx) ** 2 + (acy - tcy) ** 2
        return int(np.argmin(dist2))

    def compute_cam_for_anchor(
        self,
        raw_out: torch.Tensor,
        anchor_idx: int,
        class_id: int,
        lb: LetterboxTransform,
    ) -> np.ndarray:
        """Seçilen anchor ve sınıf skoru için klasik Grad-CAM hesaplar ve orijinal görüntü boyutuna geri oturtur."""
        self.nn_model.zero_grad(set_to_none=True)
        self.gradients = None

        target_score = raw_out[0, 4 + class_id, anchor_idx]
        target_score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM aktivasyon veya gradyan yakalanamadı.")

        acts = self.activations.detach()[0]  # (C, H, W)
        grads = self.gradients.detach()[0]   # (C, H, W)

        # Klasik Grad-CAM: kanal ağırlığı = gradyanın uzamsal ortalaması, ağırlıklı toplam, ReLU, normalize
        weights = grads.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)
        cam = torch.relu((weights * acts).sum(dim=0)).cpu().numpy()  # (H, W)

        cam_lb = cv2.resize(cam, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        return unletterbox_cam(cam_lb, lb)

    def close(self) -> None:
        self._fwd_handle.remove()


def build_randomized_model(trained_nn_model: nn.Module, seed: int = 42) -> nn.Module:
    """Adebayo vd. (2018) model randomizasyon testi için aynı mimarinin ağırlıklarını rastgele başlatır."""
    rand_model = copy.deepcopy(trained_nn_model).to("cpu").eval()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    with torch.no_grad():
        for m in rand_model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu", generator=gen)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.normal_(m.weight, mean=1.0, std=0.05, generator=gen)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                if m.running_mean is not None:
                    nn.init.zeros_(m.running_mean)
                if m.running_var is not None:
                    nn.init.ones_(m.running_var)
    return rand_model


def load_gt_boxes(lbl_path: Path, lb: LetterboxTransform, class_names: dict[int, str]) -> list[GTBox]:
    """YOLO .txt etiket dosyasını okur ve orijinal + letterbox koordinatlarında GTBox listesi döndürür."""
    if not lbl_path.is_file():
        return []
    gts: list[GTBox] = []
    for line in lbl_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:5])
        x1 = (xc - bw / 2.0) * lb.orig_w
        y1 = (yc - bh / 2.0) * lb.orig_h
        x2 = (xc + bw / 2.0) * lb.orig_w
        y2 = (yc + bh / 2.0) * lb.orig_h
        xyxy_orig = (x1, y1, x2, y2)
        xyxy_lb = box_orig_to_lb(xyxy_orig, lb)
        gts.append(
            GTBox(
                class_id=cls_id,
                class_name=class_names.get(cls_id, str(cls_id)),
                xyxy_orig=xyxy_orig,
                xyxy_lb=xyxy_lb,
            )
        )
    return gts


def render_gradcam_panel(
    img_bgr: np.ndarray,
    cam_orig: np.ndarray,
    box_orig: tuple[float, float, float, float],
    title_str: str,
    label_str: str,
    box_color_rgb: tuple[int, int, int] = (0, 255, 100),
    extra_gt_box: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Sol tarafta tam PCB üzerinde Grad-CAM + kutu, sağ tarafta kutu çevresinin yakınlaştırılmış (zoom) görünümünü üretir."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    heatmap_bgr = cv2.applyColorMap(np.uint8(255.0 * cam_orig), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.58 * img_rgb + 0.42 * heatmap_rgb)

    # Maksimum Grad-CAM noktasını işaretle
    max_y, max_x = np.unravel_index(int(np.argmax(cam_orig)), cam_orig.shape)
    cv2.drawMarker(
        overlay,
        (int(max_x), int(max_y)),
        (255, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=28,
        thickness=3,
    )

    if extra_gt_box is not None:
        gx1, gy1, gx2, gy2 = map(int, extra_gt_box)
        cv2.rectangle(overlay, (gx1, gy1), (gx2, gy2), (0, 220, 255), thickness=4)

    x1, y1, x2, y2 = map(int, box_orig)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color_rgb, thickness=5)
    cv2.putText(
        overlay,
        label_str,
        (max(10, x1), max(38, y1 - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        box_color_rgb,
        3,
        cv2.LINE_AA,
    )

    # Yakınlaştırılmış (Zoom) bölge (kutu merkezinden ±220 px)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    half_crop = max(180, int(max(x2 - x1, y2 - y1) * 2.5))
    zx1 = max(0, cx - half_crop)
    zy1 = max(0, cy - half_crop)
    zx2 = min(w, cx + half_crop)
    zy2 = min(h, cy + half_crop)
    zoom_crop = overlay[zy1:zy2, zx1:zx2]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1.35, 1.0]})
    axes[0].imshow(overlay)
    axes[0].set_title(f"{title_str}\n(Beyaz artı: Grad-CAM maksimumu)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(zoom_crop)
    axes[1].set_title(f"Yakınlaştırılmış Bölge ({label_str})", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def save_rgb_image(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tracera Tespit Bazlı Grad-CAM ve Doğrulama Testleri (Pointing Game + Model Randomizasyon) [CPU]"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("runs/detect/g_y11s_640/weights/best.pt"),
        help="Model ağırlık dosyası (varsayılan: runs/detect/g_y11s_640/weights/best.pt)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Görüntü boyutu (varsayılan: 640)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("configs/data_grouped.yaml"),
        help="Veri YAML dosyası (varsayılan: configs/data_grouped.yaml)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "test", "train"],
        default="val",
        help="Değerlendirilecek veri bölmesi (varsayılan: val)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Hedef katman indeksi (varsayılan: Detect.f listesinin ilk elemanı, P3/stride 8)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="NMS güven eşiği (varsayılan: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU eşiği (varsayılan: 0.45)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    weights_path = args.weights if args.weights.is_absolute() else (REPO_ROOT / args.weights).resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"Ağırlık dosyası bulunamadı: {weights_path}")

    data_yaml_path = args.data if args.data.is_absolute() else (REPO_ROOT / args.data).resolve()
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    raw_root = Path(data_cfg.get("path", "data/yolo/grouped"))
    dataset_root = raw_root if raw_root.is_absolute() else (REPO_ROOT / raw_root).resolve()
    split_rel = data_cfg.get(args.split, f"{args.split}/images")
    img_dir = (dataset_root / split_rel).resolve()
    lbl_dir = img_dir.parent / "labels"

    raw_names = data_cfg.get("names", {})
    if isinstance(raw_names, list):
        class_names = {i: str(n) for i, n in enumerate(raw_names)}
    else:
        class_names = {int(k): str(v) for k, v in raw_names.items()}

    yolo_obj = YOLO(str(weights_path))
    trained_nn = copy.deepcopy(yolo_obj.model).to("cpu").eval()

    detect_head = trained_nn.model[-1]
    detect_f: list[int] = list(getattr(detect_head, "f", [16, 19, 22]))
    target_layer_idx = args.layer if args.layer is not None else int(detect_f[0])

    model_label = (
        weights_path.parent.parent.name
        if weights_path.parent.name == "weights"
        else weights_path.stem
    )

    print("=" * 78)
    print(f"TRACERA GRAD-CAM VE SANITY CHECK (CPU) — model={model_label} | imgsz={args.imgsz}")
    print(f"Detect modülü .f listesi : {detect_f} (P3/P4/P5 girdileri)")
    print(f"Seçilen hedef katman     : {target_layer_idx} ({type(trained_nn.model[target_layer_idx]).__name__})")
    print(f"Veri bölmesi             : {args.split} ({img_dir})")
    print("=" * 78)

    gradcam_trained = YoloDetectionGradCAM(trained_nn, layer_idx=target_layer_idx, imgsz=args.imgsz)
    rand_nn = build_randomized_model(trained_nn, seed=42)
    gradcam_rand = YoloDetectionGradCAM(rand_nn, layer_idx=target_layer_idx, imgsz=args.imgsz)

    img_paths = sorted(
        [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=lambda p: p.name,
    )

    pointing_hits = 0
    chance_rates: list[float] = []
    spearman_corrs: list[float] = []
    n_tp = 0

    tp_examples: dict[int, tuple[np.ndarray, str]] = {}
    fp_examples: list[tuple[float, np.ndarray, str]] = []
    fn_examples: list[tuple[np.ndarray, str]] = []

    out_dir = REPO_ROOT / "results" / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in img_paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        lb = letterbox_image(img_bgr, imgsz=args.imgsz)
        gts = load_gt_boxes(lbl_dir / f"{img_path.stem}.txt", lb=lb, class_names=class_names)

        x_tensor = (
            torch.from_numpy(lb.img_lb[:, :, ::-1].transpose(2, 0, 1).copy())
            .float()
            .unsqueeze(0)
            / 255.0
        )

        raw_out_tr, anchor_xyxy = gradcam_trained.forward_image(x_tensor)
        nms_preds = non_max_suppression(
            raw_out_tr.detach().clone(),
            conf_thres=args.conf,
            iou_thres=args.iou,
        )[0]  # (M, 6): x1, y1, x2, y2, conf, cls (letterbox koordinatlarında)

        preds_np = nms_preds.cpu().numpy() if len(nms_preds) > 0 else np.zeros((0, 6), dtype=np.float32)

        # GT eşleştirme (IoU >= 0.5, aynı sınıf)
        matched_gt_indices: set[int] = set()
        det_items: list[DetectionItem] = []

        for row in preds_np:
            px1, py1, px2, py2, conf_val, cls_val = row
            cls_id = int(cls_val)
            xyxy_lb = (float(px1), float(py1), float(px2), float(py2))
            xyxy_orig = box_lb_to_orig(xyxy_lb, lb)

            best_gt: GTBox | None = None
            best_iou = 0.0
            best_gt_idx = -1

            for g_idx, g in enumerate(gts):
                if g_idx in matched_gt_indices or g.class_id != cls_id:
                    continue
                iou_val = float(compute_iou_single_vs_many(xyxy_orig, np.asarray([g.xyxy_orig]))[0])
                if iou_val >= 0.5 and iou_val > best_iou:
                    best_iou = iou_val
                    best_gt = g
                    best_gt_idx = g_idx

            if best_gt is not None and best_gt_idx >= 0:
                matched_gt_indices.add(best_gt_idx)

            det_items.append(
                DetectionItem(
                    img_path=img_path,
                    stem=img_path.stem,
                    class_id=cls_id,
                    class_name=class_names.get(cls_id, str(cls_id)),
                    conf=float(conf_val),
                    xyxy_orig=xyxy_orig,
                    xyxy_lb=xyxy_lb,
                    matched_gt=best_gt,
                    iou_with_gt=best_iou,
                )
            )

        # Bu görüntüde TP varsa randomized modelin forward'ını yalnızca 1 kez çalıştır
        has_tp = any(d.matched_gt is not None for d in det_items)
        raw_out_rand: torch.Tensor | None = None
        if has_tp:
            raw_out_rand, _ = gradcam_rand.forward_image(x_tensor)

        for det in det_items:
            anchor_idx = gradcam_trained.find_best_anchor(det.xyxy_lb, anchor_xyxy)
            if det.matched_gt is not None:
                n_tp += 1
                cam_orig = gradcam_trained.compute_cam_for_anchor(
                    raw_out=raw_out_tr,
                    anchor_idx=anchor_idx,
                    class_id=det.class_id,
                    lb=lb,
                )

                # Doğrulama 1: Pointing Game (maksimum nokta GT kutusu içinde mi?)
                max_y, max_x = np.unravel_index(int(np.argmax(cam_orig)), cam_orig.shape)
                gx1, gy1, gx2, gy2 = det.matched_gt.xyxy_orig
                if gx1 <= max_x <= gx2 and gy1 <= max_y <= gy2:
                    pointing_hits += 1

                gt_area = max(0.0, (gx2 - gx1) * (gy2 - gy1))
                chance_rates.append(gt_area / float(lb.orig_w * lb.orig_h))

                # Doğrulama 2: Model Randomizasyon Spearman korelasyonu
                assert raw_out_rand is not None
                cam_rand = gradcam_rand.compute_cam_for_anchor(
                    raw_out=raw_out_rand,
                    anchor_idx=anchor_idx,
                    class_id=det.class_id,
                    lb=lb,
                )
                # Hız için 160x160 ızgarada Spearman sıra korelasyonu
                c_tr_small = cv2.resize(cam_orig, (160, 160), interpolation=cv2.INTER_AREA).ravel()
                c_rn_small = cv2.resize(cam_rand, (160, 160), interpolation=cv2.INTER_AREA).ravel()
                if np.std(c_tr_small) < 1e-8 or np.std(c_rn_small) < 1e-8:
                    rho = 0.0
                else:
                    res_sp = spearmanr(c_tr_small, c_rn_small)
                    rho = float(res_sp.statistic) if not np.isnan(res_sp.statistic) else 0.0
                spearman_corrs.append(rho)

                # Her sınıftan 1 doğru tespit görseli sakla
                if det.class_id not in tp_examples:
                    panel = render_gradcam_panel(
                        img_bgr=img_bgr,
                        cam_orig=cam_orig,
                        box_orig=det.xyxy_orig,
                        title_str=f"[DOĞRU TESPİT - TP] {det.stem} | IoU={det.iou_with_gt:.2f}",
                        label_str=f"{det.class_name} {det.conf:.2f}",
                        box_color_rgb=(40, 235, 80),
                        extra_gt_box=det.matched_gt.xyxy_orig,
                    )
                    fname = f"tp_{det.class_id}_{det.class_name}_{det.stem}.png"
                    save_rgb_image(panel, out_dir / fname)
                    tp_examples[det.class_id] = (panel, fname)

            else:
                # Yanlış Pozitif (FP) adayları (en yüksek güvenli 2 FP'yi tut)
                if len(fp_examples) < 2 or det.conf > fp_examples[-1][0]:
                    cam_fp = gradcam_trained.compute_cam_for_anchor(
                        raw_out=raw_out_tr,
                        anchor_idx=anchor_idx,
                        class_id=det.class_id,
                        lb=lb,
                    )
                    panel_fp = render_gradcam_panel(
                        img_bgr=img_bgr,
                        cam_orig=cam_fp,
                        box_orig=det.xyxy_orig,
                        title_str=f"[YANLIŞ POZİTİF - FP] {det.stem} (GT karşılığı yok)",
                        label_str=f"FP: {det.class_name} {det.conf:.2f}",
                        box_color_rgb=(255, 50, 50),
                    )
                    fp_examples.append((det.conf, panel_fp, f"fp_{det.class_name}_{det.stem}.png"))
                    fp_examples.sort(key=lambda x: -x[0])
                    fp_examples = fp_examples[:2]

        # Kaçırılan Hatalar (FN): eşleşmeyen GT kutuları (en yakın anchor'ın haritası)
        if len(fn_examples) < 2:
            for g_idx, g in enumerate(gts):
                if g_idx in matched_gt_indices:
                    continue
                fn_anchor_idx = gradcam_trained.find_best_anchor(g.xyxy_lb, anchor_xyxy)
                fn_raw_score = float(raw_out_tr[0, 4 + g.class_id, fn_anchor_idx].item())
                cam_fn = gradcam_trained.compute_cam_for_anchor(
                    raw_out=raw_out_tr,
                    anchor_idx=fn_anchor_idx,
                    class_id=g.class_id,
                    lb=lb,
                )
                panel_fn = render_gradcam_panel(
                    img_bgr=img_bgr,
                    cam_orig=cam_fn,
                    box_orig=g.xyxy_orig,
                    title_str=f"[KAÇIRILAN HATA - FN] {img_path.stem} (En yakın anchor skoru={fn_raw_score:.3f})",
                    label_str=f"GT (FN): {g.class_name}",
                    box_color_rgb=(255, 200, 0),
                )
                fn_examples.append((panel_fn, f"fn_{g.class_name}_{img_path.stem}.png"))
                if len(fn_examples) >= 2:
                    break

    gradcam_trained.close()
    gradcam_rand.close()

    # FP ve FN görsellerini kaydet
    for idx, (_, panel_fp, fname_fp) in enumerate(fp_examples, start=1):
        save_rgb_image(panel_fp, out_dir / f"fp_{idx}_{fname_fp[3:]}")
    for idx, (panel_fn, fname_fn) in enumerate(fn_examples, start=1):
        save_rgb_image(panel_fn, out_dir / f"fn_{idx}_{fname_fn[3:]}")

    # Toplu özet görselleri (6 sınıf TP ızgarası + 2 FP / 2 FN hata analizi ızgarası)
    if tp_examples:
        fig_tp, axes_tp = plt.subplots(3, 2, figsize=(18, 16))
        for ax, cls_id in zip(axes_tp.flatten(), sorted(tp_examples.keys())):
            ax.imshow(tp_examples[cls_id][0])
            ax.axis("off")
        plt.suptitle(
            f"Grad-CAM Doğru Tespit Örnekleri (6 Sınıf) — {model_label} (katman={target_layer_idx})",
            fontsize=15,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(out_dir / "gradcam_tp_6classes.png", dpi=130, bbox_inches="tight")
        plt.close(fig_tp)

    err_panels = [p for _, p, _ in fp_examples] + [p for p, _ in fn_examples]
    if err_panels:
        fig_err, axes_err = plt.subplots(2, 2, figsize=(18, 11))
        for ax, pnl in zip(axes_err.flatten(), err_panels):
            ax.imshow(pnl)
            ax.axis("off")
        plt.suptitle(
            f"Grad-CAM Hata Analizi (Üst: 2 Yanlış Pozitif [FP], Alt: 2 Kaçırılan Hata [FN]) — {model_label}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(out_dir / "gradcam_errors_fp_fn.png", dpi=130, bbox_inches="tight")
        plt.close(fig_err)

    pointing_rate = pointing_hits / n_tp if n_tp > 0 else 0.0
    chance_mean = float(np.mean(chance_rates)) if chance_rates else 0.0
    spearman_med = float(np.median(spearman_corrs)) if spearman_corrs else 0.0
    spearman_mean = float(np.mean(spearman_corrs)) if spearman_corrs else 0.0

    sanity_csv = REPO_ROOT / "results" / "gradcam_sanity.csv"
    sanity_row = {
        "model": model_label,
        "imgsz": args.imgsz,
        "katman": target_layer_idx,
        "n_tespit": n_tp,
        "pointing_game": round(pointing_rate, 4),
        "şans_oranı": round(chance_mean, 6),
        "randomizasyon_spearman_medyan": round(spearman_med, 4),
    }

    existing_rows: list[dict[str, object]] = []
    if sanity_csv.is_file():
        df_old = pd.read_csv(sanity_csv)
        existing_rows = [
            r
            for r in df_old.to_dict(orient="records")
            if (str(r.get("model")), int(r.get("imgsz", 0)), int(r.get("katman", -1)))
            != (model_label, args.imgsz, target_layer_idx)
        ]
    existing_rows.append(sanity_row)
    df_sanity = pd.DataFrame(existing_rows)
    sanity_csv.parent.mkdir(parents=True, exist_ok=True)
    df_sanity.to_csv(sanity_csv, index=False, encoding="utf-8")

    print("\n" + "=" * 78)
    print("GRAD-CAM DOĞRULAMA (SANITY CHECK) SONUÇLARI")
    print("=" * 78)
    print(f"Doğru tespit sayısı (n_tespit, IoU>=0.5) : {n_tp}")
    print(f"Pointing Game isabet oranı               : {pointing_rate:.4f} ({pointing_hits}/{n_tp})")
    print(f"Rastgele nokta şans oranı (kutu/görüntü) : {chance_mean:.6f} ({pointing_rate / max(chance_mean, 1e-9):.1f}x şans üstü)")
    print(f"Model Randomizasyon Spearman (medyan)    : {spearman_med:.4f} (ortalama: {spearman_mean:.4f})")
    print(f"Kaydedilen görseller                     : {out_dir} ({len(list(out_dir.glob('*.png')))} PNG)")
    print(f"Kaydedilen tablo                         : {sanity_csv}")
    print("=" * 78)
    print(df_sanity.to_string(index=False))


if __name__ == "__main__":
    main()
