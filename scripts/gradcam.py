"""Tracera — Tespit bazlı CAM (Grad-CAM, HiResCAM, LayerCAM) yerel doğrulama protokolü,
mutlak bütçeli (1x/2x GT kutu alanı) yapısal kaydırmalı silme (deletion) sadakat testi,
rapor özeti (results/xai_summary.md) ve val görsel üretimi (6 TP + 2 FP + 2 FN).

Varsayılan kilitli ayar: --method layercam --layer 4
ÖNEMLİ: Yalnızca CPU üzerinde (device='cpu') çalışır; GPU'ya dokunmaz.
"""

from __future__ import annotations

import argparse
import copy
import os
import random

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
METHODS: tuple[str, ...] = ("gradcam", "hirescam", "layercam")
DELETION_ACTIVE_COMBOS: set[tuple[int, str]] = {
    (4, "hirescam"),
    (4, "layercam"),
    (16, "layercam"),
}


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
class SampledTP:
    img_path: Path
    stem: str
    class_id: int
    class_name: str
    conf: float
    xyxy_orig: tuple[float, float, float, float]
    xyxy_lb: tuple[float, float, float, float]
    matched_gt: GTBox
    all_gts: list[GTBox]
    iou_with_gt: float


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


@dataclass(frozen=True)
class GradSupportWindow:
    lb_x1: int
    lb_y1: int
    lb_x2: int
    lb_y2: int
    orig_x1: int
    orig_y1: int
    orig_x2: int
    orig_y2: int
    area_ratio: float


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
    """imgsz x imgsz boyutundaki CAM haritasından letterbox dolgularını kesip orijinal boyuta ölçekler."""
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


def compute_grad_support_window(
    grads: torch.Tensor,
    imgsz: int,
    lb: LetterboxTransform,
) -> GradSupportWindow:
    """Hedef katmanda gradyanın sıfırdan farklı olduğu bölgenin sınırlayıcı kutusunu girdi koordinatlarına taşır."""
    grad_mag = grads.abs().sum(dim=0).cpu().numpy()  # (H_f, W_f)
    hf, wf = grad_mag.shape
    nz_y, nz_x = np.where(grad_mag > 1e-12)

    if len(nz_x) == 0:
        fx1, fy1, fx2, fy2 = 0, 0, wf, hf
    else:
        fx1, fy1 = int(nz_x.min()), int(nz_y.min())
        fx2, fy2 = int(nz_x.max()) + 1, int(nz_y.max()) + 1

    stride_x = imgsz / float(wf)
    stride_y = imgsz / float(hf)

    lb_x1 = max(lb.left, min(int(round(fx1 * stride_x)), lb.left + lb.unpad_w - 1))
    lb_y1 = max(lb.top, min(int(round(fy1 * stride_y)), lb.top + lb.unpad_h - 1))
    lb_x2 = max(lb_x1 + 1, min(int(round(fx2 * stride_x)), lb.left + lb.unpad_w))
    lb_y2 = max(lb_y1 + 1, min(int(round(fy2 * stride_y)), lb.top + lb.unpad_h))

    ox1_f, oy1_f, ox2_f, oy2_f = box_lb_to_orig((float(lb_x1), float(lb_y1), float(lb_x2), float(lb_y2)), lb)
    ox1 = max(0, min(int(round(ox1_f)), lb.orig_w - 1))
    oy1 = max(0, min(int(round(oy1_f)), lb.orig_h - 1))
    ox2 = max(ox1 + 1, min(int(round(ox2_f)), lb.orig_w))
    oy2 = max(oy1 + 1, min(int(round(oy2_f)), lb.orig_h))

    win_area = float((ox2 - ox1) * (oy2 - oy1))
    img_area = float(lb.orig_w * lb.orig_h)
    return GradSupportWindow(
        lb_x1=lb_x1,
        lb_y1=lb_y1,
        lb_x2=lb_x2,
        lb_y2=lb_y2,
        orig_x1=ox1,
        orig_y1=oy1,
        orig_x2=ox2,
        orig_y2=oy2,
        area_ratio=win_area / img_area if img_area > 0 else 1.0,
    )


def compute_cam_pair_from_acts_grads(
    acts: torch.Tensor,
    grads: torch.Tensor,
    method: str,
    imgsz: int,
    lb: LetterboxTransform,
) -> tuple[np.ndarray, np.ndarray]:
    """Aktivasyon ve gradyandan hem (imgsz, imgsz) letterbox CAM hem (orig_h, orig_w) orijinal CAM döndürür."""
    if method == "gradcam":
        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam_2d = torch.relu((weights * acts).sum(dim=0)).cpu().numpy()
    elif method == "hirescam":
        cam_2d = torch.relu((grads * acts).sum(dim=0)).cpu().numpy()
    elif method == "layercam":
        cam_2d = torch.relu((torch.relu(grads) * acts).sum(dim=0)).cpu().numpy()
    else:
        raise ValueError(f"Bilinmeyen CAM yöntemi: {method}")

    cam_lb = cv2.resize(cam_2d, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    l_min, l_max = float(cam_lb.min()), float(cam_lb.max())
    if l_max > l_min:
        cam_lb = (cam_lb - l_min) / (l_max - l_min)
    else:
        cam_lb = np.zeros_like(cam_lb, dtype=np.float32)

    cam_orig = unletterbox_cam(cam_lb, lb)
    return cam_lb, cam_orig


class MultiLayerYoloCAM:
    """Birden fazla hedef katman (ör. 16 ve 4) üzerinde tek forward/backward ile Grad-CAM/HiResCAM/LayerCAM üretici."""

    def __init__(self, nn_model: nn.Module, layer_indices: list[int], imgsz: int) -> None:
        self.nn_model = nn_model.to("cpu").eval()
        self.layer_indices = sorted(set(layer_indices))
        self.imgsz = imgsz
        self.activations: dict[int, torch.Tensor] = {}
        self.gradients: dict[int, torch.Tensor] = {}
        self._handles: list[object] = []

        min_layer = min(self.layer_indices)
        for idx, submod in enumerate(self.nn_model.model):
            req = idx >= min_layer
            for p in submod.parameters():
                p.requires_grad_(req)

        for lid in self.layer_indices:
            handle = self.nn_model.model[lid].register_forward_hook(self._make_forward_hook(lid))
            self._handles.append(handle)

    def _make_forward_hook(self, lid: int):
        def _hook(module: nn.Module, inputs: object, output: torch.Tensor) -> torch.Tensor:
            out_t = output[0] if isinstance(output, (tuple, list)) else output
            if not out_t.requires_grad:
                out_t = out_t.detach().requires_grad_(True)
            self.activations[lid] = out_t

            def _grad_hook(grad: torch.Tensor) -> None:
                self.gradients[lid] = grad.detach()

            out_t.register_hook(_grad_hook)
            return out_t

        return _hook

    def forward_image(self, x: torch.Tensor) -> tuple[torch.Tensor, np.ndarray]:
        self.activations.clear()
        self.gradients.clear()
        raw_out = self.nn_model(x)[0]
        xywh = raw_out[0, :4, :].detach().cpu().numpy().T
        x1 = xywh[:, 0] - xywh[:, 2] / 2.0
        y1 = xywh[:, 1] - xywh[:, 3] / 2.0
        x2 = xywh[:, 0] + xywh[:, 2] / 2.0
        y2 = xywh[:, 1] + xywh[:, 3] / 2.0
        anchor_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        return raw_out, anchor_xyxy

    def find_best_anchor(self, target_xyxy_lb: tuple[float, float, float, float], anchor_xyxy: np.ndarray) -> int:
        ious = compute_iou_single_vs_many(target_xyxy_lb, anchor_xyxy)
        max_iou = float(ious.max()) if len(ious) > 0 else 0.0
        if max_iou > 0.0:
            return int(np.argmax(ious))
        tcx = (target_xyxy_lb[0] + target_xyxy_lb[2]) / 2.0
        tcy = (target_xyxy_lb[1] + target_xyxy_lb[3]) / 2.0
        acx = (anchor_xyxy[:, 0] + anchor_xyxy[:, 2]) / 2.0
        acy = (anchor_xyxy[:, 1] + anchor_xyxy[:, 3]) / 2.0
        dist2 = (acx - tcx) ** 2 + (acy - tcy) ** 2
        return int(np.argmin(dist2))

    def backward_and_extract(
        self,
        raw_out: torch.Tensor,
        anchor_idx: int,
        class_id: int,
        lb: LetterboxTransform,
        methods: tuple[str, ...] = METHODS,
    ) -> dict[int, tuple[GradSupportWindow, dict[str, tuple[np.ndarray, np.ndarray]]]]:
        self.nn_model.zero_grad(set_to_none=True)
        self.gradients.clear()

        target_score = raw_out[0, 4 + class_id, anchor_idx]
        target_score.backward(retain_graph=True)

        result: dict[int, tuple[GradSupportWindow, dict[str, tuple[np.ndarray, np.ndarray]]]] = {}
        for lid in self.layer_indices:
            acts = self.activations[lid].detach()[0]
            grads = self.gradients[lid].detach()[0]
            win = compute_grad_support_window(grads, imgsz=self.imgsz, lb=lb)
            cams = {
                m: compute_cam_pair_from_acts_grads(acts, grads, method=m, imgsz=self.imgsz, lb=lb)
                for m in methods
            }
            result[lid] = (win, cams)
        return result

    def close(self) -> None:
        for h in self._handles:
            h.remove()  # type: ignore[attr-defined]


def build_randomized_model(trained_nn_model: nn.Module, seed: int = 42) -> nn.Module:
    """Adebayo vd. (2018) model randomizasyon testi için aynı mimarinin ağırlıklarını rastgele başlatır."""
    rand_model = copy.deepcopy(trained_nn_model).to("cpu").eval()
    for m in rand_model.modules():
        m._forward_hooks.clear()
        m._forward_pre_hooks.clear()
        m._backward_hooks.clear()
        m._backward_pre_hooks.clear()

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


def make_overlay_and_zoom(
    img_bgr: np.ndarray,
    cam_orig: np.ndarray,
    box_orig: tuple[float, float, float, float],
    label_str: str,
    box_color_rgb: tuple[int, int, int] = (0, 255, 100),
    extra_gt_box: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Tam görüntü overlay ve kutu çevresi zoom görüntüsünü RGB olarak döndürür."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    heatmap_bgr = cv2.applyColorMap(np.uint8(255.0 * cam_orig), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.58 * img_rgb + 0.42 * heatmap_rgb)

    max_y, max_x = np.unravel_index(int(np.argmax(cam_orig)), cam_orig.shape)
    cv2.drawMarker(
        overlay,
        (int(max_x), int(max_y)),
        (255, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=30,
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

    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    half_crop = max(180, int(max(x2 - x1, y2 - y1) * 2.5))
    zx1 = max(0, cx - half_crop)
    zy1 = max(0, cy - half_crop)
    zx2 = min(w, cx + half_crop)
    zy2 = min(h, cy + half_crop)
    zoom_crop = overlay[zy1:zy2, zx1:zx2]
    return overlay, zoom_crop


def render_single_method_panel(
    img_bgr: np.ndarray,
    cam_orig: np.ndarray,
    box_orig: tuple[float, float, float, float],
    title_str: str,
    label_str: str,
    box_color_rgb: tuple[int, int, int] = (0, 255, 100),
    extra_gt_box: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Tek bir yöntem için tam PCB + yakınlaştırılmış bölge panelini üretir."""
    overlay, zoom_crop = make_overlay_and_zoom(
        img_bgr=img_bgr,
        cam_orig=cam_orig,
        box_orig=box_orig,
        label_str=label_str,
        box_color_rgb=box_color_rgb,
        extra_gt_box=extra_gt_box,
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1.35, 1.0]})
    axes[0].imshow(overlay)
    axes[0].set_title(f"{title_str}\n(Beyaz artı: harita maksimumu)", fontsize=11, fontweight="bold")
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


def compute_spearman_safe(a: np.ndarray, b: np.ndarray, max_side: int = 160) -> float:
    h, w = a.shape[:2]
    if h > max_side or w > max_side:
        nh = min(max_side, max(1, h))
        nw = min(max_side, max(1, w))
        a_flat = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_AREA).ravel()
        b_flat = cv2.resize(b, (nw, nh), interpolation=cv2.INTER_AREA).ravel()
    else:
        a_flat = a.ravel()
        b_flat = b.ravel()

    if len(a_flat) < 2 or np.std(a_flat) < 1e-8 or np.std(b_flat) < 1e-8:
        return 0.0
    res = spearmanr(a_flat, b_flat)
    val = float(res.statistic)
    return 0.0 if np.isnan(val) else val


def compute_window_energy_in_gt(
    cam_orig: np.ndarray,
    win: GradSupportWindow,
    gt_box_orig: tuple[float, float, float, float],
) -> tuple[float, float]:
    wx1, wy1, wx2, wy2 = win.orig_x1, win.orig_y1, win.orig_x2, win.orig_y2
    gx1, gy1, gx2, gy2 = gt_box_orig

    ix1 = max(wx1, int(round(gx1)))
    iy1 = max(wy1, int(round(gy1)))
    ix2 = min(wx2, int(round(gx2)))
    iy2 = min(wy2, int(round(gy2)))

    win_slice = cam_orig[wy1:wy2, wx1:wx2]
    total_energy = float(win_slice.sum())
    win_area = float(max(1, (wx2 - wx1) * (wy2 - wy1)))
    inter_area = float(max(0, ix2 - ix1) * max(0, iy2 - iy1))
    base_ratio = inter_area / win_area

    if ix2 <= ix1 or iy2 <= iy1 or total_energy <= 1e-8:
        return 0.0, base_ratio

    box_energy = float(cam_orig[iy1:iy2, ix1:ix2].sum())
    return box_energy / total_energy, base_ratio


def shift_binary_mask_within_window(
    mask_2d: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """2B ikili maskeyi pencere (wh, ww) içinde rastgele kaydırır (tamamen pencere içinde kalacak şekilde)."""
    wh, ww = mask_2d.shape
    ys, xs = np.where(mask_2d)
    if len(ys) == 0:
        return mask_2d.copy()

    my_min, my_max = int(ys.min()), int(ys.max())
    mx_min, mx_max = int(xs.min()), int(xs.max())

    dy_min, dy_max = -my_min, (wh - 1) - my_max
    dx_min, dx_max = -mx_min, (ww - 1) - mx_max

    n_valid = (dy_max - dy_min + 1) * (dx_max - dx_min + 1)
    if n_valid > 1:
        while True:
            dy = int(rng.integers(dy_min, dy_max + 1))
            dx = int(rng.integers(dx_min, dx_max + 1))
            if dy != 0 or dx != 0:
                break
        shifted = np.zeros_like(mask_2d, dtype=bool)
        shifted[ys + dy, xs + dx] = True
        return shifted

    dy = int(rng.integers(1, max(2, wh))) if wh > 1 else 0
    dx = int(rng.integers(1, max(2, ww))) if ww > 1 else 0
    return np.roll(mask_2d, shift=(dy, dx), axis=(0, 1))


def evaluate_absolute_budget_deletion(
    eval_model: nn.Module,
    img_lb: np.ndarray,
    blurred_lb: np.ndarray,
    win: GradSupportWindow,
    cam_lb: np.ndarray,
    gt_box_lb: tuple[float, float, float, float],
    anchor_idx: int,
    class_id: int,
    orig_score: float,
    rng: np.random.Generator,
    n_repeats: int = 5,
) -> dict[str, tuple[float, float]]:
    """Tek bir (katman, yöntem) için k = 1x ve 2x GT kutu alanı bütçeleriyle:
    - Haritanın en yüksek k pikseli Gauss sigma=5 ile bulanıklaştırılır -> göreli skor düşüşü (s0 - s1) / s0
    - Kontrol: aynı ikili maske pencere içinde rastgele kaydırılır (5 tekrar ortalaması) -> göreli skor düşüşü
    Döndürür: {'1x': (rel_drop_map, rel_drop_ctrl), '2x': (rel_drop_map, rel_drop_ctrl)}
    """
    x1, y1, x2, y2 = win.lb_x1, win.lb_y1, win.lb_x2, win.lb_y2
    wh = max(1, y2 - y1)
    ww = max(1, x2 - x1)
    n_win_pixels = wh * ww

    gx1, gy1, gx2, gy2 = gt_box_lb
    gt_area_lb = max(1, int(round(max(1.0, gx2 - gx1) * max(1.0, gy2 - gy1))))

    budgets = {
        "1x": min(n_win_pixels, max(1, gt_area_lb)),
        "2x": min(n_win_pixels, max(1, 2 * gt_area_lb)),
    }

    orig_patch = img_lb[y1:y2, x1:x2]
    blur_patch = blurred_lb[y1:y2, x1:x2]
    cam_patch_flat = cam_lb[y1:y2, x1:x2].ravel()

    perturbed_tensors: list[torch.Tensor] = []
    budget_keys = ["1x", "2x"]

    for b_key in budget_keys:
        k_px = budgets[b_key]
        if k_px >= n_win_pixels:
            mask_2d = np.ones((wh, ww), dtype=bool)
        else:
            top_idx = np.argpartition(cam_patch_flat, -k_px)[-k_px:]
            mask_flat = np.zeros(n_win_pixels, dtype=bool)
            mask_flat[top_idx] = True
            mask_2d = mask_flat.reshape(wh, ww)

        img_map = img_lb.copy()
        img_map[y1:y2, x1:x2] = np.where(mask_2d[:, :, None], blur_patch, orig_patch)
        perturbed_tensors.append(
            torch.from_numpy(img_map[:, :, ::-1].transpose(2, 0, 1).copy()).float() / 255.0
        )

        for _ in range(n_repeats):
            shifted_2d = shift_binary_mask_within_window(mask_2d, rng=rng)
            img_ctrl = img_lb.copy()
            img_ctrl[y1:y2, x1:x2] = np.where(shifted_2d[:, :, None], blur_patch, orig_patch)
            perturbed_tensors.append(
                torch.from_numpy(img_ctrl[:, :, ::-1].transpose(2, 0, 1).copy()).float() / 255.0
            )

    batch_tensor = torch.stack(perturbed_tensors, dim=0)
    with torch.inference_mode():
        batch_out = eval_model(batch_tensor)[0]
        scores = batch_out[:, 4 + class_id, anchor_idx].cpu().numpy()

    denom = max(abs(orig_score), 1e-8)
    stride_block = 1 + n_repeats
    out_dict: dict[str, tuple[float, float]] = {}

    for b_idx, b_key in enumerate(budget_keys):
        base = b_idx * stride_block
        s_map = float(scores[base])
        s_ctrls = scores[base + 1 : base + stride_block]
        rel_drop_map = float((orig_score - s_map) / denom)
        rel_drop_ctrl = float(np.mean((orig_score - s_ctrls) / denom))
        out_dict[b_key] = (rel_drop_map, rel_drop_ctrl)

    return out_dict


def generate_val_visuals(
    yolo_obj: YOLO,
    img_dir: Path,
    lbl_dir: Path,
    class_names: dict[int, str],
    model_label: str,
    imgsz: int,
    method: str,
    layer_idx: int,
    conf_thres: float,
    iou_thres: float,
    split: str,
) -> None:
    """Val bölmesi üzerinde 6 TP (her sınıftan 1) + 2 FP + 2 FN için LayerCAM@4 görsellerini üretir."""
    assert split != "test", "XAI görsel üretiminde test bölmesi KULLANILAMAZ (yalnızca val)."

    out_dir = REPO_ROOT / "results" / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Eski tekil tp_/fp_/fn_ dosyalarını temizle (farklı görüntü stem'lerinden çift dosya kalmaması için)
    for old_pat in ("tp_*.png", "fp_*.png", "fn_*.png"):
        for old_p in out_dir.glob(old_pat):
            old_p.unlink(missing_ok=True)

    trained_nn = copy.deepcopy(yolo_obj.model).to("cpu").eval()
    eval_nn = copy.deepcopy(yolo_obj.model).fuse().to("cpu").eval()
    cam_engine = MultiLayerYoloCAM(trained_nn, layer_indices=[layer_idx], imgsz=imgsz)

    img_paths = sorted(
        [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=lambda p: p.name,
    )

    tp_single_panels: dict[int, np.ndarray] = {}
    fp_candidates: list[tuple[float, np.ndarray, str]] = []
    fn_panels: list[tuple[np.ndarray, str]] = []

    print(
        f"\n>>> XAI GÖRSEL ÜRETİMİ (split={split}, model={model_label}, imgsz={imgsz}, "
        f"method={method}, katman={layer_idx})"
    )

    for img_path in img_paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        lb = letterbox_image(img_bgr, imgsz=imgsz)
        gts = load_gt_boxes(lbl_dir / f"{img_path.stem}.txt", lb=lb, class_names=class_names)

        x_tensor = (
            torch.from_numpy(lb.img_lb[:, :, ::-1].transpose(2, 0, 1).copy())
            .float()
            .unsqueeze(0)
            / 255.0
        )

        with torch.inference_mode():
            raw_pred = eval_nn(x_tensor)[0]
            nms_preds = non_max_suppression(raw_pred.clone(), conf_thres=conf_thres, iou_thres=iou_thres)[0]
        preds_np = nms_preds.cpu().numpy() if len(nms_preds) > 0 else np.zeros((0, 6), dtype=np.float32)

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

        need_tp = any(d.matched_gt is not None and d.class_id not in tp_single_panels for d in det_items)
        need_fp = any(
            d.matched_gt is None and (len(fp_candidates) < 2 or d.conf > fp_candidates[-1][0])
            for d in det_items
        )
        need_fn = len(fn_panels) < 2 and any(g_idx not in matched_gt_indices for g_idx in range(len(gts)))

        if not (need_tp or need_fp or need_fn):
            continue

        raw_out_tr, anchor_xyxy = cam_engine.forward_image(x_tensor)

        for det in det_items:
            if det.matched_gt is not None:
                if det.class_id not in tp_single_panels:
                    anchor_idx = cam_engine.find_best_anchor(det.xyxy_lb, anchor_xyxy)
                    extracted = cam_engine.backward_and_extract(
                        raw_out=raw_out_tr,
                        anchor_idx=anchor_idx,
                        class_id=det.class_id,
                        lb=lb,
                        methods=(method,),
                    )
                    _, cam_orig = extracted[layer_idx][1][method]
                    single_pnl = render_single_method_panel(
                        img_bgr=img_bgr,
                        cam_orig=cam_orig,
                        box_orig=det.xyxy_orig,
                        title_str=(
                            f"[{method.upper()}@{layer_idx} - TP] {det.stem} | "
                            f"imgsz={imgsz} | IoU={det.iou_with_gt:.2f}"
                        ),
                        label_str=f"{det.class_name} {det.conf:.2f}",
                        box_color_rgb=(40, 235, 80),
                        extra_gt_box=det.matched_gt.xyxy_orig,
                    )
                    tp_fname = f"tp_{det.class_id}_{det.class_name}_{det.stem}.png"
                    save_rgb_image(single_pnl, out_dir / tp_fname)
                    tp_single_panels[det.class_id] = single_pnl
                    print(f"  [TP] Kaydedildi: {tp_fname}")
            else:
                if len(fp_candidates) < 2 or det.conf > fp_candidates[-1][0]:
                    anchor_idx = cam_engine.find_best_anchor(det.xyxy_lb, anchor_xyxy)
                    extracted = cam_engine.backward_and_extract(
                        raw_out=raw_out_tr,
                        anchor_idx=anchor_idx,
                        class_id=det.class_id,
                        lb=lb,
                        methods=(method,),
                    )
                    _, cam_orig = extracted[layer_idx][1][method]
                    single_fp = render_single_method_panel(
                        img_bgr=img_bgr,
                        cam_orig=cam_orig,
                        box_orig=det.xyxy_orig,
                        title_str=(
                            f"[{method.upper()}@{layer_idx} - FP] {det.stem} | "
                            f"imgsz={imgsz} (GT karşılığı yok)"
                        ),
                        label_str=f"FP: {det.class_name} {det.conf:.2f}",
                        box_color_rgb=(255, 50, 50),
                    )
                    fp_candidates.append((det.conf, single_fp, f"{det.class_name}_{det.stem}.png"))
                    fp_candidates.sort(key=lambda x: -x[0])
                    fp_candidates = fp_candidates[:2]

        if len(fn_panels) < 2:
            for g_idx, g in enumerate(gts):
                if g_idx in matched_gt_indices:
                    continue
                fn_anchor_idx = cam_engine.find_best_anchor(g.xyxy_lb, anchor_xyxy)
                fn_raw_score = float(raw_out_tr[0, 4 + g.class_id, fn_anchor_idx].item())
                extracted = cam_engine.backward_and_extract(
                    raw_out=raw_out_tr,
                    anchor_idx=fn_anchor_idx,
                    class_id=g.class_id,
                    lb=lb,
                    methods=(method,),
                )
                _, cam_orig = extracted[layer_idx][1][method]
                fn_idx = len(fn_panels) + 1
                single_fn = render_single_method_panel(
                    img_bgr=img_bgr,
                    cam_orig=cam_orig,
                    box_orig=g.xyxy_orig,
                    title_str=(
                        f"[{method.upper()}@{layer_idx} - FN] {img_path.stem} | "
                        f"imgsz={imgsz} (En yakın anchor skoru={fn_raw_score:.3f})"
                    ),
                    label_str=f"GT (FN): {g.class_name}",
                    box_color_rgb=(255, 200, 0),
                )
                fn_fname = f"fn_{fn_idx}_{g.class_name}_{img_path.stem}.png"
                save_rgb_image(single_fn, out_dir / fn_fname)
                fn_panels.append((single_fn, fn_fname))
                print(f"  [FN] Kaydedildi: {fn_fname}")
                if len(fn_panels) >= 2:
                    break

    cam_engine.close()

    fp_saved_panels: list[np.ndarray] = []
    for idx, (_, single_fp, suffix) in enumerate(fp_candidates, start=1):
        fp_fname = f"fp_{idx}_{suffix}"
        save_rgb_image(single_fp, out_dir / fp_fname)
        fp_saved_panels.append(single_fp)
        print(f"  [FP] Kaydedildi: {fp_fname}")

    if tp_single_panels:
        fig_tp, axes_tp = plt.subplots(3, 2, figsize=(18, 16))
        for ax, cls_id in zip(axes_tp.flatten(), sorted(tp_single_panels.keys())):
            ax.imshow(tp_single_panels[cls_id])
            ax.axis("off")
        plt.suptitle(
            f"{method.upper()}@{layer_idx} Doğru Tespit Örnekleri (6 Sınıf, val) — {model_label} (imgsz={imgsz})",
            fontsize=15,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(out_dir / "gradcam_tp_6classes.png", dpi=130, bbox_inches="tight")
        plt.close(fig_tp)
        print(f"  [GRID] Kaydedildi: {out_dir / 'gradcam_tp_6classes.png'}")

    err_panels = fp_saved_panels + [p for p, _ in fn_panels]
    if err_panels:
        n_rows = 2
        n_cols = 2
        fig_err, axes_err = plt.subplots(n_rows, n_cols, figsize=(18, 11))
        flat_axes = axes_err.flatten()
        for ax_i, ax in enumerate(flat_axes):
            if ax_i < len(err_panels):
                ax.imshow(err_panels[ax_i])
            ax.axis("off")
        plt.suptitle(
            f"{method.upper()}@{layer_idx} Hata Analizi (Üst: 2 FP, Alt: 2 FN, val) — {model_label} (imgsz={imgsz})",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(out_dir / "gradcam_errors_fp_fn.png", dpi=130, bbox_inches="tight")
        plt.close(fig_err)
        print(f"  [GRID] Kaydedildi: {out_dir / 'gradcam_errors_fp_fn.png'}")


def write_xai_summary_md(sanity_csv: Path, out_md: Path) -> None:
    """results/gradcam_sanity.csv dosyasından rakamları doğrudan okuyarak results/xai_summary.md raporunu üretir."""
    if not sanity_csv.is_file():
        raise FileNotFoundError(f"CSV bulunamadı: {sanity_csv}")

    df = pd.read_csv(sanity_csv)
    by_key: dict[tuple[int, str], dict[str, object]] = {}
    for r in df.to_dict(orient="records"):
        by_key[(int(r["katman"]), str(r["method"]))] = r

    # Tablo 1 satırları
    t1_lines = [
        "| Katman | Yöntem | $n$ | Eğitilmiş Model Pointing (`kendi_kutu`) | Rastgele Model Pointing (`pointing_rastgele_model`) | Şans Tabanı (`şans_kendi_kutu`) | Pencere Alan Oranı | Pencere İçi Randomizasyon Spearman (Medyan) | Pencere İçi Enerji Kutuda | Taban (`Kutu ∩ Pencere / Pencere`) |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for lid in (4, 16):
        for m in ("layercam", "hirescam", "gradcam"):
            r = by_key[(lid, m)]
            pg_tr = float(r["kendi_kutu_tum_goruntu"])
            pg_rn = float(r["pointing_rastgele_model"])
            pg_ch = float(r["şans_kendi_kutu_tum_goruntu"])
            w_area = float(r["pencere_alan_orani"])
            sp_w = float(r["randomizasyon_spearman_medyan_pencere"])
            en_w = float(r["enerji_kutuda_pencere"])
            bs_w = float(r["taban_kutu_pencere"])
            n_det = int(r["n_tespit"])
            is_default = (lid == 4 and m == "layercam")
            m_disp = f"**`{m}` (Varsayılan)**" if is_default else f"`{m}`"
            t1_lines.append(
                f"| **{lid}** | {m_disp} | {n_det} | "
                f"{pg_tr:.4f} (%{pg_tr * 100:.2f}) | "
                f"{pg_rn:.4f} (%{pg_rn * 100:.2f}) | "
                f"{pg_ch:.6f} (%{pg_ch * 100:.2f}) | "
                f"{w_area:.4f} (%{w_area * 100:.2f}) | "
                f"{sp_w:.4f} | "
                f"{en_w:.4f} (%{en_w * 100:.2f}) | "
                f"{bs_w:.4f} (%{bs_w * 100:.2f}) |"
            )

    # Tablo 2 satırları (yalnızca silme testi uygulanan kombinasyonlar; oran sütunu YOK)
    t2_lines = [
        "| Katman | Yöntem | $1\\times$ Harita Düşüş (Ort / Med) | $1\\times$ Kontrol Düşüş (Ort / Med) | $1\\times$ Kazanma Oranı | $2\\times$ Harita Düşüş (Ort / Med) | $2\\times$ Kontrol Düşüş (Ort / Med) | $2\\times$ Kazanma Oranı |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for lid, m, note in ((4, "layercam", "Varsayılan"), (4, "hirescam", "Karşılaştırma"), (16, "layercam", "Referans")):
        r = by_key[(lid, m)]
        m1_ort = float(r["goreli_dusus_harita_ort_1x"])
        m1_med = float(r["goreli_dusus_harita_med_1x"])
        c1_ort = float(r["goreli_dusus_kontrol_ort_1x"])
        c1_med = float(r["goreli_dusus_kontrol_med_1x"])
        if abs(c1_med) < 1e-9:
            c1_med = 0.0
        w1 = float(r["goreli_kazanma_orani_1x"])

        m2_ort = float(r["goreli_dusus_harita_ort_2x"])
        m2_med = float(r["goreli_dusus_harita_med_2x"])
        c2_ort = float(r["goreli_dusus_kontrol_ort_2x"])
        c2_med = float(r["goreli_dusus_kontrol_med_2x"])
        if abs(c2_med) < 1e-9:
            c2_med = 0.0
        w2 = float(r["goreli_kazanma_orani_2x"])

        m_disp = f"**`{m}` ({note})**" if (lid == 4 and m == "layercam") else f"`{m}` ({note})"
        t2_lines.append(
            f"| **{lid}** | {m_disp} | "
            f"{m1_ort:.4f} / {m1_med:.4f} | "
            f"{c1_ort:.4f} / {c1_med:.4f} | "
            f"**{w1:.4f} (%{w1 * 100:.2f})** | "
            f"{m2_ort:.4f} / {m2_med:.4f} | "
            f"{c2_ort:.4f} / {c2_med:.4f} | "
            f"**{w2:.4f} (%{w2 * 100:.2f})** |"
        )

    r4_lc = by_key[(4, "layercam")]
    r4_hr = by_key[(4, "hirescam")]
    r4_gc = by_key[(4, "gradcam")]
    r16_lc = by_key[(16, "layercam")]
    r16_gc = by_key[(16, "gradcam")]

    # CSV'den dinamik 5 cümlelik yorum (KARARLAR.md 2026-11-12 XAI raporlama kuralları a, b, c, d + seçim gerekçesi)
    s1 = (
        f"1. **Mutlak Bütçeli Yapısal Silme Sadakati (Kural a):** Geniş gradyan desteğine sahip backbone Katman 4'te "
        f"(pencere alanı görüntünün %{float(r4_lc['pencere_alan_orani']) * 100:.2f}'i), GT kutu alanının $1\\times$ ve $2\\times$ "
        f"bütçeleriyle haritanın en yüksek pikselleri bulanıklaştırıldığında `layercam` göreli skor düşüşü ortalamada "
        f"**{float(r4_lc['goreli_dusus_harita_ort_1x']):.4f}** ($1\\times$, medyan **{float(r4_lc['goreli_dusus_harita_med_1x']):.4f}**) ve "
        f"**{float(r4_lc['goreli_dusus_harita_ort_2x']):.4f}** ($2\\times$, medyan **{float(r4_lc['goreli_dusus_harita_med_2x']):.4f}**) "
        f"olurken, aynı maskenin pencere içinde rastgele kaydırıldığı yapısal kontrolde düşüş sıfıra yakın "
        f"(**{float(r4_lc['goreli_dusus_kontrol_ort_1x']):.4f}** ve **{float(r4_lc['goreli_dusus_kontrol_ort_2x']):.4f}**) kalmakta; "
        f"böylece sıfıra yakın payda nedeniyle yapay oranlar yerine doğrudan raporlanan kazanma oranı `layercam` için "
        f"**%{float(r4_lc['goreli_kazanma_orani_1x']) * 100:.2f}** ($1\\times$) / **%{float(r4_lc['goreli_kazanma_orani_2x']) * 100:.2f}** ($2\\times$) ve "
        f"`hirescam` için **%{float(r4_hr['goreli_kazanma_orani_1x']) * 100:.2f}** ($1\\times$) / **%{float(r4_hr['goreli_kazanma_orani_2x']) * 100:.2f}** ($2\\times$) "
        f"düzeyine ulaşmaktadır."
    )
    s2 = (
        f"2. **Varsayılan Yöntem Seçimi (`LayerCAM @ Katman 4`):** Katman 4 silme testinde `layercam` ve `hirescam` "
        f"sadakat açısından eşdeğer performans gösterse de (`1x` harita düşüş ortalaması **{float(r4_lc['goreli_dusus_harita_ort_1x']):.4f}** vs. "
        f"**{float(r4_hr['goreli_dusus_harita_ort_1x']):.4f}**), `layercam` pozitif gradyan kapılaması sayesinde pencere içi harita enerjisinin "
        f"**%{float(r4_lc['enerji_kutuda_pencere']) * 100:.2f}**'unu doğrudan GT kutusu içine odaklamakta (`hirescam`: "
        f"**%{float(r4_hr['enerji_kutuda_pencere']) * 100:.2f}**, düzgün harita tabanı: **%{float(r4_lc['taban_kutu_pencere']) * 100:.2f}**) "
        f"ve daha keskin yerelleşme sağladığı için varsayılan XAI yöntemi olarak kilitlenmiştir."
    )
    s3 = (
        f"3. **Pointing Game Üçlüsü ve Mimari Katkı (Kural b):** Katman 4'te `layercam` kendi GT kutusuna pointing testinde "
        f"**%{float(r4_lc['kendi_kutu_tum_goruntu']) * 100:.2f}** (`hirescam`: **%{float(r4_hr['kendi_kutu_tum_goruntu']) * 100:.2f}**) "
        f"başarı sağlarken, rastgele ağırlıklı aynı mimarinin **%{float(r4_lc['pointing_rastgele_model']) * 100:.2f}** "
        f"(`hirescam`: **%{float(r4_hr['pointing_rastgele_model']) * 100:.2f}**) ve şans tabanının "
        f"**%{float(r4_lc['şans_kendi_kutu_tum_goruntu']) * 100:.2f}** çıkması, anchor merkezli etkin alıcı alanın (effective receptive field) "
        f"mimari bir temel oluşturduğunu ancak öğrenilmiş ağırlıkların isabet oranını iki katına çıkardığını (pencere içi randomizasyon Spearman medyanı "
        f"**{float(r4_lc['randomizasyon_spearman_medyan_pencere']):.4f}**) dürüstçe göstermektedir."
    )
    s4 = (
        f"4. **Katman 16 Referans Bulgusu ve Dar Alıcı Alan Kısıtı (Kural c):** Boyun çıkışındaki referans Katman 16'da "
        f"tek bir anchor'ın gradyan desteği görüntünün yalnızca **%{float(r16_lc['pencere_alan_orani']) * 100:.2f}**'ünü ($5\\times 5$ hücre) "
        f"kapsadığından, pencere içinde kaydırılan kontrol maskesi de GT kutusuyla örtüşerek yüksek skor düşüşüne "
        f"($1\\times$ kontrol ort. **{float(r16_lc['goreli_dusus_kontrol_ort_1x']):.4f}**, $2\\times$ kontrol ort. "
        f"**{float(r16_lc['goreli_dusus_kontrol_ort_2x']):.4f}**) neden olmakta ve bu sığ alıcı alan yüzünden Katman 16 sadakat testi için "
        f"uygun bulunmamaktadır."
    )
    s5 = (
        f"5. **Neden Klasik Grad-CAM Değil? (Kural d):** Klasik `gradcam`, gradyanların uzamsal ortalamasını alarak konum bilgisini "
        f"yok ettiği için çoklu kusur barındıran PCB kartlarında kendi kutusuna pointing oranında (**%{float(r4_gc['kendi_kutu_tum_goruntu']) * 100:.2f}** "
        f"@ Katman 4, **%{float(r16_gc['kendi_kutu_tum_goruntu']) * 100:.2f}** @ Katman 16) ve pencere içi kutu enerjisinde "
        f"(Katman 4'te **%{float(r4_gc['enerji_kutuda_pencere']) * 100:.2f}** vs. taban **%{float(r4_gc['taban_kutu_pencere']) * 100:.2f}**) "
        f"rastgele düzeyde kalmakta, dolayısıyla tespit bazlı açıklama için kullanılamamaktadır."
    )

    md_content = "\n".join(
        [
            "# Tracera — Açıklanabilirlik (XAI) Doğrulama ve Sadakat Özeti",
            "",
            "- **Kilitlenen Varsayılan Konfigürasyon:** `LayerCAM @ Katman 4` (`--method layercam --layer 4`)",
            f"- **Kaynak Veri:** `results/gradcam_sanity.csv` (`model={df['model'].iloc[0]}`, `imgsz={int(df['imgsz'].iloc[0])}`, `split=val`, `n_tespit={int(df['n_tespit'].iloc[0])}`, `seed=42`)",
            "",
            "## 1. Pointing Game Üçlüsü (Eğitilmiş / Rastgele Model / Şans) ve Yerel Pencere Metrikleri",
            "",
            *t1_lines,
            "",
            "## 2. Mutlak Bütçeli ($1\\times$ ve $2\\times$ GT Kutu Alanı) Yapısal Kaydırma Kontrollü Silme Testi — Göreli Skor Düşüşü $(s_0 - s_1)/s_0$",
            "",
            *t2_lines,
            "",
            "## 3. Bulgular ve Raporlama Kuralları Özeti (5 Cümle)",
            "",
            s1,
            s2,
            s3,
            s4,
            s5,
            "",
        ]
    )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md_content, encoding="utf-8")
    print(f"[KAYDEDİLDİ] XAI Özet Raporu: {out_md}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tracera Yerel CAM Doğrulama Protokolü, Yapısal Silme Testi ve Val Görsel Üretimi [CPU]"
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
        choices=["val", "train"],
        default="val",
        help="Değerlendirilecek veri bölmesi — test KULLANILAMAZ (varsayılan: val)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["gradcam", "hirescam", "layercam"],
        default="layercam",
        help="Kilitli varsayılan CAM yöntemi (varsayılan: layercam)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        nargs="+",
        default=[4],
        help="Kilitli varsayılan hedef katman (varsayılan: 4)",
    )
    parser.add_argument(
        "--max-dets",
        type=int,
        default=150,
        help="Rastgele örneklenecek maksimum doğru tespit sayısı (seed 42; varsayılan: 150)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Rastgelelik tohumu (varsayılan: 42)",
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
    parser.add_argument(
        "--visuals-only",
        action="store_true",
        help="Yalnızca val setinde 6 TP + 2 FP + 2 FN görsellerini üret (gradcam_sanity.csv'ye dokunma)",
    )
    parser.add_argument(
        "--write-summary-only",
        action="store_true",
        help="Yalnızca mevcut results/gradcam_sanity.csv'den results/xai_summary.md dosyasını üret",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sanity_csv = REPO_ROOT / "results" / "gradcam_sanity.csv"
    summary_md = REPO_ROOT / "results" / "xai_summary.md"

    if args.write_summary_only:
        write_xai_summary_md(sanity_csv, summary_md)
        return

    assert args.split != "test", "XAI analiz ve görsellerinde test seti KULLANILAMAZ!"

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
    target_layers: list[int] = list(args.layer) if isinstance(args.layer, list) else [int(args.layer)]
    primary_layer = int(target_layers[0])

    model_label = (
        f"{weights_path.parent.parent.name}/{weights_path.name}"
        if weights_path.parent.name == "weights"
        else weights_path.stem
    )

    if args.visuals_only:
        generate_val_visuals(
            yolo_obj=yolo_obj,
            img_dir=img_dir,
            lbl_dir=lbl_dir,
            class_names=class_names,
            model_label=model_label,
            imgsz=args.imgsz,
            method=args.method,
            layer_idx=primary_layer,
            conf_thres=args.conf,
            iou_thres=args.iou,
            split=args.split,
        )
        return

    trained_nn = copy.deepcopy(yolo_obj.model).to("cpu").eval()
    eval_nn = copy.deepcopy(yolo_obj.model).fuse().to("cpu").eval()

    detect_head = trained_nn.model[-1]
    detect_f: list[int] = list(getattr(detect_head, "f", [16, 19, 22]))

    csv_model_label = (
        weights_path.parent.parent.name
        if weights_path.parent.name == "weights"
        else weights_path.stem
    )

    print("=" * 96)
    print(
        f"TRACERA MUTLAK BÜTÇELİ (1x/2x) YAPISAL SİLME TESTİ + RASTGELE MODEL POINTING (CPU)\n"
        f"model={csv_model_label} | imgsz={args.imgsz} | max_dets={args.max_dets} | aktif_silme={sorted(DELETION_ACTIVE_COMBOS)}"
    )
    print(f"Detect modülü .f listesi : {detect_f}")
    print("=" * 96)

    img_paths = sorted(
        [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=lambda p: p.name,
    )

    all_tps: list[SampledTP] = []
    img_chance_own_list: list[float] = []
    img_chance_any_list: list[float] = []

    with torch.inference_mode():
        for img_path in img_paths:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            lb = letterbox_image(img_bgr, imgsz=args.imgsz)
            gts = load_gt_boxes(lbl_dir / f"{img_path.stem}.txt", lb=lb, class_names=class_names)
            img_area = float(lb.orig_w * lb.orig_h)
            if gts:
                gt_areas = [
                    max(0.0, (g.xyxy_orig[2] - g.xyxy_orig[0]) * (g.xyxy_orig[3] - g.xyxy_orig[1]))
                    for g in gts
                ]
                img_chance_own_list.append(float(np.mean(gt_areas)) / img_area)
                img_chance_any_list.append(float(np.sum(gt_areas)) / img_area)

            x_t = (
                torch.from_numpy(lb.img_lb[:, :, ::-1].transpose(2, 0, 1).copy())
                .float()
                .unsqueeze(0)
                / 255.0
            )
            raw_pred = eval_nn(x_t)[0]
            nms_preds = non_max_suppression(raw_pred.clone(), conf_thres=args.conf, iou_thres=args.iou)[0]
            preds_np = nms_preds.cpu().numpy() if len(nms_preds) > 0 else np.zeros((0, 6), dtype=np.float32)

            matched_gt_indices: set[int] = set()
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
                    all_tps.append(
                        SampledTP(
                            img_path=img_path,
                            stem=img_path.stem,
                            class_id=cls_id,
                            class_name=class_names.get(cls_id, str(cls_id)),
                            conf=float(conf_val),
                            xyxy_orig=xyxy_orig,
                            xyxy_lb=xyxy_lb,
                            matched_gt=best_gt,
                            all_gts=gts,
                            iou_with_gt=best_iou,
                        )
                    )

    py_rng = random.Random(args.seed)
    if args.max_dets > 0 and len(all_tps) > args.max_dets:
        sampled_tps = py_rng.sample(all_tps, args.max_dets)
        sampled_tps.sort(key=lambda s: (s.stem, -s.conf))
    else:
        sampled_tps = all_tps

    n_eval = len(sampled_tps)
    print(f"Toplam doğru tespit: {len(all_tps)} | Örneklenen tespit (seed={args.seed}): {n_eval}")

    rand_nn = build_randomized_model(yolo_obj.model, seed=args.seed)
    cam_trained = MultiLayerYoloCAM(trained_nn, layer_indices=target_layers, imgsz=args.imgsz)
    cam_rand = MultiLayerYoloCAM(rand_nn, layer_indices=target_layers, imgsz=args.imgsz)

    hits_own_full: dict[tuple[int, str], int] = {(l, m): 0 for l in target_layers for m in METHODS}
    hits_any_full: dict[tuple[int, str], int] = {(l, m): 0 for l in target_layers for m in METHODS}
    hits_own_rand_model: dict[tuple[int, str], int] = {(l, m): 0 for l in target_layers for m in METHODS}
    spearman_full: dict[tuple[int, str], list[float]] = {(l, m): [] for l in target_layers for m in METHODS}
    spearman_win: dict[tuple[int, str], list[float]] = {(l, m): [] for l in target_layers for m in METHODS}
    energy_in_gt_win: dict[tuple[int, str], list[float]] = {(l, m): [] for l in target_layers for m in METHODS}
    base_in_gt_win: dict[tuple[int, str], list[float]] = {(l, m): [] for l in target_layers for m in METHODS}
    win_area_ratios: dict[int, list[float]] = {l: [] for l in target_layers}

    rel_del_records: dict[tuple[int, str, str], list[tuple[float, float]]] = {
        (l, m, b): [] for (l, m) in DELETION_ACTIVE_COMBOS for b in ("1x", "2x")
    }

    np_rng = np.random.default_rng(args.seed)
    by_img: dict[Path, list[SampledTP]] = {}
    for item in sampled_tps:
        by_img.setdefault(item.img_path, []).append(item)

    processed = 0
    for img_path, items in by_img.items():
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        lb = letterbox_image(img_bgr, imgsz=args.imgsz)
        blurred_lb = cv2.GaussianBlur(lb.img_lb, ksize=(0, 0), sigmaX=5.0, sigmaY=5.0)

        x_tensor = (
            torch.from_numpy(lb.img_lb[:, :, ::-1].transpose(2, 0, 1).copy())
            .float()
            .unsqueeze(0)
            / 255.0
        )

        raw_out_tr, anchor_xyxy = cam_trained.forward_image(x_tensor)
        raw_out_rn, _ = cam_rand.forward_image(x_tensor)

        for sample in items:
            processed += 1
            anchor_idx = cam_trained.find_best_anchor(sample.xyxy_lb, anchor_xyxy)
            orig_score = float(raw_out_tr[0, 4 + sample.class_id, anchor_idx].item())

            tr_extracted = cam_trained.backward_and_extract(
                raw_out=raw_out_tr,
                anchor_idx=anchor_idx,
                class_id=sample.class_id,
                lb=lb,
            )
            rn_extracted = cam_rand.backward_and_extract(
                raw_out=raw_out_rn,
                anchor_idx=anchor_idx,
                class_id=sample.class_id,
                lb=lb,
            )

            gx1, gy1, gx2, gy2 = sample.matched_gt.xyxy_orig

            for lid in target_layers:
                win, cams_tr = tr_extracted[lid]
                _, cams_rn = rn_extracted[lid]
                win_area_ratios[lid].append(win.area_ratio)

                for m_name in METHODS:
                    cam_lb_tr, cam_orig_tr = cams_tr[m_name]
                    _, cam_orig_rn = cams_rn[m_name]

                    max_y, max_x = np.unravel_index(int(np.argmax(cam_orig_tr)), cam_orig_tr.shape)
                    if gx1 <= max_x <= gx2 and gy1 <= max_y <= gy2:
                        hits_own_full[(lid, m_name)] += 1
                    if any(
                        g.xyxy_orig[0] <= max_x <= g.xyxy_orig[2] and g.xyxy_orig[1] <= max_y <= g.xyxy_orig[3]
                        for g in sample.all_gts
                    ):
                        hits_any_full[(lid, m_name)] += 1

                    rmax_y, rmax_x = np.unravel_index(int(np.argmax(cam_orig_rn)), cam_orig_rn.shape)
                    if gx1 <= rmax_x <= gx2 and gy1 <= rmax_y <= gy2:
                        hits_own_rand_model[(lid, m_name)] += 1

                    rho_full = compute_spearman_safe(cam_orig_tr, cam_orig_rn, max_side=160)
                    spearman_full[(lid, m_name)].append(rho_full)

                    tr_win_slice = cam_orig_tr[win.orig_y1 : win.orig_y2, win.orig_x1 : win.orig_x2]
                    rn_win_slice = cam_orig_rn[win.orig_y1 : win.orig_y2, win.orig_x1 : win.orig_x2]
                    rho_win = compute_spearman_safe(tr_win_slice, rn_win_slice, max_side=160)
                    spearman_win[(lid, m_name)].append(rho_win)

                    e_ratio, b_ratio = compute_window_energy_in_gt(
                        cam_orig_tr,
                        win=win,
                        gt_box_orig=sample.matched_gt.xyxy_orig,
                    )
                    energy_in_gt_win[(lid, m_name)].append(e_ratio)
                    base_in_gt_win[(lid, m_name)].append(b_ratio)

                    if (lid, m_name) in DELETION_ACTIVE_COMBOS:
                        del_res = evaluate_absolute_budget_deletion(
                            eval_model=eval_nn,
                            img_lb=lb.img_lb,
                            blurred_lb=blurred_lb,
                            win=win,
                            cam_lb=cam_lb_tr,
                            gt_box_lb=sample.matched_gt.xyxy_lb,
                            anchor_idx=anchor_idx,
                            class_id=sample.class_id,
                            orig_score=orig_score,
                            rng=np_rng,
                            n_repeats=5,
                        )
                        for b_key in ("1x", "2x"):
                            rel_del_records[(lid, m_name, b_key)].append(del_res[b_key])

        if processed % 30 == 0 or processed == n_eval:
            print(f"  İlerleme: {processed}/{n_eval} tespit tamamlandı...")

    cam_trained.close()
    cam_rand.close()

    chance_own_full = float(np.mean(img_chance_own_list)) if img_chance_own_list else 0.0
    chance_any_full = float(np.mean(img_chance_any_list)) if img_chance_any_list else 0.0

    prev_map: dict[tuple[int, str], dict[str, object]] = {}
    if sanity_csv.is_file():
        df_prev = pd.read_csv(sanity_csv)
        for r in df_prev.to_dict(orient="records"):
            prev_map[(int(r.get("katman", -1)), str(r.get("method", "")))] = r

    rows: list[dict[str, object]] = []
    for lid in target_layers:
        mean_win_area = float(np.mean(win_area_ratios[lid])) if win_area_ratios[lid] else 0.0
        for m_name in METHODS:
            key = (lid, m_name)
            pg_own_f = hits_own_full[key] / n_eval if n_eval > 0 else 0.0
            pg_any_f = hits_any_full[key] / n_eval if n_eval > 0 else 0.0
            pg_rand_m = hits_own_rand_model[key] / n_eval if n_eval > 0 else 0.0
            sp_full_med = float(np.median(spearman_full[key])) if spearman_full[key] else 0.0
            sp_win_med = float(np.median(spearman_win[key])) if spearman_win[key] else 0.0
            e_win_mean = float(np.mean(energy_in_gt_win[key])) if energy_in_gt_win[key] else 0.0
            b_win_mean = float(np.mean(base_in_gt_win[key])) if base_in_gt_win[key] else 0.0

            row: dict[str, object] = {
                "model": csv_model_label,
                "imgsz": args.imgsz,
                "katman": lid,
                "method": m_name,
                "n_tespit": n_eval,
                "kendi_kutu_tum_goruntu": round(pg_own_f, 4),
                "pointing_rastgele_model": round(pg_rand_m, 4),
                "şans_kendi_kutu_tum_goruntu": round(chance_own_full, 6),
                "herhangi_kutu_tum_goruntu": round(pg_any_f, 4),
                "şans_herhangi_kutu_tum_goruntu": round(chance_any_full, 6),
                "randomizasyon_spearman_medyan_tum_goruntu": round(sp_full_med, 4),
                "pencere_alan_orani": round(mean_win_area, 4),
                "randomizasyon_spearman_medyan_pencere": round(sp_win_med, 4),
                "enerji_kutuda_pencere": round(e_win_mean, 4),
                "taban_kutu_pencere": round(b_win_mean, 4),
            }

            if key in DELETION_ACTIVE_COMBOS:
                for b_key in ("1x", "2x"):
                    recs = rel_del_records[(lid, m_name, b_key)]
                    map_drops = np.asarray([x[0] for x in recs], dtype=np.float64)
                    ctrl_drops = np.asarray([x[1] for x in recs], dtype=np.float64)
                    wins = float(np.mean(map_drops > ctrl_drops)) if len(recs) > 0 else 0.0
                    mean_m = float(np.mean(map_drops))
                    mean_c = float(np.mean(ctrl_drops))
                    med_m = float(np.median(map_drops))
                    med_c = float(np.median(ctrl_drops))

                    row[f"goreli_dusus_harita_ort_{b_key}"] = round(mean_m, 4)
                    row[f"goreli_dusus_harita_med_{b_key}"] = round(med_m, 4)
                    row[f"goreli_dusus_kontrol_ort_{b_key}"] = round(mean_c, 4)
                    row[f"goreli_dusus_kontrol_med_{b_key}"] = round(med_c, 4)
                    row[f"goreli_kazanma_orani_{b_key}"] = round(wins, 4)
            else:
                old_r = prev_map.get(key, {})
                for b_key in ("1x", "2x"):
                    row[f"goreli_dusus_harita_ort_{b_key}"] = old_r.get(f"goreli_dusus_harita_ort_{b_key}", float("nan"))
                    row[f"goreli_dusus_harita_med_{b_key}"] = old_r.get(f"goreli_dusus_harita_med_{b_key}", float("nan"))
                    row[f"goreli_dusus_kontrol_ort_{b_key}"] = old_r.get(f"goreli_dusus_kontrol_ort_{b_key}", float("nan"))
                    row[f"goreli_dusus_kontrol_med_{b_key}"] = old_r.get(f"goreli_dusus_kontrol_med_{b_key}", float("nan"))
                    row[f"goreli_kazanma_orani_{b_key}"] = old_r.get(f"goreli_kazanma_orani_{b_key}", float("nan"))

            rows.append(row)

    df_sanity = pd.DataFrame(rows)
    sanity_csv.parent.mkdir(parents=True, exist_ok=True)
    df_sanity.to_csv(sanity_csv, index=False, encoding="utf-8")

    if len(df_sanity) >= 6:
        write_xai_summary_md(sanity_csv, summary_md)


if __name__ == "__main__":
    main()
