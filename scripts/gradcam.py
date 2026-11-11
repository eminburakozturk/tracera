"""Tracera — Tespit bazlı CAM (Grad-CAM, HiResCAM, LayerCAM) yerel doğrulama protokolü ve mutlak bütçeli (1x/2x GT kutu alanı)
yapısal kaydırmalı silme (deletion) sadakat testi.

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
                for m in METHODS
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
    """2B ikili maskeyi pencere (wh, ww) içinde rastgele kaydırır (tamamen pencere içinde kalacak şekilde).

    Aktif piksellerin sınırlayıcı kutusu pencereden küçükse kesmesiz katı öteleme (rigid translation) uygular;
    aktif pikseller pencerenin tüm kenarlarına değiyorsa pencere içi dairesel kaydırma (roll) uygular.
    """
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

    # Aktif pikseller pencerenin uçlarına yayıldığı için katı öteleme alanı yoksa pencere içi sıfırdan farklı kaydırma
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

        # 1. Harita maskesi
        img_map = img_lb.copy()
        img_map[y1:y2, x1:x2] = np.where(mask_2d[:, :, None], blur_patch, orig_patch)
        perturbed_tensors.append(
            torch.from_numpy(img_map[:, :, ::-1].transpose(2, 0, 1).copy()).float() / 255.0
        )

        # 2. Yapısal kontrol: aynı ikili maskenin pencere içinde 5 kez rastgele kaydırılmış hali
        for _ in range(n_repeats):
            shifted_2d = shift_binary_mask_within_window(mask_2d, rng=rng)
            img_ctrl = img_lb.copy()
            img_ctrl[y1:y2, x1:x2] = np.where(shifted_2d[:, :, None], blur_patch, orig_patch)
            perturbed_tensors.append(
                torch.from_numpy(img_ctrl[:, :, ::-1].transpose(2, 0, 1).copy()).float() / 255.0
            )

    batch_tensor = torch.stack(perturbed_tensors, dim=0)  # (2 * (1 + n_repeats), 3, imgsz, imgsz)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tracera Yerel CAM Doğrulama Protokolü + Mutlak Bütçeli (1x/2x) Yapısal Silme Testi [CPU]"
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
        "--method",
        type=str,
        choices=["gradcam", "hirescam", "layercam"],
        default="layercam",
        help="Birincil CAM yöntemi (varsayılan: layercam)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        nargs="+",
        default=[16, 4],
        help="Değerlendirilecek hedef katman(lar) (varsayılan: 16 4)",
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
    eval_nn = copy.deepcopy(yolo_obj.model).fuse().to("cpu").eval()

    detect_head = trained_nn.model[-1]
    detect_f: list[int] = list(getattr(detect_head, "f", [16, 19, 22]))
    target_layers: list[int] = list(args.layer) if isinstance(args.layer, list) else [int(args.layer)]

    model_label = (
        weights_path.parent.parent.name
        if weights_path.parent.name == "weights"
        else weights_path.stem
    )

    print("=" * 96)
    print(
        f"TRACERA MUTLAK BÜTÇELİ (1x/2x) YAPISAL SİLME TESTİ + RASTGELE MODEL POINTING (CPU)\n"
        f"model={model_label} | imgsz={args.imgsz} | max_dets={args.max_dets} | aktif_silme={sorted(DELETION_ACTIVE_COMBOS)}"
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

    # Göreli silme düşüşleri: key = (lid, method, budget_str) -> list of (rel_drop_map, rel_drop_ctrl)
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

                    # Eğitilmiş model pointing game (tüm görüntü)
                    max_y, max_x = np.unravel_index(int(np.argmax(cam_orig_tr)), cam_orig_tr.shape)
                    if gx1 <= max_x <= gx2 and gy1 <= max_y <= gy2:
                        hits_own_full[(lid, m_name)] += 1
                    if any(
                        g.xyxy_orig[0] <= max_x <= g.xyxy_orig[2] and g.xyxy_orig[1] <= max_y <= g.xyxy_orig[3]
                        for g in sample.all_gts
                    ):
                        hits_any_full[(lid, m_name)] += 1

                    # Rastgele ağırlıklı model kendi_kutu pointing game (tüm görüntü)
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

                    # Mutlak bütçeli (1x ve 2x GT kutu alanı) yapısal kaydırmalı silme testi
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

    sanity_csv = REPO_ROOT / "results" / "gradcam_sanity.csv"
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
                "model": model_label,
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
                    ratio_mean = mean_m / mean_c if abs(mean_c) > 1e-9 else float("nan")

                    row[f"goreli_dusus_harita_ort_{b_key}"] = round(mean_m, 4)
                    row[f"goreli_dusus_harita_med_{b_key}"] = round(med_m, 4)
                    row[f"goreli_dusus_kontrol_ort_{b_key}"] = round(mean_c, 4)
                    row[f"goreli_dusus_kontrol_med_{b_key}"] = round(med_c, 4)
                    row[f"goreli_oran_{b_key}"] = round(ratio_mean, 2)
                    row[f"goreli_kazanma_orani_{b_key}"] = round(wins, 4)
            else:
                # Önceki sonuçları koru, yeni 1x/2x sütunlarına NaN yaz
                old_r = prev_map.get(key, {})
                for b_key in ("1x", "2x"):
                    row[f"goreli_dusus_harita_ort_{b_key}"] = old_r.get(f"goreli_dusus_harita_ort_{b_key}", float("nan"))
                    row[f"goreli_dusus_harita_med_{b_key}"] = old_r.get(f"goreli_dusus_harita_med_{b_key}", float("nan"))
                    row[f"goreli_dusus_kontrol_ort_{b_key}"] = old_r.get(f"goreli_dusus_kontrol_ort_{b_key}", float("nan"))
                    row[f"goreli_dusus_kontrol_med_{b_key}"] = old_r.get(f"goreli_dusus_kontrol_med_{b_key}", float("nan"))
                    row[f"goreli_oran_{b_key}"] = old_r.get(f"goreli_oran_{b_key}", float("nan"))
                    row[f"goreli_kazanma_orani_{b_key}"] = old_r.get(f"goreli_kazanma_orani_{b_key}", float("nan"))

            rows.append(row)

    df_sanity = pd.DataFrame(rows)
    sanity_csv.parent.mkdir(parents=True, exist_ok=True)
    df_sanity.to_csv(sanity_csv, index=False, encoding="utf-8")

    print("\n" + "=" * 120)
    print("SONUÇ TABLOSU — POINTING GAME (EĞİTİLMİŞ vs RASTGELE MODEL) VE YEREL PENCERE METRİKLERİ")
    print("=" * 120)
    cols_part1 = [
        "katman",
        "method",
        "n_tespit",
        "kendi_kutu_tum_goruntu",
        "pointing_rastgele_model",
        "pencere_alan_orani",
        "randomizasyon_spearman_medyan_pencere",
        "enerji_kutuda_pencere",
        "taban_kutu_pencere",
    ]
    print(df_sanity[cols_part1].to_string(index=False))

    print("\n" + "=" * 120)
    print("SONUÇ TABLOSU — MUTLAK BÜTÇELİ (1x ve 2x KUTU ALANI) YAPISAL KAYDIRMALI SİLME TESTİ (GÖRELİ SKOR DÜŞÜŞÜ)")
    print("=" * 120)
    cols_part2 = [
        "katman",
        "method",
        "goreli_dusus_harita_ort_1x",
        "goreli_dusus_harita_med_1x",
        "goreli_dusus_kontrol_ort_1x",
        "goreli_dusus_kontrol_med_1x",
        "goreli_oran_1x",
        "goreli_kazanma_orani_1x",
        "goreli_dusus_harita_ort_2x",
        "goreli_dusus_harita_med_2x",
        "goreli_dusus_kontrol_ort_2x",
        "goreli_dusus_kontrol_med_2x",
        "goreli_oran_2x",
        "goreli_kazanma_orani_2x",
    ]
    print(df_sanity[cols_part2].to_string(index=False))
    print("=" * 120)
    print(f"Kaydedilen CSV: {sanity_csv}")


if __name__ == "__main__":
    main()
