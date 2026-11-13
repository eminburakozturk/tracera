"""Ortak ön işleme sarmalayıcıları (scripts/gradcam.py letterbox mantığını kullanır)."""

from __future__ import annotations

import sys

import numpy as np
import torch

from tracera_core.config import IMGSZ, REPO_ROOT

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gradcam import (  # noqa: E402
    LetterboxTransform,
    box_lb_to_orig,
    box_orig_to_lb,
    letterbox_image,
)


def preprocess_letterbox(img_bgr: np.ndarray, imgsz: int = IMGSZ) -> LetterboxTransform:
    """CONTRACT §1 ile uyumlu oran koruyucu letterbox dönüşümünü uygular."""
    return letterbox_image(img_bgr, imgsz=imgsz)


def preprocess_tensor(
    img_bgr: np.ndarray,
    imgsz: int = IMGSZ,
) -> tuple[torch.Tensor, LetterboxTransform]:
    """BGR görüntüyü letterbox + RGB + /255 + NCHW float32 tensörüne çevirir."""
    lb = preprocess_letterbox(img_bgr, imgsz=imgsz)
    rgb_chw = lb.img_lb[:, :, ::-1].transpose(2, 0, 1).copy()
    tensor_x = torch.from_numpy(rgb_chw).float().unsqueeze(0) / 255.0
    return tensor_x, lb


__all__ = [
    "LetterboxTransform",
    "box_lb_to_orig",
    "box_orig_to_lb",
    "letterbox_image",
    "preprocess_letterbox",
    "preprocess_tensor",
]
