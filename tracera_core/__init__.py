"""Tracera ortak çekirdek paketi — Gradio bağımlılığı içermez."""

from __future__ import annotations

from tracera_core.config import (
    CLASS_COLORS,
    CLASS_NAMES,
    DEFAULT_CONF,
    DEFAULT_IOU_NMS,
    IMGSZ,
    LETTERBOX_PAD,
    META_PATH,
    OPENVINO_MODEL_PATH,
    PYTORCH_MODEL_PATH,
    REPO_ROOT,
    RESULTS_DIR,
    WEIGHTS_DIR,
    get_repo_root,
    load_meta,
)
from tracera_core.engine import (
    MODELS,
    ModelManager,
    MultiLayerYoloCAM,
    active_detect_backend,
    detect_openvino,
    explain_layercam,
    make_overlay_and_zoom,
)
from tracera_core.metrics_loader import build_metrics_payload
from tracera_core.preprocess import (
    LetterboxTransform,
    box_lb_to_orig,
    box_orig_to_lb,
    letterbox_image,
    preprocess_letterbox,
    preprocess_tensor,
)
from tracera_core.schemas import Detection, DetectResult, ExplainResult, Timing

__all__ = [
    "CLASS_COLORS",
    "CLASS_NAMES",
    "DEFAULT_CONF",
    "DEFAULT_IOU_NMS",
    "IMGSZ",
    "LETTERBOX_PAD",
    "META_PATH",
    "MODELS",
    "OPENVINO_MODEL_PATH",
    "PYTORCH_MODEL_PATH",
    "REPO_ROOT",
    "RESULTS_DIR",
    "WEIGHTS_DIR",
    "DetectResult",
    "Detection",
    "ExplainResult",
    "LetterboxTransform",
    "ModelManager",
    "MultiLayerYoloCAM",
    "Timing",
    "active_detect_backend",
    "box_lb_to_orig",
    "box_orig_to_lb",
    "build_metrics_payload",
    "detect_openvino",
    "explain_layercam",
    "get_repo_root",
    "letterbox_image",
    "load_meta",
    "make_overlay_and_zoom",
    "preprocess_letterbox",
    "preprocess_tensor",
]
