"""Tracera ortak çekirdek sabitleri ve yol çözümleyicileri (Gradio bağımlılığı yoktur)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def get_repo_root() -> Path:
    """Normal Python ve PyInstaller paketlenmiş çalışma için kök dizini döndürür."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent


REPO_ROOT = get_repo_root()
_SPACE_WEIGHTS = REPO_ROOT / "weights"
_RUN_WEIGHTS = REPO_ROOT / "runs" / "detect" / "g_y11s_1024_full" / "weights"
WEIGHTS_DIR = (
    _SPACE_WEIGHTS
    if ((_SPACE_WEIGHTS / "last.pt").is_file() or (_SPACE_WEIGHTS / "last_int8_openvino_model").is_dir())
    else _RUN_WEIGHTS
)
OPENVINO_MODEL_PATH = WEIGHTS_DIR / "last_int8_openvino_model"
PYTORCH_MODEL_PATH = WEIGHTS_DIR / "last.pt"
META_PATH = REPO_ROOT / "web" / "public" / "models" / "meta.json"
RESULTS_DIR = REPO_ROOT / "results"

IMGSZ = 1024
DEFAULT_CONF = 0.395
DEFAULT_IOU_NMS = 0.7
LETTERBOX_PAD = 114

CLASS_NAMES: dict[int, str] = {
    0: "missing_hole",
    1: "mouse_bite",
    2: "open_circuit",
    3: "short",
    4: "spur",
    5: "spurious_copper",
}

CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (230, 25, 75),    # Kırmızımsı
    1: (60, 180, 75),    # Yeşil
    2: (255, 225, 25),   # Sarı
    3: (0, 130, 200),    # Mavi
    4: (245, 130, 48),   # Turuncu
    5: (145, 30, 180),   # Mor
}


def load_meta(meta_path: Path | None = None) -> dict[str, Any]:
    """CONTRACT §1 model sabitlerini meta.json'dan okur; yoksa varsayılanları döndürür."""
    target = meta_path or META_PATH
    if target.is_file():
        return json.loads(target.read_text(encoding="utf-8"))
    return {
        "version": "1",
        "model": "yolo11s",
        "imgsz": IMGSZ,
        "conf": DEFAULT_CONF,
        "iou_nms": DEFAULT_IOU_NMS,
        "letterbox_pad": LETTERBOX_PAD,
        "classes": [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)],
    }


TRACERA_MODEL_REPO_ENV = "TRACERA_MODEL_REPO"


def weights_ready(
    openvino_path: Path | None = None,
    pytorch_path: Path | None = None,
) -> bool:
    """Yerel OpenVINO INT8 ve PyTorch last.pt ağırlıklarının mevcut olup olmadığını doğrular."""
    if os.environ.get("TRACERA_FORCE_NO_WEIGHTS") == "1":
        return False
    ov = openvino_path or OPENVINO_MODEL_PATH
    pt = pytorch_path or PYTORCH_MODEL_PATH
    return pt.is_file() and ov.is_dir() and any(ov.glob("*.xml"))


def ensure_weights_from_hf(
    repo_id: str | None = None,
    weights_dir: Path | None = None,
    openvino_path: Path | None = None,
    pytorch_path: Path | None = None,
) -> bool:
    """Yerel ağırlıklar eksikse TRACERA_MODEL_REPO'dan indirir; mevcut ağırlığa dokunmaz."""
    ov = openvino_path or OPENVINO_MODEL_PATH
    pt = pytorch_path or PYTORCH_MODEL_PATH
    if weights_ready(ov, pt):
        return True
    if os.environ.get("TRACERA_FORCE_NO_WEIGHTS") == "1":
        return False

    target_repo = (repo_id or os.environ.get(TRACERA_MODEL_REPO_ENV, "")).strip()
    if not target_repo:
        return False

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return False

    dst = weights_dir or WEIGHTS_DIR
    try:
        dst.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=target_repo,
            local_dir=str(dst),
            allow_patterns=["last.pt", "last_int8_openvino_model/**"],
        )
    except Exception:
        return False

    return weights_ready(ov, pt)
