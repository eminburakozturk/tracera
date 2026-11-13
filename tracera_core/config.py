"""Tracera ortak çekirdek sabitleri ve yol çözümleyicileri (Gradio bağımlılığı yoktur)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def get_repo_root() -> Path:
    """Normal Python ve PyInstaller paketlenmiş çalışma için kök dizini döndürür."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent


REPO_ROOT = get_repo_root()
WEIGHTS_DIR = REPO_ROOT / "runs" / "detect" / "g_y11s_1024_full" / "weights"
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
