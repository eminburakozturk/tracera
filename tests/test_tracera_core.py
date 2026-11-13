"""tracera_core ortak çekirdek birimleri doğrulama testleri."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracera_core import (  # noqa: E402
    DEFAULT_CONF,
    DEFAULT_IOU_NMS,
    IMGSZ,
    OPENVINO_MODEL_PATH,
    RESULTS_DIR,
    DetectResult,
    build_metrics_payload,
    detect_openvino,
    explain_layercam,
    preprocess_letterbox,
    preprocess_tensor,
)

VAL_IMAGE_PATH = REPO_ROOT / "app" / "examples" / "val_05_missing_hole_01.jpg"
REF_OV_JSON = REPO_ROOT / "web_spike" / "ref" / "05_missing_hole_01__openvino-int8.json"
WEB_METRICS_PATH = REPO_ROOT / "web" / "public" / "metrics.json"


def test_tracera_core_does_not_import_gradio() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, tracera_core; assert 'gradio' not in sys.modules, 'tracera_core gradio import etmemeli!'",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr}"


def test_preprocess_letterbox_and_tensor() -> None:
    img_bgr = cv2.imread(str(VAL_IMAGE_PATH))
    assert img_bgr is not None
    lb = preprocess_letterbox(img_bgr, imgsz=IMGSZ)
    assert lb.img_lb.shape == (IMGSZ, IMGSZ, 3)

    tensor_x, lb2 = preprocess_tensor(img_bgr, imgsz=IMGSZ)
    assert tuple(tensor_x.shape) == (1, 3, IMGSZ, IMGSZ)
    assert float(tensor_x.min()) >= 0.0 and float(tensor_x.max()) <= 1.0
    assert lb2.orig_w == img_bgr.shape[1] and lb2.orig_h == img_bgr.shape[0]


def test_detect_openvino_matches_reference_output() -> None:
    img_bgr = cv2.imread(str(VAL_IMAGE_PATH))
    assert img_bgr is not None
    res = detect_openvino(img_bgr, conf=DEFAULT_CONF, iou_nms=DEFAULT_IOU_NMS, imgsz=IMGSZ)
    assert isinstance(res, DetectResult)
    assert res.backend == "openvino-int8"
    assert res.width == img_bgr.shape[1] and res.height == img_bgr.shape[0]
    assert len(res.detections) > 0

    ov = YOLO(str(OPENVINO_MODEL_PATH), task="detect")
    ov_res = ov.predict(img_bgr, imgsz=IMGSZ, conf=DEFAULT_CONF, iou=DEFAULT_IOU_NMS, device="cpu", verbose=False)[0]
    ov_confs = [float(c) for c in ov_res.boxes.conf.cpu().numpy()] if ov_res.boxes is not None else []
    assert len(res.detections) == len(ov_confs)
    assert abs(max(d.conf for d in res.detections) - max(ov_confs)) < 1e-4

    if REF_OV_JSON.is_file():
        ref = json.loads(REF_OV_JSON.read_text(encoding="utf-8"))
        assert len(res.detections) == len(ref["detections"])
        ref_max = max(d["conf"] for d in ref["detections"])
        assert abs(max(d.conf for d in res.detections) - ref_max) < 1e-3


def test_explain_layercam_returns_rgb_arrays() -> None:
    img_bgr = cv2.imread(str(VAL_IMAGE_PATH))
    assert img_bgr is not None
    res = detect_openvino(img_bgr)
    top = res.detections[0]
    overlay_rgb, zoom_rgb, timing = explain_layercam(
        img_bgr=img_bgr,
        box_orig=tuple(top.box),  # type: ignore[arg-type]
        cls_id=top.cls_id,
    )
    assert isinstance(overlay_rgb, np.ndarray) and overlay_rgb.ndim == 3 and overlay_rgb.shape[2] == 3
    assert isinstance(zoom_rgb, np.ndarray) and zoom_rgb.ndim == 3 and zoom_rgb.shape[2] == 3
    assert timing.total_ms > 0


def test_metrics_loader_matches_web_metrics() -> None:
    payload = build_metrics_payload(RESULTS_DIR)
    for k in ("operating_point", "edge_benchmark", "cv", "xai", "aoi"):
        assert k in payload
    if WEB_METRICS_PATH.is_file():
        assert payload == json.loads(WEB_METRICS_PATH.read_text(encoding="utf-8"))
