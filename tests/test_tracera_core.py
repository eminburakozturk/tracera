"""tracera_core ortak çekirdek birimleri doğrulama testleri."""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pytest
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
SAMPLES_JSON = REPO_ROOT / "web_spike" / "samples.json"
VAL_IMAGES_DIR = REPO_ROOT / "data" / "yolo" / "grouped" / "val" / "images"
REF_DIR = REPO_ROOT / "web_spike" / "ref"


def _box_iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / max(1e-9, area_a + area_b - inter)


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


@pytest.mark.needs_weights
def test_detect_openvino_matches_reference_output(record_property) -> None:
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
        cur_max = max(d.conf for d in res.detections)
        ref_max = max(d["conf"] for d in ref["detections"])
        abs_diff = abs(cur_max - ref_max)
        count_diff = len(res.detections) - len(ref["detections"])
        if sys.platform == "win32":
            assert len(res.detections) == len(ref["detections"])
            assert abs_diff < 1e-3
        else:
            # OpenVINO/OpenCV platform kayması, KARARLAR 2026-11-14
            msg = (
                f"[platform-drift] {sys.platform} OpenVINO vs Windows ref ({REF_OV_JSON.name}): "
                f"max_conf={cur_max:.6f} vs ref={ref_max:.6f}, |dconf|={abs_diff:.6f}, "
                f"count={len(res.detections)} vs ref={len(ref['detections'])} ({count_diff:+d})"
            )
            record_property("platform_drift", msg)
            warnings.warn(UserWarning(msg), stacklevel=1)


@pytest.mark.needs_weights
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


@pytest.mark.needs_weights
def test_detect_openvino_10_val_ultralytics_parity() -> None:
    if not VAL_IMAGES_DIR.is_dir():
        pytest.skip(f"Val goruntu dizini yok (CI): {VAL_IMAGES_DIR}")
    assert SAMPLES_JSON.is_file()
    samples = json.loads(SAMPLES_JSON.read_text(encoding="utf-8"))
    assert len(samples) == 10

    ov = YOLO(str(OPENVINO_MODEL_PATH), task="detect")
    for fname in samples:
        img_path = VAL_IMAGES_DIR / fname
        assert img_path.is_file(), f"Val görüntüsü eksik: {img_path}"
        img_bgr = cv2.imread(str(img_path))
        assert img_bgr is not None

        ref_res = ov.predict(img_bgr, imgsz=IMGSZ, conf=DEFAULT_CONF, iou=DEFAULT_IOU_NMS, device="cpu", verbose=False)[0]
        ref_dets = sorted(
            [
                (int(c), float(s), [float(v) for v in b])
                for c, s, b in zip(
                    ref_res.boxes.cls.cpu().numpy(),
                    ref_res.boxes.conf.cpu().numpy(),
                    ref_res.boxes.xyxy.cpu().numpy(),
                )
            ],
            key=lambda x: -x[1],
        )

        tc_res = detect_openvino(img_bgr, conf=DEFAULT_CONF, iou_nms=DEFAULT_IOU_NMS, imgsz=IMGSZ)
        tc_dets = sorted([(d.cls_id, d.conf, d.box) for d in tc_res.detections], key=lambda x: -x[1])

        assert len(tc_dets) == len(ref_dets), f"{fname}: adet uyuşmazlığı {len(tc_dets)} vs {len(ref_dets)}"
        for (rc, rs, rb), (tc, ts, tb) in zip(ref_dets, tc_dets):
            assert tc == rc
            assert abs(ts - rs) < 0.002, f"{fname}: |Δconf|={abs(ts - rs)}"
            assert _box_iou(rb, tb) >= 0.999, f"{fname}: IoU={_box_iou(rb, tb)}"

        ref_json_path = REF_DIR / f"{img_path.stem}__openvino-int8.json"
        if ref_json_path.is_file():
            ref_json = json.loads(ref_json_path.read_text(encoding="utf-8"))
            assert len(tc_dets) == len(ref_json["detections"]), f"{fname}: JSON adet uyuşmazlığı"
