"""Tracera FastAPI servisi (api/main.py) doğrulama testleri."""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.needs_weights
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.main import MAX_IMAGE_BYTES, app  # noqa: E402

VAL_IMAGE_PATH = REPO_ROOT / "app" / "examples" / "val_05_missing_hole_01.jpg"
OV_MODEL_PATH = REPO_ROOT / "runs" / "detect" / "g_y11s_1024_full" / "weights" / "last_int8_openvino_model"
WEB_METRICS_PATH = REPO_ROOT / "web" / "public" / "metrics.json"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as tc:
        yield tc


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model"] == "yolo11s"
    assert data["version"] == "1.1.0"
    assert data["device"] == "cpu"
    assert "openvino-int8" in data["backends"]


def test_metrics_matches_static_json(client: TestClient) -> None:
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    for key in ("operating_point", "edge_benchmark", "cv", "xai", "aoi"):
        assert key in data
    assert len(data["operating_point"]) >= 2
    assert len(data["edge_benchmark"]) >= 4

    assert WEB_METRICS_PATH.is_file()
    static_data = json.loads(WEB_METRICS_PATH.read_text(encoding="utf-8"))
    assert data == static_data


def test_detect_matches_direct_openvino(client: TestClient) -> None:
    assert VAL_IMAGE_PATH.is_file()
    img_bytes = VAL_IMAGE_PATH.read_bytes()

    resp = client.post("/detect", files={"image": ("val_05.jpg", img_bytes, "image/jpeg")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["backend"] == "openvino-int8"
    assert body["width"] > 0 and body["height"] > 0
    assert body["timing"]["total_ms"] > 0

    ov = YOLO(str(OV_MODEL_PATH), task="detect")
    img_bgr = cv2.imread(str(VAL_IMAGE_PATH))
    ov_res = ov.predict(img_bgr, imgsz=1024, conf=0.395, iou=0.7, device="cpu", verbose=False)[0]
    ov_confs = [float(c) for c in ov_res.boxes.conf.cpu().numpy()] if ov_res.boxes is not None else []

    assert len(body["detections"]) == len(ov_confs)
    if ov_confs:
        api_max_conf = max(d["conf"] for d in body["detections"])
        assert abs(api_max_conf - max(ov_confs)) < 1e-4


def test_explain_returns_two_valid_pngs(client: TestClient) -> None:
    img_bytes = VAL_IMAGE_PATH.read_bytes()
    det_resp = client.post("/detect", files={"image": ("val_05.jpg", img_bytes, "image/jpeg")})
    assert det_resp.status_code == 200
    detections = det_resp.json()["detections"]
    assert len(detections) > 0

    top_det = detections[0]
    box_str = ",".join(str(v) for v in top_det["box"])
    exp_resp = client.post(
        "/explain",
        files={"image": ("val_05.jpg", img_bytes, "image/jpeg")},
        data={"box": box_str, "cls_id": str(top_det["cls_id"])},
    )
    assert exp_resp.status_code == 200
    exp_data = exp_resp.json()
    assert exp_data["method"] == "layercam"
    assert exp_data["layer"] == 4
    assert exp_data["timing"]["total_ms"] > 0
    assert isinstance(exp_data["energy_in_box"], float)
    assert 0.0 <= exp_data["energy_in_box"] <= 1.0
    assert isinstance(exp_data["peak_in_box"], bool)
    assert isinstance(exp_data["peak_xy"], list) and len(exp_data["peak_xy"]) == 2
    img_w = det_resp.json()["width"]
    img_h = det_resp.json()["height"]
    assert 0 <= exp_data["peak_xy"][0] < img_w and 0 <= exp_data["peak_xy"][1] < img_h

    for field in ("overlay_png", "zoom_png"):
        b64_str = exp_data[field]
        assert not b64_str.startswith("data:")
        raw = base64.b64decode(b64_str, validate=True)
        assert raw.startswith(b"\x89PNG\r\n\x1a\n")
        decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None
        assert decoded.shape[0] > 0 and decoded.shape[1] > 0


def test_corrupt_and_missing_fields_return_4xx(client: TestClient) -> None:
    # Bozuk görüntü
    r_bad_img = client.post(
        "/detect",
        files={"image": ("corrupt.jpg", b"not-a-valid-image-bytes", "image/jpeg")},
    )
    assert 400 <= r_bad_img.status_code < 500
    assert "error" in r_bad_img.json()

    # Eksik image alanı
    r_no_img = client.post("/detect")
    assert 400 <= r_no_img.status_code < 500
    assert "error" in r_no_img.json()

    # /explain eksik box/cls_id
    img_bytes = VAL_IMAGE_PATH.read_bytes()
    r_missing_box = client.post(
        "/explain",
        files={"image": ("val_05.jpg", img_bytes, "image/jpeg")},
        data={"cls_id": "0"},
    )
    assert 400 <= r_missing_box.status_code < 500
    assert "error" in r_missing_box.json()

    # /explain geçersiz cls_id
    r_bad_cls = client.post(
        "/explain",
        files={"image": ("val_05.jpg", img_bytes, "image/jpeg")},
        data={"box": "10,10,50,50", "cls_id": "99"},
    )
    assert 400 <= r_bad_cls.status_code < 500
    assert "error" in r_bad_cls.json()


def test_oversize_file_rejected(client: TestClient) -> None:
    oversized = b"\x00" * (MAX_IMAGE_BYTES + 1024)
    resp = client.post(
        "/detect",
        files={"image": ("huge.jpg", oversized, "image/jpeg")},
    )
    assert resp.status_code == 413
    assert "error" in resp.json()
