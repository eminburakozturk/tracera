"""Gradio Space (/detect_api ve /explain_api) uçları için canlı gradio_client testi."""

from __future__ import annotations

import base64
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest
from gradio_client import Client, handle_file

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracera_core import DEFAULT_CONF, detect_openvino  # noqa: E402

VAL_IMAGE_PATH = REPO_ROOT / "app" / "examples" / "val_05_missing_hole_01.jpg"


@pytest.mark.needs_weights
def test_gradio_api_detect_and_explain() -> None:
    assert VAL_IMAGE_PATH.is_file()
    port = 7863
    base_url = f"http://127.0.0.1:{port}"

    env = os.environ.copy()
    env["GRADIO_SERVER_NAME"] = "127.0.0.1"
    env["GRADIO_SERVER_PORT"] = str(port)
    env["CUDA_VISIBLE_DEVICES"] = ""

    proc = subprocess.Popen(
        [sys.executable, str(REPO_ROOT / "app" / "app.py")],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        for _ in range(90):
            if proc.poll() is not None:
                out = proc.stdout.read() if proc.stdout else ""
                raise RuntimeError(f"app/app.py erken kapandı (code={proc.returncode}):\n{out}")
            try:
                r = httpx.get(f"{base_url}/", timeout=1.5)
                if r.status_code == 200:
                    break
            except Exception:
                time.sleep(0.5)
        else:
            raise TimeoutError("app/app.py 45 sn içinde açılmadı.")

        client = Client(base_url, verbose=False)

        # 1) /detect_api
        det_out = client.predict(
            image=handle_file(str(VAL_IMAGE_PATH)),
            conf=DEFAULT_CONF,
            api_name="/detect_api",
        )
        assert isinstance(det_out, dict)
        assert det_out["backend"] in ("openvino-int8", "pytorch")
        assert det_out["width"] > 0 and det_out["height"] > 0
        assert det_out["timing"]["total_ms"] > 0

        img_bgr = cv2.imread(str(VAL_IMAGE_PATH))
        assert img_bgr is not None
        core_ref = detect_openvino(img_bgr, conf=DEFAULT_CONF)
        assert len(det_out["detections"]) == len(core_ref.detections)
        assert abs(max(d["conf"] for d in det_out["detections"]) - max(d.conf for d in core_ref.detections)) < 1e-4

        # 2) /explain_api
        top_det = det_out["detections"][0]
        box_str = ",".join(str(v) for v in top_det["box"])
        exp_out = client.predict(
            image=handle_file(str(VAL_IMAGE_PATH)),
            box=box_str,
            cls_id=int(top_det["cls_id"]),
            api_name="/explain_api",
        )
        assert isinstance(exp_out, dict)
        assert exp_out["method"] == "layercam"
        assert exp_out["layer"] == 4
        assert exp_out["timing"]["total_ms"] > 0
        assert isinstance(exp_out["energy_in_box"], float)
        assert 0.0 <= exp_out["energy_in_box"] <= 1.0
        assert isinstance(exp_out["peak_in_box"], bool)
        assert isinstance(exp_out["peak_xy"], list) and len(exp_out["peak_xy"]) == 2
        assert 0 <= exp_out["peak_xy"][0] < det_out["width"]
        assert 0 <= exp_out["peak_xy"][1] < det_out["height"]

        for field in ("overlay_png", "zoom_png"):
            b64_str = exp_out[field]
            assert isinstance(b64_str, str) and not b64_str.startswith("data:")
            raw = base64.b64decode(b64_str, validate=True)
            assert raw.startswith(b"\x89PNG\r\n\x1a\n")
            decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            assert decoded is not None and decoded.shape[0] > 0 and decoded.shape[1] > 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
