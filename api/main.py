"""Tracera yerel FastAPI servisi (CONTRACT §3).

Çalıştırma:
    uvicorn api.main:app --port 8765
"""

from __future__ import annotations

import argparse
import base64
import math
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

_REPO_ROOT = (
    Path(sys._MEIPASS)  # type: ignore[attr-defined]
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")
    else Path(__file__).resolve().parent.parent
)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from api.schemas import (  # noqa: E402
    DetectResult,
    ErrorResponse,
    ExplainResult,
    HealthResponse,
    MetricsResponse,
)
from tracera_core import (  # noqa: E402
    CLASS_NAMES,
    DEFAULT_CONF,
    DEFAULT_IOU_NMS,
    IMGSZ,
    MODELS,
    RESULTS_DIR,
    build_metrics_payload,
    detect_openvino,
    explain_layercam_with_metrics,
    load_meta,
)

MAX_IMAGE_BYTES = 25 * 1024 * 1024  # CONTRACT §3: 25 MB

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "tauri://localhost",
    "http://tauri.localhost",
    "https://tauri.localhost",
]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """OpenVINO INT8 dedektörü ve PyTorch LayerCAM motorunu başlangıçta bir kez yükler."""
    MODELS.get_detector()
    MODELS.get_cam_engine()
    yield


app = FastAPI(
    title="Tracera Local API",
    version=str(load_meta().get("version", "1")),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def _http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"error": detail})


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    msgs = [f"{'.'.join(str(x) for x in err.get('loc', []))}: {err.get('msg', '')}" for err in exc.errors()]
    return JSONResponse(status_code=422, content={"error": "; ".join(msgs) or "Geçersiz istek parametreleri."})


async def _read_and_decode_image(image: UploadFile) -> np.ndarray:
    chunks: list[bytes] = []
    total = 0
    chunk_size = 1024 * 1024
    while True:
        chunk = await image.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Görüntü boyutu 25 MB sınırını aşıyor (>{MAX_IMAGE_BYTES} bayt).",
            )
        chunks.append(chunk)

    raw_bytes = b"".join(chunks)
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Boş görüntü dosyası gönderildi.")

    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None or img_bgr.size == 0:
        raise HTTPException(status_code=400, detail="Görüntü çözümlenemedi; bozuk veya desteklenmeyen dosya.")
    return img_bgr


def _parse_box(box_str: str, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    parts = [p.strip() for p in box_str.split(",")]
    if len(parts) != 4:
        raise HTTPException(status_code=422, detail="box alanı 'x1,y1,x2,y2' biçiminde 4 sayı olmalıdır.")
    try:
        vals = [float(p) for p in parts]
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="box koordinatları geçerli sayı olmalıdır.") from exc

    if any(math.isnan(v) or math.isinf(v) for v in vals):
        raise HTTPException(status_code=422, detail="box koordinatları sonlu olmalıdır.")

    x1, y1, x2, y2 = vals
    if x2 <= x1 or y2 <= y1:
        raise HTTPException(status_code=422, detail="box koordinatları x2 > x1 ve y2 > y1 olmalıdır.")

    x1 = max(0.0, min(x1, float(img_w)))
    y1 = max(0.0, min(y1, float(img_h)))
    x2 = max(0.0, min(x2, float(img_w)))
    y2 = max(0.0, min(y2, float(img_h)))
    if x2 <= x1 or y2 <= y1:
        raise HTTPException(status_code=422, detail="box koordinatları görüntü sınırları dışında veya sıfır alanlı.")
    return (x1, y1, x2, y2)


def _encode_rgb_png_base64(rgb_img: np.ndarray) -> str:
    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr_img)
    if not ok:
        raise HTTPException(status_code=500, detail="PNG kodlaması başarısız oldu.")
    return base64.b64encode(buf.tobytes()).decode("ascii")


@app.get("/health", response_model=HealthResponse, responses={500: {"model": ErrorResponse}})
def get_health() -> HealthResponse:
    meta = load_meta()
    MODELS.get_detector()
    MODELS.get_cam_engine()
    return HealthResponse(
        status="ok",
        model=str(meta.get("model", "yolo11s")),
        version=str(meta.get("version", "1")),
        device="cpu",
        backends=MODELS.available_backends(),
    )


@app.post("/detect", response_model=DetectResult, responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 422: {"model": ErrorResponse}})
async def post_detect(
    image: UploadFile = File(...),
    conf: float | None = Form(None),
) -> DetectResult:
    meta = load_meta()
    imgsz = int(meta.get("imgsz", IMGSZ))
    iou_nms = float(meta.get("iou_nms", DEFAULT_IOU_NMS))
    conf_val = float(meta.get("conf", DEFAULT_CONF)) if conf is None else float(conf)
    if math.isnan(conf_val) or math.isinf(conf_val) or not (0.0 < conf_val <= 1.0):
        raise HTTPException(status_code=422, detail="conf parametresi (0, 1] aralığında olmalıdır.")

    img_bgr = await _read_and_decode_image(image)
    return detect_openvino(img_bgr, conf=conf_val, iou_nms=iou_nms, imgsz=imgsz, manager=MODELS)


@app.post("/explain", response_model=ExplainResult, responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 422: {"model": ErrorResponse}})
async def post_explain(
    image: UploadFile = File(...),
    box: str = Form(...),
    cls_id: int = Form(...),
) -> ExplainResult:
    if cls_id not in CLASS_NAMES:
        raise HTTPException(
            status_code=422,
            detail=f"Geçersiz cls_id={cls_id}; beklenen aralık: 0..{len(CLASS_NAMES) - 1}.",
        )

    img_bgr = await _read_and_decode_image(image)
    h, w = img_bgr.shape[:2]
    box_orig = _parse_box(box, img_w=w, img_h=h)

    meta = load_meta()
    imgsz = int(meta.get("imgsz", IMGSZ))

    overlay_rgb, zoom_crop_rgb, timing, cam_metrics = explain_layercam_with_metrics(
        img_bgr=img_bgr,
        box_orig=box_orig,
        cls_id=cls_id,
        imgsz=imgsz,
        manager=MODELS,
    )
    return ExplainResult(
        overlay_png=_encode_rgb_png_base64(overlay_rgb),
        zoom_png=_encode_rgb_png_base64(zoom_crop_rgb),
        method="layercam",
        layer=4,
        timing=timing,
        energy_in_box=float(cam_metrics["energy_in_box"]),
        peak_in_box=bool(cam_metrics["peak_in_box"]),
        peak_xy=list(cam_metrics["peak_xy"]),
    )


@app.get("/metrics", response_model=MetricsResponse, responses={500: {"model": ErrorResponse}})
def get_metrics() -> dict[str, Any]:
    return build_metrics_payload(RESULTS_DIR)


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Tracera yerel FastAPI sunucusu")
    parser.add_argument("--host", default=os.environ.get("TRACERA_API_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("TRACERA_API_PORT", "8765")))
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
