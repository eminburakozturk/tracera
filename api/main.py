"""Tracera yerel FastAPI servisi (CONTRACT §3).

Çalıştırma:
    uvicorn api.main:app --port 8765
"""

from __future__ import annotations

import base64
import json
import math
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    REPO_ROOT = Path(sys._MEIPASS)  # type: ignore[attr-defined]
else:
    REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.app import (  # noqa: E402
    CLASS_COLORS,
    CLASS_NAMES,
    DEFAULT_CONF,
    IMGSZ,
    MODELS,
    OPENVINO_MODEL_PATH,
)
from scripts.gradcam import (  # noqa: E402
    box_orig_to_lb,
    letterbox_image,
    make_overlay_and_zoom,
)
from api.metrics_loader import build_metrics_payload  # noqa: E402
from api.schemas import (  # noqa: E402
    Detection,
    DetectResult,
    ErrorResponse,
    ExplainResult,
    HealthResponse,
    MetricsResponse,
    Timing,
)

MAX_IMAGE_BYTES = 25 * 1024 * 1024  # CONTRACT §3: 25 MB
META_PATH = REPO_ROOT / "web" / "public" / "models" / "meta.json"
RESULTS_DIR = REPO_ROOT / "results"

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "tauri://localhost",
    "http://tauri.localhost",
    "https://tauri.localhost",
]

_CAM_LOCK = threading.Lock()


def _load_meta() -> dict[str, Any]:
    if META_PATH.is_file():
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    return {
        "version": "1",
        "model": "yolo11s",
        "imgsz": IMGSZ,
        "conf": DEFAULT_CONF,
        "iou_nms": 0.7,
        "classes": [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)],
    }


def _active_detect_backend() -> str:
    return "openvino-int8" if OPENVINO_MODEL_PATH.exists() else "pytorch"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """OpenVINO INT8 dedektörü ve PyTorch LayerCAM motorunu başlangıçta bir kez yükler."""
    MODELS.get_detector()
    MODELS.get_cam_engine()
    yield


app = FastAPI(
    title="Tracera Local API",
    version=_load_meta().get("version", "1"),
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
    meta = _load_meta()
    MODELS.get_detector()
    MODELS.get_cam_engine()
    backends = ["openvino-int8", "pytorch"] if OPENVINO_MODEL_PATH.exists() else ["pytorch"]
    return HealthResponse(
        status="ok",
        model=str(meta.get("model", "yolo11s")),
        version=str(meta.get("version", "1")),
        device="cpu",
        backends=backends,
    )


@app.post("/detect", response_model=DetectResult, responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 422: {"model": ErrorResponse}})
async def post_detect(
    image: UploadFile = File(...),
    conf: float | None = Form(None),
) -> DetectResult:
    meta = _load_meta()
    imgsz = int(meta.get("imgsz", IMGSZ))
    iou_nms = float(meta.get("iou_nms", 0.7))
    conf_val = float(meta.get("conf", DEFAULT_CONF)) if conf is None else float(conf)
    if math.isnan(conf_val) or math.isinf(conf_val) or not (0.0 < conf_val <= 1.0):
        raise HTTPException(status_code=422, detail="conf parametresi (0, 1] aralığında olmalıdır.")

    img_bgr = await _read_and_decode_image(image)
    h, w = img_bgr.shape[:2]

    detector, _ = MODELS.get_detector()
    t0 = time.perf_counter()
    results = detector.predict(
        source=img_bgr,
        imgsz=imgsz,
        conf=conf_val,
        iou=iou_nms,
        device="cpu",
        verbose=False,
    )
    wall_ms = (time.perf_counter() - t0) * 1000.0
    res0 = results[0]
    speed = getattr(res0, "speed", {}) or {}
    pre_ms = float(speed.get("preprocess", 0.0))
    model_ms = float(speed.get("inference", 0.0))
    post_ms = float(speed.get("postprocess", 0.0))
    total_ms = pre_ms + model_ms + post_ms if (pre_ms + model_ms + post_ms) > 0.0 else wall_ms

    detections: list[Detection] = []
    boxes_obj = res0.boxes
    if boxes_obj is not None and len(boxes_obj) > 0:
        xyxy_arr = boxes_obj.xyxy.cpu().numpy()
        conf_arr = boxes_obj.conf.cpu().numpy()
        cls_arr = boxes_obj.cls.cpu().numpy().astype(int)
        for idx, (xyxy, c_val, cls_id) in enumerate(zip(xyxy_arr, conf_arr, cls_arr)):
            cls_int = int(cls_id)
            detections.append(
                Detection(
                    id=idx,
                    cls=CLASS_NAMES.get(cls_int, f"class_{cls_int}"),
                    cls_id=cls_int,
                    conf=round(float(c_val), 6),
                    box=[round(float(v), 3) for v in xyxy.tolist()],
                )
            )

    return DetectResult(
        width=int(w),
        height=int(h),
        detections=detections,
        timing=Timing(
            pre_ms=round(pre_ms, 3),
            model_ms=round(model_ms, 3),
            post_ms=round(post_ms, 3),
            total_ms=round(total_ms, 3),
        ),
        backend=_active_detect_backend(),
    )


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

    meta = _load_meta()
    imgsz = int(meta.get("imgsz", IMGSZ))

    t0 = time.perf_counter()
    lb = letterbox_image(img_bgr, imgsz=imgsz)
    tensor_x = torch.from_numpy(lb.img_lb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    box_lb = box_orig_to_lb(box_orig, lb)
    t_pre = time.perf_counter()

    cam_engine = MODELS.get_cam_engine()
    with _CAM_LOCK:
        raw_out, anchor_xyxy = cam_engine.forward_image(tensor_x)
        best_anchor = cam_engine.find_best_anchor(box_lb, anchor_xyxy)
        extracted = cam_engine.backward_and_extract(
            raw_out=raw_out,
            anchor_idx=best_anchor,
            class_id=cls_id,
            lb=lb,
            methods=("layercam",),
        )
    _win, cam_dict = extracted[4]
    _cam_lb, cam_orig = cam_dict["layercam"]
    t_model = time.perf_counter()

    cls_name = CLASS_NAMES[cls_id]
    box_color_rgb = CLASS_COLORS.get(cls_id, (0, 255, 100))
    overlay_rgb, zoom_crop_rgb = make_overlay_and_zoom(
        img_bgr=img_bgr,
        cam_orig=cam_orig,
        box_orig=box_orig,
        label_str=cls_name,
        box_color_rgb=box_color_rgb,
    )
    overlay_b64 = _encode_rgb_png_base64(overlay_rgb)
    zoom_b64 = _encode_rgb_png_base64(zoom_crop_rgb)
    t_end = time.perf_counter()

    return ExplainResult(
        overlay_png=overlay_b64,
        zoom_png=zoom_b64,
        method="layercam",
        layer=4,
        timing=Timing(
            pre_ms=round((t_pre - t0) * 1000.0, 3),
            model_ms=round((t_model - t_pre) * 1000.0, 3),
            post_ms=round((t_end - t_model) * 1000.0, 3),
            total_ms=round((t_end - t0) * 1000.0, 3),
        ),
    )


@app.get("/metrics", response_model=MetricsResponse, responses={500: {"model": ErrorResponse}})
def get_metrics() -> dict[str, Any]:
    return build_metrics_payload(RESULTS_DIR)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765)
