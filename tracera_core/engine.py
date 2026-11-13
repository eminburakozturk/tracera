"""OpenVINO tespit ve LayerCAM (scripts/gradcam.py) ortak çıkarım motoru."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from tracera_core.config import (
    CLASS_COLORS,
    CLASS_NAMES,
    DEFAULT_CONF,
    DEFAULT_IOU_NMS,
    IMGSZ,
    OPENVINO_MODEL_PATH,
    PYTORCH_MODEL_PATH,
    REPO_ROOT,
)
from tracera_core.preprocess import box_orig_to_lb, preprocess_tensor
from tracera_core.schemas import Detection, DetectResult, Timing

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gradcam import MultiLayerYoloCAM, make_overlay_and_zoom  # noqa: E402


class ModelManager:
    """OpenVINO INT8 dedektörünü ve PyTorch LayerCAM motorunu önbellekte tutar."""

    def __init__(
        self,
        openvino_path: Path | None = None,
        pytorch_path: Path | None = None,
        imgsz: int = IMGSZ,
    ) -> None:
        self.openvino_path = openvino_path or OPENVINO_MODEL_PATH
        self.pytorch_path = pytorch_path or PYTORCH_MODEL_PATH
        self.imgsz = imgsz
        self.detect_model: YOLO | None = None
        self.detect_engine_name: str = ""
        self.detect_backend_id: str = "openvino-int8"
        self.pt_model: YOLO | None = None
        self.cam_engine: MultiLayerYoloCAM | None = None
        self._load_lock = threading.Lock()
        self._cam_lock = threading.Lock()

    def get_detector(self) -> tuple[YOLO, str]:
        with self._load_lock:
            if self.detect_model is None:
                if self.openvino_path.exists():
                    self.detect_model = YOLO(str(self.openvino_path), task="detect")
                    self.detect_engine_name = "OpenVINO INT8 (CPU Optimized)"
                    self.detect_backend_id = "openvino-int8"
                elif self.pytorch_path.exists():
                    self.detect_model = YOLO(str(self.pytorch_path))
                    self.detect_engine_name = "PyTorch FP32 (last.pt)"
                    self.detect_backend_id = "pytorch"
                else:
                    raise FileNotFoundError(
                        f"Ne OpenVINO ({self.openvino_path}) ne de PyTorch ({self.pytorch_path}) bulundu!"
                    )
        return self.detect_model, self.detect_engine_name

    def get_cam_engine(self) -> MultiLayerYoloCAM:
        with self._load_lock:
            if self.cam_engine is None:
                if not self.pytorch_path.exists():
                    raise FileNotFoundError(
                        f"LayerCAM için PyTorch modeli bulunamadı: {self.pytorch_path}"
                    )
                self.pt_model = YOLO(str(self.pytorch_path))
                self.cam_engine = MultiLayerYoloCAM(
                    nn_model=self.pt_model.model,
                    layer_indices=[4],
                    imgsz=self.imgsz,
                    device="cpu",
                )
        return self.cam_engine

    def available_backends(self) -> list[str]:
        return ["openvino-int8", "pytorch"] if self.openvino_path.exists() else ["pytorch"]


MODELS = ModelManager()


def active_detect_backend(manager: ModelManager | None = None) -> str:
    mgr = manager or MODELS
    return "openvino-int8" if mgr.openvino_path.exists() else "pytorch"


def detect_openvino(
    img_bgr: np.ndarray,
    *,
    conf: float = DEFAULT_CONF,
    iou_nms: float = DEFAULT_IOU_NMS,
    imgsz: int = IMGSZ,
    manager: ModelManager | None = None,
) -> DetectResult:
    """BGR görüntü üzerinde OpenVINO INT8 tespiti çalıştırıp CONTRACT §2 DetectResult döndürür."""
    mgr = manager or MODELS
    detector, _ = mgr.get_detector()
    h, w = img_bgr.shape[:2]

    t0 = time.perf_counter()
    results = detector.predict(
        source=img_bgr,
        imgsz=imgsz,
        conf=float(conf),
        iou=float(iou_nms),
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
        backend=active_detect_backend(mgr),
    )


def explain_layercam(
    img_bgr: np.ndarray,
    box_orig: tuple[float, float, float, float],
    cls_id: int,
    *,
    imgsz: int = IMGSZ,
    label_str: str | None = None,
    box_color_rgb: tuple[int, int, int] | None = None,
    manager: ModelManager | None = None,
) -> tuple[np.ndarray, np.ndarray, Timing]:
    """scripts/gradcam.py üzerinden LayerCAM @ katman 4 overlay/zoom RGB görüntüleri ve süreleri üretir."""
    mgr = manager or MODELS
    t0 = time.perf_counter()
    tensor_x, lb = preprocess_tensor(img_bgr, imgsz=imgsz)
    box_lb = box_orig_to_lb(box_orig, lb)
    t_pre = time.perf_counter()

    cam_engine = mgr.get_cam_engine()
    with mgr._cam_lock:
        raw_out, anchor_xyxy = cam_engine.forward_image(tensor_x)
        best_anchor = cam_engine.find_best_anchor(box_lb, anchor_xyxy)
        extracted = cam_engine.backward_and_extract(
            raw_out=raw_out,
            anchor_idx=best_anchor,
            class_id=int(cls_id),
            lb=lb,
            methods=("layercam",),
        )
    _win, cam_dict = extracted[4]
    _cam_lb, cam_orig = cam_dict["layercam"]
    t_model = time.perf_counter()

    lbl = label_str if label_str is not None else CLASS_NAMES.get(int(cls_id), f"class_{cls_id}")
    color_rgb = box_color_rgb if box_color_rgb is not None else CLASS_COLORS.get(int(cls_id), (0, 255, 100))
    overlay_rgb, zoom_crop_rgb = make_overlay_and_zoom(
        img_bgr=img_bgr,
        cam_orig=cam_orig,
        box_orig=box_orig,
        label_str=lbl,
        box_color_rgb=color_rgb,
    )
    t_end = time.perf_counter()

    timing = Timing(
        pre_ms=round((t_pre - t0) * 1000.0, 3),
        model_ms=round((t_model - t_pre) * 1000.0, 3),
        post_ms=round((t_end - t_model) * 1000.0, 3),
        total_ms=round((t_end - t0) * 1000.0, 3),
    )
    return overlay_rgb, zoom_crop_rgb, timing
