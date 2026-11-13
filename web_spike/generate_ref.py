"""10 doğrulama görüntüsü için CONTRACT §2 DetectResult referans JSON'larını (PyTorch FP32 & OpenVINO INT8) üretir."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
META_PATH = REPO_ROOT / "web" / "public" / "models" / "meta.json"
VAL_IMAGES_DIR = REPO_ROOT / "data" / "yolo" / "grouped" / "val" / "images"
PT_WEIGHTS = REPO_ROOT / "runs" / "detect" / "g_y11s_1024_full" / "weights" / "last.pt"
OV_WEIGHTS = REPO_ROOT / "runs" / "detect" / "g_y11s_1024_full" / "weights" / "last_int8_openvino_model"
REF_DIR = REPO_ROOT / "web_spike" / "ref"
SAMPLES_JSON = REPO_ROOT / "web_spike" / "samples.json"

VAL_SAMPLE_FILES = [
    "05_missing_hole_01.jpg",
    "05_short_01.jpg",
    "11_mouse_bite_01.jpg",
    "11_spur_01.jpg",
    "05_mouse_bite_01.jpg",
    "05_open_circuit_01.jpg",
    "05_spurious_copper_01.jpg",
    "11_missing_hole_01.jpg",
    "11_open_circuit_01.jpg",
    "11_spurious_copper_01.jpg",
]


def result_to_detect_result(res, width: int, height: int, classes: list[str], backend: str, wall_ms: float) -> dict:
    speed = getattr(res, "speed", {}) or {}
    pre_ms = round(float(speed.get("preprocess", 0.0)), 3)
    model_ms = round(float(speed.get("inference", 0.0)), 3)
    post_ms = round(float(speed.get("postprocess", 0.0)), 3)
    total_ms = round(pre_ms + model_ms + post_ms, 3) if (pre_ms + model_ms + post_ms) > 0 else round(wall_ms, 3)

    detections = []
    boxes = res.boxes
    if boxes is not None and len(boxes) > 0:
        xyxy_arr = boxes.xyxy.cpu().numpy()
        conf_arr = boxes.conf.cpu().numpy()
        cls_arr = boxes.cls.cpu().numpy().astype(int)
        for idx, (xyxy, conf_val, cls_id) in enumerate(zip(xyxy_arr, conf_arr, cls_arr)):
            detections.append(
                {
                    "id": idx,
                    "cls": classes[int(cls_id)],
                    "cls_id": int(cls_id),
                    "conf": round(float(conf_val), 6),
                    "box": [round(float(v), 3) for v in xyxy.tolist()],
                }
            )

    return {
        "width": int(width),
        "height": int(height),
        "detections": detections,
        "timing": {
            "pre_ms": pre_ms,
            "model_ms": model_ms,
            "post_ms": post_ms,
            "total_ms": total_ms,
        },
        "backend": backend,
    }


def main() -> None:
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    imgsz = int(meta["imgsz"])
    conf = float(meta["conf"])
    iou_nms = float(meta["iou_nms"])
    classes = list(meta["classes"])

    REF_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_JSON.write_text(json.dumps(VAL_SAMPLE_FILES, indent=2) + "\n", encoding="utf-8")

    pt_model = YOLO(str(PT_WEIGHTS))
    ov_model = YOLO(str(OV_WEIGHTS), task="detect")

    # Warmup
    first_img = cv2.imread(str(VAL_IMAGES_DIR / VAL_SAMPLE_FILES[0]))
    pt_model.predict(first_img, imgsz=imgsz, conf=conf, iou=iou_nms, rect=False, device="cpu", verbose=False)
    ov_model.predict(first_img, imgsz=imgsz, conf=conf, iou=iou_nms, device="cpu", verbose=False)

    for fname in VAL_SAMPLE_FILES:
        img_path = VAL_IMAGES_DIR / fname
        if not img_path.is_file():
            raise FileNotFoundError(f"Görüntü bulunamadı: {img_path}")
        img_bgr = cv2.imread(str(img_path))
        h, w = img_bgr.shape[:2]
        stem = img_path.stem

        t0 = time.perf_counter()
        res_pt = pt_model.predict(
            img_bgr,
            imgsz=imgsz,
            conf=conf,
            iou=iou_nms,
            rect=False,
            device="cpu",
            verbose=False,
        )[0]
        pt_ms = (time.perf_counter() - t0) * 1000.0
        out_pt = result_to_detect_result(res_pt, w, h, classes, backend="pytorch", wall_ms=pt_ms)
        (REF_DIR / f"{stem}__pytorch.json").write_text(
            json.dumps(out_pt, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        t1 = time.perf_counter()
        res_ov = ov_model.predict(
            img_bgr,
            imgsz=imgsz,
            conf=conf,
            iou=iou_nms,
            device="cpu",
            verbose=False,
        )[0]
        ov_ms = (time.perf_counter() - t1) * 1000.0
        out_ov = result_to_detect_result(res_ov, w, h, classes, backend="openvino-int8", wall_ms=ov_ms)
        (REF_DIR / f"{stem}__openvino-int8.json").write_text(
            json.dumps(out_ov, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            f"{fname:26s} | PT={len(out_pt['detections'])} ({out_pt['timing']['model_ms']:.1f} ms) "
            f"| OV_INT8={len(out_ov['detections'])} ({out_ov['timing']['model_ms']:.1f} ms)"
        )


if __name__ == "__main__":
    main()
