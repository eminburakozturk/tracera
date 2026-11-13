"""Tracera Web Spike — 10 val görüntüsü için Python (PyTorch FP32 & OpenVINO INT8) referans JSON üretici.

Çıktılar:
- web_spike/ref/<stem>__pytorch.json       (CONTRACT §2 DetectResult, backend="pytorch")
- web_spike/ref/<stem>__openvino-int8.json (CONTRACT §2 DetectResult, backend="openvino-int8")
- web_spike/val_images/<stem>.jpg          (10 val görüntüsünün kopyası — spike test/benchmark için)
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import cv2
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
SPIKE_DIR = Path(__file__).resolve().parent
VAL_IMG_DIR = REPO_ROOT / "data" / "yolo" / "grouped" / "val" / "images"
REF_DIR = SPIKE_DIR / "ref"
COPY_IMG_DIR = SPIKE_DIR / "val_images"

PT_PATH = REPO_ROOT / "runs" / "detect" / "g_y11s_1024_full" / "weights" / "last.pt"
OV_PATH = REPO_ROOT / "runs" / "detect" / "g_y11s_1024_full" / "weights" / "last_int8_openvino_model"

CLASSES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]

# Val kartlarından (05 ve 11) tüm 6 sınıfı kapsayan 10 görüntü
VAL_FILES = [
    "05_missing_hole_01.jpg",
    "05_open_circuit_01.jpg",
    "05_short_01.jpg",
    "05_spur_01.jpg",
    "05_spurious_copper_01.jpg",
    "11_missing_hole_01.jpg",
    "11_mouse_bite_01.jpg",
    "11_open_circuit_01.jpg",
    "11_spur_01.jpg",
    "11_spurious_copper_01.jpg",
]

IMGSZ = 1024
CONF = 0.395
IOU_NMS = 0.7


def predict_to_contract(model: YOLO, img_bgr, backend: str) -> dict:
    h, w = img_bgr.shape[:2]
    t0 = time.perf_counter()
    results = model.predict(
        source=img_bgr,
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU_NMS,
        device="cpu",
        verbose=False,
    )
    total_ms = (time.perf_counter() - t0) * 1000.0
    res = results[0]
    speed = getattr(res, "speed", {}) or {}
    pre_ms = float(speed.get("preprocess", 0.0))
    model_ms = float(speed.get("inference", 0.0))
    post_ms = float(speed.get("postprocess", max(0.0, total_ms - pre_ms - model_ms)))

    detections = []
    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)
        # Güvene göre azalan sırala (tutarlı karşılaştırma)
        order = confs.argsort()[::-1]
        for det_id, idx in enumerate(order):
            cid = int(clss[idx])
            detections.append(
                {
                    "id": det_id,
                    "cls": CLASSES[cid],
                    "cls_id": cid,
                    "conf": round(float(confs[idx]), 6),
                    "box": [round(float(v), 3) for v in xyxy[idx].tolist()],
                }
            )

    return {
        "width": int(w),
        "height": int(h),
        "detections": detections,
        "timing": {
            "pre_ms": round(pre_ms, 3),
            "model_ms": round(model_ms, 3),
            "post_ms": round(post_ms, 3),
            "total_ms": round(total_ms, 3),
        },
        "backend": backend,
    }


def main() -> None:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    COPY_IMG_DIR.mkdir(parents=True, exist_ok=True)

    print("🔄 PyTorch FP32 ve OpenVINO INT8 modelleri yükleniyor (CPU)...")
    pt_model = YOLO(str(PT_PATH))
    ov_model = YOLO(str(OV_PATH), task="detect")

    # Warmup
    first_img = cv2.imread(str(VAL_IMG_DIR / VAL_FILES[0]))
    if first_img is not None:
        _ = predict_to_contract(pt_model, first_img, "pytorch")
        _ = predict_to_contract(ov_model, first_img, "openvino-int8")

    manifest = []
    for fname in VAL_FILES:
        src_path = VAL_IMG_DIR / fname
        if not src_path.is_file():
            raise FileNotFoundError(f"Val görüntüsü bulunamadı: {src_path}")

        dst_img = COPY_IMG_DIR / fname
        shutil.copy2(src_path, dst_img)

        img_bgr = cv2.imread(str(src_path))
        if img_bgr is None:
            raise RuntimeError(f"Görüntü okunamadı: {src_path}")

        pt_res = predict_to_contract(pt_model, img_bgr, "pytorch")
        ov_res = predict_to_contract(ov_model, img_bgr, "openvino-int8")

        stem = src_path.stem
        pt_out = REF_DIR / f"{stem}__pytorch.json"
        ov_out = REF_DIR / f"{stem}__openvino-int8.json"
        pt_out.write_text(json.dumps(pt_res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        ov_out.write_text(json.dumps(ov_res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        print(
            f"  {fname:<28} | PT: {len(pt_res['detections']):2d} det ({pt_res['timing']['total_ms']:6.1f} ms) "
            f"| OV-INT8: {len(ov_res['detections']):2d} det ({ov_res['timing']['total_ms']:6.1f} ms)"
        )
        manifest.append(fname)

    (SPIKE_DIR / "val_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n✅ 10 görüntü için 20 referans JSON üretildi -> {REF_DIR}")


if __name__ == "__main__":
    main()
