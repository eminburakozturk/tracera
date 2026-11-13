"""Val kartlarından (05 ve 11) 20 saniyelik pan+zoom MP4 test videosu üretir.

Kesin kısıt: Test kartları (06 ve 10) ASLA kullanılmaz.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_EXAMPLES = REPO_ROOT / "web" / "public" / "examples"
OUT_PATH = WEB_EXAMPLES / "val_pan.mp4"

# Yalnız val kartlarından (05 ve 11) seçilen doğrulama görselleri
SOURCE_FILES = [
    "val_05_missing_hole_01.jpg",
    "val_05_short_01.jpg",
    "val_11_mouse_bite_01.jpg",
    "val_11_spur_01.jpg",
]

FPS = 12
SECONDS_PER_CLIP = 5
WIDTH = 960
HEIGHT = 540


def ease_in_out(t: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * max(0.0, min(1.0, t)))


def main() -> None:
    for name in SOURCE_FILES:
        if "_06_" in name or "_10_" in name or name.startswith(("06_", "10_")):
            raise RuntimeError(f"Test kartı yasak: {name}")
        if not ("_05_" in name or "_11_" in name):
            raise RuntimeError(f"Yalnız val kartları (05/11) beklenir: {name}")

    WEB_EXAMPLES.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"H264")
    writer = cv2.VideoWriter(str(OUT_PATH), cv2.CAP_MSMF, fourcc, FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"H264 VideoWriter açılamadı: {OUT_PATH}")

    frames_per_clip = FPS * SECONDS_PER_CLIP
    total_frames = 0

    try:
        for idx, name in enumerate(SOURCE_FILES):
            src_path = WEB_EXAMPLES / name
            img = cv2.imread(str(src_path))
            if img is None or img.size == 0:
                raise FileNotFoundError(f"Görsel okunamadı: {src_path}")
            ih, iw = img.shape[:2]

            # Hedef 16:9 kadrajı görsel içinde yakınlaştırma + yavaş kaydırmayla gezdir
            target_ar = WIDTH / HEIGHT
            base_h = min(ih, iw / target_ar)
            base_w = base_h * target_ar

            # Her klipte farklı köşe ekseninden merkeze doğru yumuşak pan/zoom
            dx_sign = 1.0 if idx % 2 == 0 else -1.0
            dy_sign = 1.0 if idx < 2 else -1.0

            for f in range(frames_per_clip):
                t = ease_in_out(f / max(1, frames_per_clip - 1))
                zoom = 1.02 + 0.36 * t
                crop_w = base_w / zoom
                crop_h = base_h / zoom

                max_shift_x = max(0.0, (iw - crop_w) * 0.42)
                max_shift_y = max(0.0, (ih - crop_h) * 0.42)
                cx = iw * 0.5 + dx_sign * max_shift_x * (t - 0.5)
                cy = ih * 0.5 + dy_sign * max_shift_y * (0.5 - t)

                x1 = int(round(max(0.0, min(iw - crop_w, cx - crop_w * 0.5))))
                y1 = int(round(max(0.0, min(ih - crop_h, cy - crop_h * 0.5))))
                x2 = int(round(min(iw, x1 + crop_w)))
                y2 = int(round(min(ih, y1 + crop_h)))

                patch = img[y1:y2, x1:x2]
                frame = cv2.resize(patch, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                writer.write(frame)
                total_frames += 1
    finally:
        writer.release()

    size_mb = OUT_PATH.stat().st_size / (1024 * 1024)
    dur_s = total_frames / FPS
    print(
        f"[make_val_video] Yazıldı: {OUT_PATH} ({total_frames} kare, {dur_s:.1f} s, {WIDTH}x{HEIGHT}, {size_mb:.2f} MB)"
    )


if __name__ == "__main__":
    main()
