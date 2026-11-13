"""ADIM 3: PyInstaller onedir ile dist/tracera-api/ derleme ve doğrulama scripti."""

from __future__ import annotations

import base64
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = REPO_ROOT / "dist" / "tracera-api"
BUILD_DIR = REPO_ROOT / "build"
EXE_PATH = DIST_DIR / "tracera-api.exe"
SAMPLE_IMG = REPO_ROOT / "app" / "examples" / "val_05_missing_hole_01.jpg"


def dir_size_mb(path: Path) -> tuple[float, int]:
    files = [f for f in path.rglob("*") if f.is_file()]
    total_bytes = sum(f.stat().st_size for f in files)
    return total_bytes / (1024 * 1024), len(files)


def build_exe() -> float:
    sep = ";" if os.name == "nt" else ":"
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--onedir",
        "--name",
        "tracera-api",
        "--noconfirm",
        "--clean",
        "--distpath",
        str(REPO_ROOT / "dist"),
        "--workpath",
        str(BUILD_DIR),
        "--specpath",
        str(BUILD_DIR),
        "--paths",
        str(REPO_ROOT),
        "--add-data",
        f"{REPO_ROOT / 'runs' / 'detect' / 'g_y11s_1024_full' / 'weights' / 'last.pt'}{sep}runs/detect/g_y11s_1024_full/weights",
        "--add-data",
        f"{REPO_ROOT / 'runs' / 'detect' / 'g_y11s_1024_full' / 'weights' / 'last_int8_openvino_model'}{sep}runs/detect/g_y11s_1024_full/weights/last_int8_openvino_model",
        "--add-data",
        f"{REPO_ROOT / 'web' / 'public' / 'models' / 'meta.json'}{sep}web/public/models",
        "--add-data",
        f"{REPO_ROOT / 'results' / '*.csv'}{sep}results",
        "--add-data",
        f"{REPO_ROOT / 'app' / 'app.py'}{sep}app",
        "--add-data",
        f"{REPO_ROOT / 'scripts' / 'gradcam.py'}{sep}scripts",
        "--collect-all",
        "openvino",
        "--collect-all",
        "torchvision",
        "--collect-data",
        "ultralytics",
        "--exclude-module",
        "gradio",
        "--exclude-module",
        "gradio_client",
        "--exclude-module",
        "tensorrt",
        "--exclude-module",
        "tensorrt_libs",
        "--exclude-module",
        "mlflow",
        "--hidden-import",
        "app.app",
        "--hidden-import",
        "scripts.gradcam",
        "--hidden-import",
        "uvicorn.logging",
        "--hidden-import",
        "uvicorn.loops.auto",
        "--hidden-import",
        "uvicorn.protocols.http.auto",
        "--hidden-import",
        "uvicorn.lifespan.on",
        str(REPO_ROOT / "api" / "main.py"),
    ]
    print("PyInstaller başlatılıyor...")
    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    return time.perf_counter() - t0


def test_compiled_exe() -> None:
    if not EXE_PATH.is_file():
        raise FileNotFoundError(f"Derlenmiş EXE bulunamadı: {EXE_PATH}")

    print(f"\nDerlenmiş EXE başlatılıyor: {EXE_PATH}")
    t_start = time.perf_counter()
    proc = subprocess.Popen(
        [str(EXE_PATH)],
        cwd=str(DIST_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        health_data = None
        cold_start_s = 0.0
        for _ in range(240):
            if proc.poll() is not None:
                out = proc.stdout.read() if proc.stdout else ""
                raise RuntimeError(f"tracera-api.exe erken kapandı (code={proc.returncode}):\n{out}")
            try:
                r = httpx.get("http://127.0.0.1:8765/health", timeout=2.0)
                if r.status_code == 200:
                    cold_start_s = time.perf_counter() - t_start
                    health_data = r.json()
                    break
            except Exception:
                time.sleep(0.5)
        else:
            raise TimeoutError("/health 120 sn içinde 200 dönmedi.")

        print(f"[HEALTH] soğuk açılış={cold_start_s:.2f} sn | yanıt={health_data}")

        img_bytes = SAMPLE_IMG.read_bytes()
        t_det = time.perf_counter()
        r_det = httpx.post(
            "http://127.0.0.1:8765/detect",
            files={"image": ("val_05.jpg", img_bytes, "image/jpeg")},
            timeout=60.0,
        )
        det_wall_ms = (time.perf_counter() - t_det) * 1000.0
        r_det.raise_for_status()
        det_json = r_det.json()
        print(
            f"[DETECT] HTTP {r_det.status_code} | backend={det_json['backend']} | "
            f"tespit={len(det_json['detections'])} | model_ms={det_json['timing']['model_ms']:.1f} ms | "
            f"wall={det_wall_ms:.1f} ms"
        )

        top_det = det_json["detections"][0]
        box_str = ",".join(str(v) for v in top_det["box"])
        t_exp = time.perf_counter()
        r_exp = httpx.post(
            "http://127.0.0.1:8765/explain",
            files={"image": ("val_05.jpg", img_bytes, "image/jpeg")},
            data={"box": box_str, "cls_id": str(top_det["cls_id"])},
            timeout=60.0,
        )
        exp_wall_ms = (time.perf_counter() - t_exp) * 1000.0
        r_exp.raise_for_status()
        exp_json = r_exp.json()
        ov_raw = base64.b64decode(exp_json["overlay_png"], validate=True)
        zm_raw = base64.b64decode(exp_json["zoom_png"], validate=True)
        assert ov_raw.startswith(b"\x89PNG\r\n\x1a\n")
        assert zm_raw.startswith(b"\x89PNG\r\n\x1a\n")
        print(
            f"[EXPLAIN] HTTP {r_exp.status_code} | method={exp_json['method']} layer={exp_json['layer']} | "
            f"overlay={len(ov_raw)} B, zoom={len(zm_raw)} B | wall={exp_wall_ms:.1f} ms"
        )

        r_met = httpx.get("http://127.0.0.1:8765/metrics", timeout=10.0)
        r_met.raise_for_status()
        met_keys = list(r_met.json().keys())
        print(f"[METRICS] HTTP {r_met.status_code} | anahtarlar={met_keys}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> None:
    build_sec = build_exe()
    size_mb, n_files = dir_size_mb(DIST_DIR)
    print(f"\n=== PyInstaller Derleme Özeti ===")
    print(f"Derleme süresi : {build_sec:.1f} sn")
    print(f"Klasör         : {DIST_DIR}")
    print(f"Toplam boyut   : {size_mb:.1f} MB ({size_mb / 1024:.2f} GB), {n_files} dosya")
    test_compiled_exe()


if __name__ == "__main__":
    main()
