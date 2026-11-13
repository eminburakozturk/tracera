"""ADIM 5: tracera-build (CPU PyTorch) ortamıyla PyInstaller onedir derleme ve ölçüm scripti."""

from __future__ import annotations

import argparse
import base64
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import httpx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DIST_DIR = REPO_ROOT / "dist" / "tracera-api"
BUILD_DIR = REPO_ROOT / "build"
EXE_PATH = DIST_DIR / "tracera-api.exe"
SAMPLE_IMG = REPO_ROOT / "app" / "examples" / "val_05_missing_hole_01.jpg"
DEFAULT_BUILD_PY = Path.home() / "anaconda3" / "envs" / "tracera-build" / "python.exe"

EXCLUDED_MODULES = [
    "gradio",
    "gradio_client",
    "tensorrt",
    "tensorrt_libs",
    "mlflow",
    "onnxruntime",
    "polars",
    "_polars_runtime_32",
    "polars_runtime_32",
    "pyarrow",
    "tkinter",
    "_tkinter",
    "IPython",
    "jupyter",
    "notebook",
    "pytest",
]


def dir_size_mb(path: Path) -> tuple[float, int]:
    files = [f for f in path.rglob("*") if f.is_file()]
    total_bytes = sum(f.stat().st_size for f in files)
    return total_bytes / (1024 * 1024), len(files)


def top_subdirs_mb(root: Path, top_k: int = 10) -> list[tuple[str, float]]:
    if not root.is_dir():
        return []
    rows: list[tuple[float, str]] = []
    for p in root.iterdir():
        if p.is_file():
            sz = p.stat().st_size
        else:
            sz = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        rows.append((sz / (1024 * 1024), p.name))
    rows.sort(reverse=True)
    return [(name, mb) for mb, name in rows[:top_k]]


def percentile_nearest(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil((p / 100.0) * len(ordered)) - 1))
    return ordered[idx]


def median_val(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.median(np.array(values, dtype=np.float64)))


def build_exe(python_exe: Path) -> float:
    sep = ";" if os.name == "nt" else ":"
    cmd = [
        str(python_exe),
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
        f"{REPO_ROOT / 'scripts' / 'gradcam.py'}{sep}scripts",
        "--collect-all",
        "openvino",
        "--collect-all",
        "torchvision",
        "--collect-data",
        "ultralytics",
        "--hidden-import",
        "tracera_core",
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
    ]
    for mod in EXCLUDED_MODULES:
        cmd.extend(["--exclude-module", mod])
    cmd.append(str(REPO_ROOT / "api" / "main.py"))

    print(f"PyInstaller başlatılıyor ({python_exe})...")
    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    return time.perf_counter() - t0


def test_and_benchmark_exe(warm_n: int = 20, port: int = 18765) -> None:
    from tracera_core import detect_openvino

    if not EXE_PATH.is_file():
        raise FileNotFoundError(f"Derlenmiş EXE bulunamadı: {EXE_PATH}")

    base_url = f"http://127.0.0.1:{port}"
    print(f"\nDerlenmiş EXE başlatılıyor: {EXE_PATH} (port={port})")
    t_start = time.perf_counter()
    proc = subprocess.Popen(
        [str(EXE_PATH), "--port", str(port)],
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
                r = httpx.get(f"{base_url}/health", timeout=2.0)
                if r.status_code == 200:
                    cold_start_s = time.perf_counter() - t_start
                    health_data = r.json()
                    break
            except Exception:
                time.sleep(0.25)
        else:
            raise TimeoutError("/health 60 sn içinde 200 dönmedi.")

        print(f"[HEALTH] soğuk açılış={cold_start_s:.2f} sn | yanıt={health_data}")

        img_bytes = SAMPLE_IMG.read_bytes()
        img_bgr = cv2.imread(str(SAMPLE_IMG))
        assert img_bgr is not None
        py_ref = detect_openvino(img_bgr)

        with httpx.Client(timeout=60.0) as client:
            # 1) İlk /detect isteği (ayrı)
            t0 = time.perf_counter()
            r_det_first = client.post(
                f"{base_url}/detect",
                files={"image": ("val_05.jpg", img_bytes, "image/jpeg")},
            )
            det_first_wall = (time.perf_counter() - t0) * 1000.0
            r_det_first.raise_for_status()
            det_json = r_det_first.json()

            # Python parity doğrulaması
            assert len(det_json["detections"]) == len(py_ref.detections), (
                f"Tespit sayısı uyuşmuyor: EXE={len(det_json['detections'])} vs PY={len(py_ref.detections)}"
            )
            for d_exe, d_py in zip(det_json["detections"], py_ref.detections):
                assert d_exe["cls_id"] == d_py.cls_id
                assert abs(d_exe["conf"] - d_py.conf) < 1e-4
                assert max(abs(a - b) for a, b in zip(d_exe["box"], d_py.box)) < 0.05
            max_conf_exe = max(d["conf"] for d in det_json["detections"])
            max_conf_py = max(d.conf for d in py_ref.detections)
            print(
                f"[PARITY] Python ile birebir eşleşti: tespit={len(det_json['detections'])}, "
                f"max_conf EXE={max_conf_exe:.6f} vs PY={max_conf_py:.6f}"
            )
            print(
                f"[DETECT #1] wall={det_first_wall:.1f} ms | "
                f"api_total={det_json['timing']['total_ms']:.1f} ms (model={det_json['timing']['model_ms']:.1f} ms)"
            )

            # 2) n=20 sıcak /detect
            det_walls: list[float] = []
            det_models: list[float] = []
            for _ in range(warm_n):
                t_w = time.perf_counter()
                rw = client.post(
                    f"{base_url}/detect",
                    files={"image": ("val_05.jpg", img_bytes, "image/jpeg")},
                )
                det_walls.append((time.perf_counter() - t_w) * 1000.0)
                rw.raise_for_status()
                det_models.append(float(rw.json()["timing"]["model_ms"]))
            print(
                f"[DETECT warm n={warm_n}] wall medyan={median_val(det_walls):.1f} ms, p95={percentile_nearest(det_walls, 95):.1f} ms | "
                f"model medyan={median_val(det_models):.1f} ms, p95={percentile_nearest(det_models, 95):.1f} ms"
            )

            # 3) İlk /explain isteği (ayrı)
            top_det = det_json["detections"][0]
            box_str = ",".join(str(v) for v in top_det["box"])
            t1 = time.perf_counter()
            r_exp_first = client.post(
                f"{base_url}/explain",
                files={"image": ("val_05.jpg", img_bytes, "image/jpeg")},
                data={"box": box_str, "cls_id": str(top_det["cls_id"])},
            )
            exp_first_wall = (time.perf_counter() - t1) * 1000.0
            r_exp_first.raise_for_status()
            exp_json = r_exp_first.json()
            ov_raw = base64.b64decode(exp_json["overlay_png"], validate=True)
            zm_raw = base64.b64decode(exp_json["zoom_png"], validate=True)
            assert ov_raw.startswith(b"\x89PNG\r\n\x1a\n") and zm_raw.startswith(b"\x89PNG\r\n\x1a\n")
            print(
                f"[EXPLAIN #1] wall={exp_first_wall:.1f} ms | "
                f"api_total={exp_json['timing']['total_ms']:.1f} ms (model={exp_json['timing']['model_ms']:.1f} ms) | "
                f"overlay={len(ov_raw)} B, zoom={len(zm_raw)} B"
            )

            # 4) n=20 sıcak /explain
            exp_walls: list[float] = []
            exp_models: list[float] = []
            for _ in range(warm_n):
                t_w = time.perf_counter()
                rw = client.post(
                    f"{base_url}/explain",
                    files={"image": ("val_05.jpg", img_bytes, "image/jpeg")},
                    data={"box": box_str, "cls_id": str(top_det["cls_id"])},
                )
                exp_walls.append((time.perf_counter() - t_w) * 1000.0)
                rw.raise_for_status()
                exp_models.append(float(rw.json()["timing"]["model_ms"]))
            print(
                f"[EXPLAIN warm n={warm_n}] wall medyan={median_val(exp_walls):.1f} ms, p95={percentile_nearest(exp_walls, 95):.1f} ms | "
                f"model medyan={median_val(exp_models):.1f} ms, p95={percentile_nearest(exp_models, 95):.1f} ms"
            )

            # 5) /metrics kontrolü
            r_met = client.get(f"{base_url}/metrics")
            r_met.raise_for_status()
            print(f"[METRICS] HTTP {r_met.status_code} | anahtarlar={list(r_met.json().keys())}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description="tracera-api PyInstaller onedir derleme ve ölçüm")
    parser.add_argument(
        "--python",
        type=Path,
        default=DEFAULT_BUILD_PY if DEFAULT_BUILD_PY.is_file() else Path(sys.executable),
        help="PyInstaller için Python yorumlayıcı yolu (varsayılan: tracera-build)",
    )
    parser.add_argument("--warm-n", type=int, default=20)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    build_sec = 0.0 if args.skip_build else build_exe(args.python)
    size_mb, n_files = dir_size_mb(DIST_DIR)
    print("\n=== PyInstaller Derleme Özeti ===")
    if not args.skip_build:
        print(f"Derleme süresi : {build_sec:.1f} sn")
    print(f"Klasör         : {DIST_DIR}")
    print(f"Toplam boyut   : {size_mb:.1f} MB ({size_mb / 1024:.2f} GB), {n_files} dosya")
    print("En büyük 10 alt klasör (_internal):")
    for name, mb in top_subdirs_mb(DIST_DIR / "_internal", top_k=10):
        print(f"  {mb:8.1f} MB  {name}")

    test_and_benchmark_exe(warm_n=args.warm_n)


if __name__ == "__main__":
    main()
