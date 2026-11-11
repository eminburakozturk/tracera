"""Tracera YOLO modellerinin farklı edge formatlarında (PyTorch, ONNX CPU/CUDA, OpenVINO FP32/INT8, TensorRT FP16/INT8)
boyut (MB), gecikme/FPS (batch=1, 20 ısınma + 200 ölçüm; ön/son işlem hariç ve dahil ortalama/medyan/p95 ms, medyandan FPS)
ve doğrulama (val) mAP kıyaslama scripti.

ÖNEMLİ: Değerlendirme yalnızca split='val' üzerinde yapılır (TEST KULLANILMAZ).
"""

from __future__ import annotations

import argparse
import os
import platform
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

# Windows'ta onnxruntime-gpu'nun PyTorch CUDA/cuDNN DLL'lerini bulabilmesi için:
if os.name == "nt":
    _torch_lib = Path(torch.__file__).resolve().parent / "lib"
    if _torch_lib.is_dir() and hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(str(_torch_lib))
        except OSError:
            pass

import ultralytics
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class BenchmarkTarget:
    format_name: str
    backend_key: str
    artifact_path: Path
    device: str
    is_gpu: bool


@dataclass(frozen=True)
class LatencyAndValMetrics:
    inf_mean_ms: float
    inf_median_ms: float
    inf_p95_ms: float
    e2e_mean_ms: float
    e2e_median_ms: float
    e2e_p95_ms: float
    fps_infer_only: float
    fps_end2end: float
    map50: float
    map50_95: float


def get_package_version(module_name: str) -> str:
    """İlgili paketin yüklü sürüm numarasını döndürür (yoksa '-')."""
    try:
        mod = __import__(module_name)
        return str(getattr(mod, "__version__", "-"))
    except Exception:
        return "-"


def get_cpu_model_name() -> str:
    """İşletim sisteminden gerçek CPU model adını okur."""
    if os.name == "nt":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            cpu_name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            winreg.CloseKey(key)
            if cpu_name and str(cpu_name).strip():
                return str(cpu_name).strip()
        except Exception:
            pass
    elif Path("/proc/cpuinfo").is_file():
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore").splitlines():
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return platform.processor() or platform.machine() or "CPU"


def get_gpu_model_name(device_idx: int = 0) -> str:
    """PyTorch üzerinden aktif GPU model adını döndürür."""
    if torch.cuda.is_available():
        try:
            return str(torch.cuda.get_device_name(device_idx)).strip()
        except Exception:
            return "CUDA GPU"
    return "GPU Yok (CPU)"


def get_size_mb(path: Path) -> float:
    """Dosya veya klasörün toplam boyutunu MB cinsinden döndürür."""
    if not path.exists():
        return 0.0
    if path.is_file():
        return round(path.stat().st_size / (1024 * 1024), 2)
    total_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total_bytes / (1024 * 1024), 2)


def resolve_data_yaml(data_yaml_path: Path, tag: str) -> tuple[Path, Path]:
    """data.yaml içindeki 'path' alanını repo köküne göre mutlak yola çevirir ve val görüntü dizinini döndürür."""
    if not data_yaml_path.is_absolute():
        data_yaml_path = (REPO_ROOT / data_yaml_path).resolve()

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_path = Path(cfg.get("path", "data/yolo/grouped"))
    abs_dataset_path = raw_path.resolve() if raw_path.is_absolute() else (REPO_ROOT / raw_path).resolve()
    cfg["path"] = abs_dataset_path.as_posix()

    val_rel = cfg.get("val", "val/images")
    val_img_dir = (abs_dataset_path / val_rel).resolve()

    runs_dir = REPO_ROOT / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_yaml = runs_dir / f"data_resolved_bench_{tag}.yaml"

    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    return out_yaml, val_img_dir


def load_val_images(val_img_dir: Path, count: int) -> list[np.ndarray]:
    """Val setinden görüntüleri belleğe yükler ve istenen ölçüm sayısına (ör. 220) tamamlar."""
    img_paths = sorted(
        [
            p
            for p in val_img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ],
        key=lambda p: p.name,
    )
    if not img_paths:
        raise FileNotFoundError(f"Val görüntü klasöründe görüntü bulunamadı: {val_img_dir}")

    loaded: list[np.ndarray] = []
    for p in img_paths:
        im = cv2.imread(str(p))
        if im is not None:
            loaded.append(im)
        if len(loaded) >= count:
            break

    if not loaded:
        raise RuntimeError(f"Val görüntüleri okunamadı: {val_img_dir}")

    result: list[np.ndarray] = []
    idx = 0
    while len(result) < count:
        result.append(loaded[idx % len(loaded)])
        idx += 1
    return result


def discover_targets(weights_path: Path, requested_formats: list[str], gpu_device: str) -> list[BenchmarkTarget]:
    """Ağırlık dosyasının bulunduğu klasördeki dışa aktarılmış modelleri hedeflere dönüştürür."""
    stem = weights_path.stem
    parent = weights_path.parent
    targets: list[BenchmarkTarget] = []

    for fmt in requested_formats:
        if fmt == "pt":
            targets.append(
                BenchmarkTarget(
                    format_name="PyTorch FP32 (GPU)",
                    backend_key="pt_fp32_gpu",
                    artifact_path=weights_path,
                    device=gpu_device,
                    is_gpu=True,
                )
            )
        elif fmt == "onnx":
            onnx_path = parent / f"{stem}.onnx"
            targets.append(
                BenchmarkTarget(
                    format_name="ONNX Runtime (CPU)",
                    backend_key="onnx_cpu",
                    artifact_path=onnx_path,
                    device="cpu",
                    is_gpu=False,
                )
            )
            targets.append(
                BenchmarkTarget(
                    format_name="ONNX Runtime (CUDA)",
                    backend_key="onnx_cuda",
                    artifact_path=onnx_path,
                    device=gpu_device,
                    is_gpu=True,
                )
            )
        elif fmt == "openvino":
            ov_fp32_dir = parent / f"{stem}_openvino_model"
            ov_int8_dir = parent / f"{stem}_int8_openvino_model"
            targets.append(
                BenchmarkTarget(
                    format_name="OpenVINO FP32 (CPU)",
                    backend_key="openvino_fp32_cpu",
                    artifact_path=ov_fp32_dir,
                    device="cpu",
                    is_gpu=False,
                )
            )
            targets.append(
                BenchmarkTarget(
                    format_name="OpenVINO INT8 (CPU)",
                    backend_key="openvino_int8_cpu",
                    artifact_path=ov_int8_dir,
                    device="cpu",
                    is_gpu=False,
                )
            )
        elif fmt == "engine":
            trt_fp16 = parent / f"{stem}.engine"
            targets.append(
                BenchmarkTarget(
                    format_name="TensorRT FP16 (GPU)",
                    backend_key="tensorrt_fp16_gpu",
                    artifact_path=trt_fp16,
                    device=gpu_device,
                    is_gpu=True,
                )
            )
            trt_int8 = parent / f"{stem}_int8.engine"
            if trt_int8.exists():
                targets.append(
                    BenchmarkTarget(
                        format_name="TensorRT INT8 (GPU)",
                        backend_key="tensorrt_int8_gpu",
                        artifact_path=trt_int8,
                        device=gpu_device,
                        is_gpu=True,
                    )
                )
        elif fmt == "engine_int8":
            trt_int8 = parent / f"{stem}_int8.engine"
            targets.append(
                BenchmarkTarget(
                    format_name="TensorRT INT8 (GPU)",
                    backend_key="tensorrt_int8_gpu",
                    artifact_path=trt_int8,
                    device=gpu_device,
                    is_gpu=True,
                )
            )

    return targets


def measure_fps_and_val(
    target: BenchmarkTarget,
    imgsz: int,
    warmup: int,
    runs: int,
    val_frames: list[np.ndarray],
    resolved_yaml: Path,
) -> LatencyAndValMetrics:
    """Tek bir format için (1) 20 ısınma + 200 ölçüm gecikme (ortalama, medyan, p95; medyandan FPS) ve (2) split='val' mAP hesaplar."""
    if not target.artifact_path.exists():
        raise FileNotFoundError(f"Model dosyası/klasörü bulunamadı: {target.artifact_path}")

    model = YOLO(str(target.artifact_path), task="detect")

    # 1. Isınma (Warmup)
    for i in range(warmup):
        model.predict(
            source=val_frames[i],
            imgsz=imgsz,
            batch=1,
            device=target.device,
            verbose=False,
        )
        if i == 0 and target.backend_key == "onnx_cuda":
            backend_obj = getattr(getattr(model, "predictor", None), "model", None)
            session = getattr(backend_obj, "session", None)
            if session is not None:
                active_providers = session.get_providers()
                if "CUDAExecutionProvider" not in active_providers:
                    raise RuntimeError(
                        f"ONNX CUDA istendi fakat CUDAExecutionProvider aktif değil! Aktif: {active_providers}"
                    )

    if target.is_gpu and torch.cuda.is_available():
        torch.cuda.synchronize()

    # 2. Ölçüm (batch=1, 200 iterasyon)
    infer_only_ms_list: list[float] = []
    end2end_ms_list: list[float] = []

    for i in range(runs):
        frame = val_frames[warmup + i]
        t0 = time.perf_counter()
        preds = model.predict(
            source=frame,
            imgsz=imgsz,
            batch=1,
            device=target.device,
            verbose=False,
        )
        if target.is_gpu and torch.cuda.is_available():
            torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t0) * 1000.0

        spd = preds[0].speed
        inf_ms = float(spd.get("inference", wall_ms))
        pre_ms = float(spd.get("preprocess", 0.0))
        post_ms = float(spd.get("postprocess", 0.0))
        total_spd_ms = pre_ms + inf_ms + post_ms if (pre_ms + post_ms) > 0 else wall_ms

        infer_only_ms_list.append(inf_ms)
        end2end_ms_list.append(total_spd_ms)

    inf_arr = np.asarray(infer_only_ms_list, dtype=np.float64)
    e2e_arr = np.asarray(end2end_ms_list, dtype=np.float64)

    inf_mean_ms = float(np.mean(inf_arr))
    inf_median_ms = float(np.median(inf_arr))
    inf_p95_ms = float(np.percentile(inf_arr, 95))

    e2e_mean_ms = float(np.mean(e2e_arr))
    e2e_median_ms = float(np.median(e2e_arr))
    e2e_p95_ms = float(np.percentile(e2e_arr, 95))

    # FPS medyandan hesaplanır
    fps_infer_only = 1000.0 / inf_median_ms if inf_median_ms > 0 else 0.0
    fps_end2end = 1000.0 / e2e_median_ms if e2e_median_ms > 0 else 0.0

    # 3. Doğrulama mAP (YALNIZCA split="val", TEST DEĞİL!)
    eval_split = "val"
    assert eval_split == "val", "Benchmark sırasında test seti KULLANILAMAZ!"
    project_dir = (REPO_ROOT / "runs" / "benchmark").resolve()
    val_model = YOLO(str(target.artifact_path), task="detect")
    metrics = val_model.val(
        data=str(resolved_yaml),
        split=eval_split,
        imgsz=imgsz,
        batch=1,
        device=target.device,
        workers=2,
        project=str(project_dir),
        name=f"{target.backend_key}_{imgsz}_val",
        exist_ok=True,
        verbose=False,
    )
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)

    return LatencyAndValMetrics(
        inf_mean_ms=inf_mean_ms,
        inf_median_ms=inf_median_ms,
        inf_p95_ms=inf_p95_ms,
        e2e_mean_ms=e2e_mean_ms,
        e2e_median_ms=e2e_median_ms,
        e2e_p95_ms=e2e_p95_ms,
        fps_infer_only=fps_infer_only,
        fps_end2end=fps_end2end,
        map50=map50,
        map50_95=map50_95,
    )


def upsert_benchmark_csv(new_rows: list[dict[str, object]], out_csv: Path) -> pd.DataFrame:
    """Yeni benchmark satırlarını results/edge_benchmark.csv dosyasına ekler veya günceller."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict[str, object]] = []
    if out_csv.is_file():
        df_old = pd.read_csv(out_csv)
        existing_rows = df_old.to_dict(orient="records")

    new_keys = {(str(r["model"]), int(r["imgsz"]), str(r["format"])) for r in new_rows}  # type: ignore[arg-type]
    filtered = [
        r
        for r in existing_rows
        if (str(r.get("model")), int(r.get("imgsz", 0)), str(r.get("format"))) not in new_keys
    ]
    combined = filtered + new_rows
    df_out = pd.DataFrame(combined)
    df_out.to_csv(out_csv, index=False, encoding="utf-8")
    return df_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tracera Edge Benchmark Scripti (Boyut MB, Ortalama/Medyan/p95 Gecikme, Medyan FPS, Val mAP)"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Referans PyTorch model ağırlığı (.pt); diğer formatlar aynı klasörden otomatik bulunur",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("configs/data_grouped.yaml"),
        help="Val değerlendirmesi için veri YAML dosyası (varsayılan: configs/data_grouped.yaml)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="Görüntü boyutu (ör. 640 veya 1024; varsayılan: 1024)",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pt", "onnx", "openvino", "engine"],
        choices=["pt", "onnx", "openvino", "engine", "engine_int8"],
        help="Ölçülecek format grupları (varsayılan: pt onnx openvino engine)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Isınma iterasyon sayısı (varsayılan: 20)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=200,
        help="Ölçüm iterasyon sayısı (varsayılan: 200)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="GPU hedefleri için CUDA cihaz kimliği (varsayılan: 0)",
    )
    parser.add_argument(
        "--power",
        type=str,
        choices=["plugged", "battery"],
        default="plugged",
        help="Güç durumu: plugged (prize takılı) veya battery (pil) (varsayılan: plugged)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/edge_benchmark.csv"),
        help="Çıktı CSV yolu (varsayılan: results/edge_benchmark.csv)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Tabloda görünecek model etiketi (varsayılan: üst klasör / ağırlık adı)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    weights_path = args.weights if args.weights.is_absolute() else (REPO_ROOT / args.weights).resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"Referans ağırlık dosyası bulunamadı: {weights_path}")

    data_yaml = args.data
    if not data_yaml.is_absolute() and not (REPO_ROOT / data_yaml).exists():
        fallback = REPO_ROOT / "configs" / "data.yaml"
        if fallback.exists():
            data_yaml = fallback

    model_label = args.model_name or (
        weights_path.parent.parent.name
        if weights_path.parent.name == "weights"
        else weights_path.stem
    )

    resolved_yaml, val_img_dir = resolve_data_yaml(data_yaml, tag=f"{model_label}_{args.imgsz}")
    val_frames = load_val_images(val_img_dir, count=args.warmup + args.runs)

    cpu_hw = get_cpu_model_name()
    gpu_hw = get_gpu_model_name(0)
    ultralytics_ver = str(ultralytics.__version__)
    torch_ver = str(torch.__version__)
    ort_ver = get_package_version("onnxruntime")
    ov_ver = get_package_version("openvino")
    num_threads = int(torch.get_num_threads())

    targets = discover_targets(weights_path, list(args.formats), gpu_device=args.device)

    print("=" * 88)
    print(f"TRACERA EDGE BENCHMARK — model={model_label} | imgsz={args.imgsz} | split=val | güç={args.power}")
    print(f"CPU Donanımı : {cpu_hw} (torch threads={num_threads})")
    print(f"GPU Donanımı : {gpu_hw}")
    print(
        f"Sürümler     : ultralytics={ultralytics_ver}, torch={torch_ver}, "
        f"onnxruntime={ort_ver}, openvino={ov_ver}"
    )
    print(f"Protokol     : batch=1, {args.warmup} ısınma + {args.runs} ölçüm (FPS medyandan)")
    print("=" * 88)

    rows: list[dict[str, object]] = []
    for target in targets:
        hw_name = gpu_hw if target.is_gpu else cpu_hw
        print(f"\n>>> Ölçülüyor: {target.format_name} ({target.artifact_path.name}) [{hw_name}] ...")
        try:
            mb = get_size_mb(target.artifact_path)
            m = measure_fps_and_val(
                target=target,
                imgsz=args.imgsz,
                warmup=args.warmup,
                runs=args.runs,
                val_frames=val_frames,
                resolved_yaml=resolved_yaml,
            )
            row: dict[str, object] = {
                "model": model_label,
                "imgsz": args.imgsz,
                "format": target.format_name,
                "boyut_mb": round(mb, 2),
                "gecikme_inf_ort_ms": round(m.inf_mean_ms, 2),
                "gecikme_inf_medyan_ms": round(m.inf_median_ms, 2),
                "gecikme_inf_p95_ms": round(m.inf_p95_ms, 2),
                "gecikme_toplam_ort_ms": round(m.e2e_mean_ms, 2),
                "gecikme_toplam_medyan_ms": round(m.e2e_median_ms, 2),
                "gecikme_toplam_p95_ms": round(m.e2e_p95_ms, 2),
                "fps_sadece_inference": round(m.fps_infer_only, 1),
                "fps_on_son_islem_dahil": round(m.fps_end2end, 1),
                "mAP50": round(m.map50, 4),
                "mAP50-95": round(m.map50_95, 4),
                "donanım": hw_name,
                "güç": args.power,
                "cpu_threads": num_threads,
                "ultralytics_ver": ultralytics_ver,
                "torch_ver": torch_ver,
                "onnxruntime_ver": ort_ver,
                "openvino_ver": ov_ver,
            }
            rows.append(row)
            print(
                f"[TAMAM] {target.format_name}: {mb:.2f} MB | "
                f"FPS(inf,med)={m.fps_infer_only:.1f} (med={m.inf_median_ms:.2f}, ort={m.inf_mean_ms:.2f}, p95={m.inf_p95_ms:.2f} ms) | "
                f"FPS(dahil,med)={m.fps_end2end:.1f} (med={m.e2e_median_ms:.2f}, ort={m.e2e_mean_ms:.2f}, p95={m.e2e_p95_ms:.2f} ms) | "
                f"mAP50={m.map50:.4f} | mAP50-95={m.map50_95:.4f}"
            )
        except Exception as exc:
            print(f"[ATLANDI / HATA] {target.format_name}: {exc}")
            traceback.print_exc()

    if rows:
        out_csv = args.out if args.out.is_absolute() else (REPO_ROOT / args.out).resolve()
        df_all = upsert_benchmark_csv(rows, out_csv)
        print("\n" + "=" * 88)
        print(f"GÜNCEL EDGE BENCHMARK TABLOSU ({out_csv})")
        print("=" * 88)
        print(df_all.to_string(index=False))
    else:
        print("\n[UYARI] Hiçbir hedef başarıyla ölçülemedi.")


if __name__ == "__main__":
    main()
