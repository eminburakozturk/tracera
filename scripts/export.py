"""Tracera YOLO modellerini edge formatlarına (ONNX, OpenVINO FP32/INT8, TensorRT FP16/INT8) dışa aktarma scripti."""

from __future__ import annotations

import argparse
import shutil
import time
import traceback
from pathlib import Path

import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
VALID_FORMATS = ("onnx", "openvino", "engine", "engine_int8")


def resolve_data_yaml(data_yaml_path: Path, tag: str) -> Path:
    """data.yaml içindeki 'path' alanını repo köküne göre mutlak yola çevirip runs/ altına yazar."""
    if not data_yaml_path.is_absolute():
        data_yaml_path = (REPO_ROOT / data_yaml_path).resolve()

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_path = Path(cfg.get("path", "data/yolo/grouped"))
    if not raw_path.is_absolute():
        abs_dataset_path = (REPO_ROOT / raw_path).resolve()
    else:
        abs_dataset_path = raw_path.resolve()

    cfg["path"] = abs_dataset_path.as_posix()

    runs_dir = REPO_ROOT / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_yaml = runs_dir / f"data_resolved_export_{tag}.yaml"

    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    return out_yaml


def get_size_mb(path: Path) -> float:
    """Dosya veya klasörün toplam boyutunu MB cinsinden hesaplar."""
    if not path.exists():
        return 0.0
    if path.is_file():
        return round(path.stat().st_size / (1024 * 1024), 2)
    total_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total_bytes / (1024 * 1024), 2)


def export_onnx(weights_path: Path, imgsz: int) -> Path:
    """ONNX (opset 17, dynamic=False, simplify=True) formatına dışa aktarır."""
    model = YOLO(str(weights_path))
    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=17,
        dynamic=False,
        simplify=True,
    )
    return Path(exported)


def export_openvino_fp32(weights_path: Path, imgsz: int) -> Path:
    """OpenVINO FP32 formatına dışa aktarır."""
    model = YOLO(str(weights_path))
    exported = model.export(
        format="openvino",
        imgsz=imgsz,
        half=False,
        int8=False,
    )
    return Path(exported)


def export_openvino_int8(weights_path: Path, imgsz: int, resolved_data_yaml: Path) -> Path:
    """OpenVINO INT8 (split='train' kalibrasyonlu) formatına dışa aktarır."""
    model = YOLO(str(weights_path))
    exported = model.export(
        format="openvino",
        imgsz=imgsz,
        int8=True,
        data=str(resolved_data_yaml),
        split="train",
    )
    return Path(exported)


def export_tensorrt_fp16(weights_path: Path, imgsz: int, device: str) -> Path:
    """TensorRT FP16 (.engine) formatına dışa aktarır."""
    model = YOLO(str(weights_path))
    exported = model.export(
        format="engine",
        imgsz=imgsz,
        half=True,
        int8=False,
        dynamic=False,
        device=device,
    )
    return Path(exported)


def export_tensorrt_int8(
    weights_path: Path,
    imgsz: int,
    resolved_data_yaml: Path,
    device: str,
) -> Path:
    """TensorRT INT8 (split='train' kalibrasyonlu .engine) formatına dışa aktarır; FP16 dosyasını ezmemek için _int8.engine olarak kaydeder."""
    default_engine = weights_path.with_suffix(".engine")
    fp16_backup: Path | None = None
    if default_engine.exists():
        fp16_backup = weights_path.with_name(f"{weights_path.stem}_fp16_backup.engine")
        shutil.copy2(default_engine, fp16_backup)

    try:
        model = YOLO(str(weights_path))
        exported = Path(
            model.export(
                format="engine",
                imgsz=imgsz,
                int8=True,
                half=True,
                dynamic=False,
                data=str(resolved_data_yaml),
                split="train",
                device=device,
            )
        )
        int8_target = weights_path.with_name(f"{weights_path.stem}_int8.engine")
        if exported.resolve() != int8_target.resolve():
            shutil.move(str(exported), str(int8_target))
        return int8_target
    finally:
        if fp16_backup is not None and fp16_backup.exists():
            shutil.move(str(fp16_backup), str(default_engine))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tracera YOLO Edge Export Scripti (ONNX, OpenVINO FP32/INT8, TensorRT FP16/INT8)"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Dışa aktarılacak PyTorch model ağırlığı (.pt)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="Dışa aktarma görüntü boyutu (ör. 640 veya 1024; varsayılan: 1024)",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["onnx", "openvino", "engine"],
        choices=list(VALID_FORMATS),
        help="Dışa aktarılacak formatlar (varsayılan: onnx openvino engine; opsiyonel: engine_int8)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("configs/data_grouped.yaml"),
        help="INT8 kalibrasyonu için veri YAML dosyası (varsayılan: configs/data_grouped.yaml)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="TensorRT export için GPU cihaz kimliği (varsayılan: 0)",
    )
    parser.add_argument(
        "--trt-int8",
        action="store_true",
        help="Opsiyonel TensorRT INT8 dışa aktarımını da çalıştır",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    weights_path = args.weights if args.weights.is_absolute() else (REPO_ROOT / args.weights).resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"Ağırlık dosyası bulunamadı: {weights_path}")

    data_yaml = args.data
    if not data_yaml.is_absolute() and not (REPO_ROOT / data_yaml).exists():
        fallback = REPO_ROOT / "configs" / "data.yaml"
        if fallback.exists():
            data_yaml = fallback

    resolved_yaml = resolve_data_yaml(data_yaml, tag=f"{weights_path.stem}_{args.imgsz}")

    requested_formats = list(args.formats)
    if args.trt_int8 and "engine_int8" not in requested_formats:
        requested_formats.append("engine_int8")

    tasks: list[tuple[str, object]] = []
    for fmt in requested_formats:
        if fmt == "onnx":
            tasks.append(("ONNX (opset=17, simplify=True)", lambda: export_onnx(weights_path, args.imgsz)))
        elif fmt == "openvino":
            tasks.append(("OpenVINO FP32", lambda: export_openvino_fp32(weights_path, args.imgsz)))
            tasks.append(
                (
                    "OpenVINO INT8",
                    lambda: export_openvino_int8(weights_path, args.imgsz, resolved_yaml),
                )
            )
        elif fmt == "engine":
            tasks.append(
                (
                    "TensorRT FP16 (.engine)",
                    lambda: export_tensorrt_fp16(weights_path, args.imgsz, args.device),
                )
            )
        elif fmt == "engine_int8":
            tasks.append(
                (
                    "TensorRT INT8 (_int8.engine)",
                    lambda: export_tensorrt_int8(weights_path, args.imgsz, resolved_yaml, args.device),
                )
            )

    print("=" * 72)
    print(f"TRACERA EDGE EXPORT — weights={weights_path} | imgsz={args.imgsz}")
    print(f"Kalibrasyon verisi : {resolved_yaml}")
    print("=" * 72)

    summary: list[tuple[str, str, str, float, float]] = []
    for label, fn in tasks:
        print(f"\n>>> Başlatılıyor: {label} ...")
        t0 = time.perf_counter()
        try:
            out_path: Path = fn()  # type: ignore[operator]
            elapsed = round(time.perf_counter() - t0, 2)
            mb = get_size_mb(out_path)
            print(f"[BAŞARILI] {label} -> {out_path} ({mb:.2f} MB, {elapsed:.2f} sn)")
            summary.append((label, "BAŞARILI", str(out_path), mb, elapsed))
        except Exception as exc:
            elapsed = round(time.perf_counter() - t0, 2)
            print(f"[HATA] {label} dışa aktarımı başarısız oldu ({elapsed:.2f} sn): {exc}")
            traceback.print_exc()
            summary.append((label, f"HATA: {type(exc).__name__}", "-", 0.0, elapsed))

    print("\n" + "=" * 72)
    print(f"{'Format':<30} {'Durum':<20} {'Boyut (MB)':>10} {'Süre (sn)':>10}")
    print("-" * 72)
    for label, status, _, mb, elapsed in summary:
        print(f"{label:<30} {status:<20} {mb:>10.2f} {elapsed:>10.2f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
