"""Tracera tarayıcı (ONNX Runtime Web) modelleri ve meta.json dışa aktarım scripti."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = REPO_ROOT / "runs" / "detect" / "g_y11s_1024_full" / "weights" / "last.pt"
DEFAULT_OUT_DIR = REPO_ROOT / "web" / "public" / "models"

CONTRACT_META = {
    "version": "1",
    "model": "yolo11s",
    "imgsz": 1024,
    "conf": 0.395,
    "iou_nms": 0.7,
    "letterbox_pad": 114,
    "classes": [
        "missing_hole",
        "mouse_bite",
        "open_circuit",
        "short",
        "spur",
        "spurious_copper",
    ],
    "colors": [
        "#38bdf8",
        "#f472b6",
        "#facc15",
        "#f87171",
        "#a78bfa",
        "#34d399",
    ],
    "onnx": "tracera_1024.onnx",
}


def file_size_mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 * 1024), 2)


def export_fp32_onnx(weights_path: Path, out_onnx: Path, imgsz: int) -> Path:
    """last.pt'yi opset=17, dynamic=False, nms=False ile FP32 ONNX olarak dışa aktarır."""
    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_weights = Path(tmp_dir) / weights_path.name
        shutil.copy2(weights_path, tmp_weights)
        model = YOLO(str(tmp_weights))
        exported_path = Path(
            model.export(
                format="onnx",
                imgsz=imgsz,
                opset=17,
                dynamic=False,
                nms=False,
                simplify=True,
            )
        )
        shutil.copy2(exported_path, out_onnx)
    return out_onnx


def export_dynamic_int8_onnx(fp32_onnx: Path, out_int8_onnx: Path) -> Path:
    """FP32 ONNX modeline ONNX Runtime dynamic quantization (QUInt8) uygular."""
    out_int8_onnx.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        pre_path = Path(tmp_dir) / "preprocessed.onnx"
        quant_pre_process(
            input_model=str(fp32_onnx),
            output_model_path=str(pre_path),
            skip_optimization=False,
        )
        quantize_dynamic(
            model_input=str(pre_path),
            model_output=str(out_int8_onnx),
            weight_type=QuantType.QUInt8,
        )
    return out_int8_onnx


def write_meta_json(out_dir: Path) -> Path:
    """CONTRACT §1'e birebir uygun web/public/models/meta.json dosyasını yazar."""
    meta_path = out_dir / "meta.json"
    meta_path.write_text(
        json.dumps(CONTRACT_META, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return meta_path


def inspect_onnx_io(onnx_path: Path) -> tuple[list[int], list[tuple[str, list[int]]], list[tuple[str, list[int]]]]:
    m = onnx.load(str(onnx_path))
    opsets = [o.version for o in m.opset_import]
    inputs = [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input]
    outputs = [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output]
    return opsets, inputs, outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tracera tarayıcı (ONNX Runtime Web) model dışa aktarma")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--imgsz", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights_path = args.weights if args.weights.is_absolute() else (REPO_ROOT / args.weights).resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() else (REPO_ROOT / args.out_dir).resolve()

    if not weights_path.is_file():
        raise FileNotFoundError(f"Ağırlık dosyası bulunamadı: {weights_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = out_dir / f"tracera_{args.imgsz}.onnx"
    int8_path = out_dir / f"tracera_{args.imgsz}_int8.onnx"

    print(f"[1/3] FP32 ONNX dışa aktarılıyor: {weights_path} -> {fp32_path}")
    export_fp32_onnx(weights_path, fp32_path, imgsz=args.imgsz)

    print(f"[2/3] ORT Dynamic Quantization (INT8) uygulanıyor -> {int8_path}")
    export_dynamic_int8_onnx(fp32_path, int8_path)

    print(f"[3/3] meta.json yazılıyor -> {out_dir / 'meta.json'}")
    meta_path = write_meta_json(out_dir)

    fp32_opset, fp32_in, fp32_out = inspect_onnx_io(fp32_path)
    int8_opset, int8_in, int8_out = inspect_onnx_io(int8_path)

    print("\n=== Export Özeti ===")
    print(f"FP32 ONNX : {fp32_path} | {file_size_mb(fp32_path):.2f} MB | opset={fp32_opset} | in={fp32_in} | out={fp32_out}")
    print(f"INT8 ONNX : {int8_path} | {file_size_mb(int8_path):.2f} MB | opset={int8_opset} | in={int8_in} | out={int8_out}")
    print(f"meta.json : {meta_path} | {meta_path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
