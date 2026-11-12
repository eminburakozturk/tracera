"""Tracera — Hugging Face Spaces için kendi kendine yeten paket oluşturucu ve izole test scripti.

Bu script:
1. space/ klasörünü sıfırdan oluşturur.
2. app.py, tracera_xai/gradcam.py, weights/ (last.pt + last_int8_openvino_model),
   examples/ (4 val görüntüsü), results/ (operating_point.csv + edge_benchmark.csv) kopyalar.
3. requirements.txt dosyasını ortamdaki gerçek sürümlerle ve CPU torch derlemesiyle üretir.
4. README.md dosyasını Hugging Face frontmatter metadata'sı ile yazar.
5. Klasörün toplam boyutunu ve paketlenen kütüphane sürümlerini raporlar.
6. Geçici bir dizinde repo dışı izole CPU testi çalıştırarak tespit + LayerCAM doğrulaması yapar.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# -----------------------------------------------------------------------------
# Paketleme Kaynak Yolları
# -----------------------------------------------------------------------------
SRC_APP = REPO_ROOT / "app" / "app.py"
SRC_GRADCAM = REPO_ROOT / "scripts" / "gradcam.py"
SRC_WEIGHTS_PT = REPO_ROOT / "runs" / "detect" / "g_y11s_1024_full" / "weights" / "last.pt"
SRC_WEIGHTS_OV = REPO_ROOT / "runs" / "detect" / "g_y11s_1024_full" / "weights" / "last_int8_openvino_model"
SRC_EXAMPLES_DIR = REPO_ROOT / "app" / "examples"
SRC_OPERATING_POINT = REPO_ROOT / "results" / "operating_point.csv"
SRC_EDGE_BENCHMARK = REPO_ROOT / "results" / "edge_benchmark.csv"

# 4 adet val örneği (Kart 05 ve 11; test kartları 06 ve 10 kesinlikle hariç)
EXAMPLE_FILENAMES = (
    "val_05_missing_hole_01.jpg",
    "val_05_short_01.jpg",
    "val_11_mouse_bite_01.jpg",
    "val_11_spur_01.jpg",
)


def get_clean_version(pkg_name: str) -> str:
    """Paketin kurulu sürümünü alır; torch yerel etiketini (+cu126 vb.) temizler."""
    raw_ver = importlib.metadata.version(pkg_name)
    return raw_ver.split("+")[0]


def get_package_versions() -> dict[str, str]:
    """Tracera ortamındaki gerçek paket sürümlerini toplar."""
    pkgs = [
        "torch",
        "torchvision",
        "ultralytics",
        "openvino",
        "gradio",
        "numpy",
        "opencv-python-headless",
        "pandas",
        "scipy",
        "matplotlib",
    ]
    versions: dict[str, str] = {}
    for pkg in pkgs:
        try:
            versions[pkg] = get_clean_version(pkg)
        except importlib.metadata.PackageNotFoundError:
            # opencv fallback kontrolü
            if pkg == "opencv-python-headless":
                try:
                    versions[pkg] = get_clean_version("opencv-python")
                except Exception:
                    versions[pkg] = "4.10.0.84"
            else:
                raise
    return versions


def build_space(output_dir: Path) -> dict[str, object]:
    """space/ klasörünü tüm bağımlılıkları ve modelleriyle inşa eder."""
    print(f"📦 [build_space] Dağıtım paketi oluşturuluyor: {output_dir}")

    if output_dir.exists():
        print(f"🧹 [build_space] Eski dizin temizleniyor: {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dizin hiyerarşisi
    xai_dir = output_dir / "tracera_xai"
    weights_dir = output_dir / "weights"
    examples_dir = output_dir / "examples"
    results_dir = output_dir / "results"

    for d in (xai_dir, weights_dir, examples_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 2. app.py
    print("📄 [build_space] app.py kopyalanıyor...")
    shutil.copy2(SRC_APP, output_dir / "app.py")

    # 3. tracera_xai/gradcam.py ve __init__.py
    print("🔬 [build_space] tracera_xai modülü oluşturuluyor...")
    shutil.copy2(SRC_GRADCAM, xai_dir / "gradcam.py")
    init_content = (
        '"""Tracera Açıklanabilir Yapay Zeka (XAI) Modülü."""\n'
        "from .gradcam import MultiLayerYoloCAM, box_orig_to_lb, letterbox_image, make_overlay_and_zoom\n\n"
        '__all__ = ["MultiLayerYoloCAM", "box_orig_to_lb", "letterbox_image", "make_overlay_and_zoom"]\n'
    )
    (xai_dir / "__init__.py").write_text(init_content, encoding="utf-8")

    # 4. Modeller: PyTorch last.pt ve OpenVINO INT8
    print("⚖️ [build_space] Ağırlıklar kopyalanıyor (last.pt + OpenVINO INT8)...")
    if not SRC_WEIGHTS_PT.is_file():
        raise FileNotFoundError(f"PyTorch ağırlığı bulunamadı: {SRC_WEIGHTS_PT}")
    shutil.copy2(SRC_WEIGHTS_PT, weights_dir / "last.pt")

    if not SRC_WEIGHTS_OV.is_dir():
        raise FileNotFoundError(f"OpenVINO INT8 modeli bulunamadı: {SRC_WEIGHTS_OV}")
    shutil.copytree(SRC_WEIGHTS_OV, weights_dir / "last_int8_openvino_model")

    # 5. Örnekler (4 adet val görüntüsü)
    print("🖼️ [build_space] Val örnek görüntüleri kopyalanıyor...")
    for fn in EXAMPLE_FILENAMES:
        src_ex = SRC_EXAMPLES_DIR / fn
        if not src_ex.is_file():
            raise FileNotFoundError(f"Örnek görsel bulunamadı: {src_ex}")
        shutil.copy2(src_ex, examples_dir / fn)

    # 6. Raporlanmış metrik tabloları
    print("📊 [build_space] Metrik tabloları (results/) kopyalanıyor...")
    if not SRC_OPERATING_POINT.is_file():
        raise FileNotFoundError(f"Operating point tablosu bulunamadı: {SRC_OPERATING_POINT}")
    shutil.copy2(SRC_OPERATING_POINT, results_dir / "operating_point.csv")

    if not SRC_EDGE_BENCHMARK.is_file():
        raise FileNotFoundError(f"Edge benchmark tablosu bulunamadı: {SRC_EDGE_BENCHMARK}")
    shutil.copy2(SRC_EDGE_BENCHMARK, results_dir / "edge_benchmark.csv")

    # 7. requirements.txt
    print("📋 [build_space] requirements.txt üretiliyor...")
    pkg_versions = get_package_versions()
    req_lines = [
        "--extra-index-url https://download.pytorch.org/whl/cpu",
        f"torch=={pkg_versions['torch']}",
        f"torchvision=={pkg_versions['torchvision']}",
        f"ultralytics=={pkg_versions['ultralytics']}",
        f"openvino=={pkg_versions['openvino']}",
        f"gradio=={pkg_versions['gradio']}",
        f"numpy=={pkg_versions['numpy']}",
        f"opencv-python-headless=={pkg_versions['opencv-python-headless']}",
        f"pandas=={pkg_versions['pandas']}",
        f"scipy=={pkg_versions['scipy']}",
        f"matplotlib=={pkg_versions['matplotlib']}",
        "",
    ]
    (output_dir / "requirements.txt").write_text("\n".join(req_lines), encoding="utf-8")

    # 8. README.md (Hugging Face Spaces frontmatter)
    print("📝 [build_space] README.md (HF metadata) üretiliyor...")
    gradio_ver = pkg_versions["gradio"]
    readme_content = f"""---
title: Tracera
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: {gradio_ver}
app_file: app.py
pinned: false
python_version: 3.11
license: agpl-3.0
---

# 🔍 Tracera: PCB Hata Tespiti ve Açıklanabilir Yapay Zeka (XAI)

Baskı devre kartlarında (PCB) mikron düzeyindeki 6 sınıf kusuru tespit eder ve **LayerCAM (Katman 4)** ile modelin karar gerekçesini görselleştirir.

## 🚀 Özellikler
- **Hızlı CPU Çıkarımı:** OpenVINO INT8 kuantize model ile CPU üzerinde ~90 ms gecikme.
- **Doğrulanmış XAI Haritaları:** Backbone katman 4 üzerinde LayerCAM ısı haritaları ve $4\\times$ yakınlaştırılmış hata bölgeleri.
- **Dinamik Raporlama:** F1-maksimum çalışma noktası (conf=0.395) ve edge benchmark metrikleri.

## 📌 Model ve Veri Kümesi
Model, 10 kartlık PKU-Market-PCB setinin 6 kartında eğitildi (2 kart val, 2 kart test).
Test setinde şablon kaynaklı tekrarlayan artifactlar hariç tutulduğunda F1 skoru **%91.13** seviyesindedir.
"""
    (output_dir / "README.md").write_text(readme_content, encoding="utf-8")

    # 9. Boyut hesabı
    total_bytes = 0
    file_count = 0
    for root, _, files in os.walk(output_dir):
        for f in files:
            fp = Path(root) / f
            total_bytes += fp.stat().st_size
            file_count += 1

    total_mb = total_bytes / (1024 * 1024)
    print(f"✅ [build_space] Paketleme tamamlandı! Toplam dosya: {file_count}, Toplam boyut: {total_mb:.2f} MB ({total_bytes:,} bayt)")

    return {
        "output_dir": str(output_dir),
        "total_files": file_count,
        "total_bytes": total_bytes,
        "total_mb": round(total_mb, 2),
        "package_versions": pkg_versions,
    }


def run_isolated_test(space_dir: Path) -> None:
    """space/ klasörünü repo dışı geçici bir dizine kopyalayıp uçtan uca test eder."""
    print("\n" + "=" * 70)
    print("🧪 [isolated_test] Repo Dışı İzole Test Başlatılıyor...")
    print("=" * 70)

    with tempfile.TemporaryDirectory(prefix="tracera_space_isolated_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        print(f"📁 [isolated_test] Geçici izole test dizini: {temp_dir}")

        # Paketi geçici dizine kopyala
        shutil.copytree(space_dir, temp_dir, dirs_exist_ok=True)

        test_code = """
import os, sys, time, cv2
from pathlib import Path
import numpy as np

# GPU kullanımını engelle
os.environ['CUDA_VISIBLE_DEVICES'] = ''

BASE_DIR = Path('.').resolve()
print(f'[TestRunner] Çalışma dizini: {BASE_DIR}')
assert (BASE_DIR / 'weights' / 'last.pt').is_file(), 'last.pt eksik!'
assert (BASE_DIR / 'weights' / 'last_int8_openvino_model').is_dir(), 'OpenVINO modeli eksik!'
assert (BASE_DIR / 'tracera_xai' / 'gradcam.py').is_file(), 'gradcam.py eksik!'

# app modülünü doğrudan izole dizinden import et
import app

print('=== 1. Gradio Uygulama Yapılandırma Testi ===')
demo = app.build_app()
print('Gradio demo başarıyla inşa edildi.')

print('=== 2. CPU Tespit Testi (OpenVINO INT8) ===')
sample_img_path = BASE_DIR / 'examples' / 'val_05_missing_hole_01.jpg'
assert sample_img_path.is_file(), f'Örnek görsel bulunamadı: {sample_img_path}'

img_bgr = cv2.imread(str(sample_img_path))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Isınma / ilk çağrı
ann_img, df_res, choices, state, st_str = app.run_pcb_detection(img_rgb, conf_threshold=0.395)
# Ölçüm çağrısı
t0 = time.perf_counter()
ann_img, df_res, choices, state, st_str = app.run_pcb_detection(img_rgb, conf_threshold=0.395)
det_ms = (time.perf_counter() - t0) * 1000.0

print(f'Sıcak tespit süresi: {det_ms:.1f} ms')
print(f'Durum: {st_str}')
print(f'Tespit sayısı: {len(df_res)}')
assert len(df_res) > 0, 'Hiçbir tespit bulunamadı!'
assert len(choices) > 0, 'Dropdown seçenekleri boş!'

print('=== 3. XAI LayerCAM Testi (Katman 4 PyTorch CPU) ===')
target_choice = choices[0]
t1 = time.perf_counter()
overlay, zoom, cam_status = app.run_explainability(state, target_choice)
cam_ms = (time.perf_counter() - t1) * 1000.0

print(f'LayerCAM üretim süresi: {cam_ms:.1f} ms')
print(f'CAM Durumu: {cam_status}')
assert overlay is not None, 'CAM overlay görseli üretilemedi!'
assert zoom is not None, 'CAM zoom görseli üretilemedi!'
assert isinstance(overlay, np.ndarray), 'Overlay numpy dizisi değil!'
assert isinstance(zoom, np.ndarray), 'Zoom numpy dizisi değil!'

print(f'Overlay boyutu: {overlay.shape}, Zoom boyutu: {zoom.shape}')
print('🎉 İZOLE TEST TÜM AŞAMALARDAN BAŞARIYLA GEÇTİ!')
"""
        test_script_path = temp_dir / "run_test.py"
        test_script_path.write_text(test_code, encoding="utf-8")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["PYTHONPATH"] = str(temp_dir)

        # Python çalıştırıcı
        py_exe = sys.executable

        res = subprocess.run(
            [py_exe, str(test_script_path)],
            cwd=str(temp_dir),
            env=env,
            capture_output=True,
            text=True,
        )

        print(res.stdout)
        if res.returncode != 0:
            print("STDERR:")
            print(res.stderr)
            raise RuntimeError(f"İzole test başarısız oldu! (çıkış kodu: {res.returncode})")

    print("=" * 70)
    print("✅ [isolated_test] İzole test temiz bir şekilde tamamlandı.")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hugging Face Spaces için paket inşa et ve test et.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "space",
        help="Hedef paket dizini (varsayılan: space/)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=True,
        help="Paketleme sonrası geçici dizinde izole test çalıştır (varsayılan: True)",
    )
    parser.add_argument(
        "--no-test",
        action="store_false",
        dest="test",
        help="İzole testi atla",
    )

    args = parser.parse_args()
    summary = build_space(args.output_dir)

    print("\n📋 Paketlenen Kütüphane Sürümleri:")
    for pkg, ver in summary["package_versions"].items():
        print(f"  - {pkg}: {ver}")

    print(f"\n📦 Toplam Paket Boyutu: {summary['total_mb']} MB")

    if args.test:
        run_isolated_test(args.output_dir)


if __name__ == "__main__":
    main()
