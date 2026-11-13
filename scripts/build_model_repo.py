"""Tracera Hugging Face Hub model deposu (model_repo/) hazirlayici.

Kullanim:
    python scripts/build_model_repo.py

Not: Tum metrikler results/*.csv dosyalarindan dinamik olarak okunur; elle sayi yazilmaz.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_WEIGHTS_DIR = REPO_ROOT / "runs" / "detect" / "g_y11s_1024_full" / "weights"
DEFAULT_ONNX_PATH = REPO_ROOT / "web" / "public" / "models" / "tracera_1024.onnx"
DEFAULT_META_PATH = REPO_ROOT / "web" / "public" / "models" / "meta.json"
DEFAULT_OUT_DIR = REPO_ROOT / "model_repo"

CLASSES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]

FORBIDDEN_TERMS = ("mikron", "micron", "telefonda", "gerçek zamanlı", "gercek zamanli")


def _dir_size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return total / (1024 * 1024)


def _fmt_mean_std(series: pd.Series, decimals: int = 4) -> str:
    m = float(series.mean())
    s = float(series.std(ddof=1)) if len(series) > 1 else float("nan")
    if np.isnan(s):
        return f"{m:.{decimals}f}"
    return f"{m:.{decimals}f} ± {s:.{decimals}f}"


def build_model_card(
    out_dir: Path,
    meta: dict,
    repo_id: str,
) -> str:
    cv_res = pd.read_csv(RESULTS_DIR / "cv_resolution.csv")
    cv_leak = pd.read_csv(RESULTS_DIR / "cv_leakage.csv")
    exp_df = pd.read_csv(RESULTS_DIR / "experiments.csv")
    final_val = pd.read_csv(RESULTS_DIR / "final_val.csv")
    final_val_last = final_val[final_val["ağırlık"] == "last.pt"].iloc[0]
    final_test = pd.read_csv(RESULTS_DIR / "final_test.csv").iloc[0]
    cv_total_imgs = int(cv_leak[cv_leak["şema"] == "grouped"].iloc[0]["n_train"] + cv_leak[cv_leak["şema"] == "grouped"].iloc[0]["n_val"])
    op_df = pd.read_csv(RESULTS_DIR / "operating_point.csv")
    tpl_df = pd.read_csv(RESULTS_DIR / "template_check.csv")
    edge_df = pd.read_csv(RESULTS_DIR / "edge_benchmark.csv")
    quant_df = pd.read_csv(RESULTS_DIR / "quant_parity.csv")
    web_par = pd.read_csv(RESULTS_DIR / "web_parity.csv")
    gcam_df = pd.read_csv(RESULTS_DIR / "gradcam_sanity.csv")
    aoi_df = pd.read_csv(RESULTS_DIR / "aoi_transfer.csv")

    # CV resolution 1024 & 640 (last.pt)
    cv_1024 = cv_res[(cv_res["imgsz"] == 1024) & (cv_res["ağırlık"] == "last")].sort_values("kat")
    cv_640 = cv_res[(cv_res["imgsz"] == 640) & (cv_res["ağırlık"] == "last")].sort_values("kat")
    cv_1024_folds = ", ".join(
        f"Kat {int(r['kat'])}={r['mAP50']:.4f}" for _, r in cv_1024.iterrows()
    )

    # Leakage comparison (640, 5 folds)
    leak_rnd = cv_leak[cv_leak["şema"] == "random"]
    leak_grp = cv_leak[cv_leak["şema"] == "grouped"]
    leak_diff = float(leak_grp["mAP50"].mean() - leak_rnd["mAP50"].mean())

    # Experiment row (g_y11s_1024_full)
    exp_row = exp_df[exp_df["deney"] == "g_y11s_1024_full"].iloc[0]

    # Operating point rows
    op_val = op_df[op_df["split"] == "val"].iloc[0]
    op_test = op_df[op_df["split"] == "test"].iloc[0]
    op_sens = op_df[op_df["split"] == "test_sablon_noktalari_haric"].iloc[0]

    # Clean template row (Kart 06)
    tpl_06 = tpl_df[tpl_df["kart_no"].astype(str).str.zfill(2) == "06"].iloc[0]

    # Edge benchmarks @ 1024
    e1024 = edge_df[edge_df["imgsz"] == 1024]

    # Quant parity
    q025 = quant_df[quant_df["conf_esik"] == 0.25].iloc[0]
    q395 = quant_df[quant_df["conf_esik"] == 0.395].iloc[0]

    # Web parity SUMMARY & median latencies
    wp_rows = web_par[web_par["image"] != "SUMMARY"]
    wp_sum = web_par[web_par["image"] == "SUMMARY"].iloc[0]
    wp_med_model = pd.to_numeric(wp_rows["model_ms"]).median()
    wp_med_total = pd.to_numeric(wp_rows["total_ms"]).median()

    # LayerCAM @ Layer 4 (g_y11s_1024_full)
    lc_row = gcam_df[
        (gcam_df["model"] == "g_y11s_1024_full")
        & (gcam_df["katman"] == 4)
        & (gcam_df["method"] == "layercam")
    ].iloc[0]

    # AoI transfer test summary (3 seeds)
    aoi_test = aoi_df[aoi_df["split"] == "test"]
    aoi_coco = aoi_test[aoi_test["baslangic"] == "coco"]["mAP50"]
    aoi_pku = aoi_test[aoi_test["baslangic"] == "pku"]["mAP50"]

    # Export artifact sizes inside out_dir
    pt_mb = _dir_size_mb(out_dir / "last.pt")
    ov_mb = _dir_size_mb(out_dir / "last_int8_openvino_model")
    onnx_mb = _dir_size_mb(out_dir / "tracera_1024.onnx")
    classes_str = ", ".join(f"`{c}`" for c in meta.get("classes", CLASSES))

    lines = [
        "---",
        "language:",
        "  - tr",
        "  - en",
        "license: agpl-3.0",
        "library_name: ultralytics",
        "pipeline_tag: object-detection",
        "tags:",
        "  - pcb",
        "  - defect-detection",
        "  - yolo11",
        "  - openvino",
        "  - onnx",
        "  - explainable-ai",
        "  - layercam",
        "---",
        "",
        "# Tracera — YOLO11s PCB Kusur Tespiti Model Paketi / PCB Defect Detection Model Weights",
        "",
        "Bu model deposu, **Tracera** projesinin kart bazlı bölünmüş PKU-Market-PCB veri kümesi üzerinde "
        "eğitilmiş önceden kilitli (`last.pt`) YOLO11s @ 1024 ağırlıklarını ve çıkarım dışa aktarımlarını "
        "(OpenVINO INT8 + FP32 ONNX) içerir. Açıklanabilirlik arayüzünde tespit kararları LayerCAM (Katman 4) "
        "ile görselleştirilir.",
        "",
        "*English summary:* This model repository hosts the **Tracera** YOLO11s (`imgsz=1024`, pre-locked `last.pt`) "
        "weights trained on board-grouped PKU-Market-PCB data for 6 PCB surface defect classes, packaged alongside "
        "OpenVINO INT8 (CPU) and raw FP32 ONNX (`opset=17`, client-side NMS) exports.",
        "",
        "---",
        "",
        "## 1. Depo İçeriği ve Dosyalar / Artifacts",
        "",
        "| Dosya / Klasör | Format | Boyut (MB) | Kullanım Alanı |",
        "|---|---|---|---|",
        f"| `last.pt` | PyTorch Checkpoint | {pt_mb:.2f} MB | Eğitim sonu kilitli ağırlık, PyTorch GPU çıkarımı ve LayerCAM @ Katman 4 gradyan hesabı |",
        f"| `last_int8_openvino_model/` | OpenVINO IR INT8 (`last.xml`, `last.bin`, `metadata.yaml`) | {ov_mb:.2f} MB | Yerel FastAPI / Tauri masaüstü CPU çıkarımı |",
        f"| `tracera_1024.onnx` | ONNX FP32 (`opset=17`, ham çıktı `[1, 10, 21504]`) | {onnx_mb:.2f} MB | Tarayıcı içi ve Android WebView ONNX Runtime Web çıkarımı |",
        f"| `meta.json` | JSON (`version={meta.get('version')}`) | <0.01 MB | Ön/son işleme sabitleri (`imgsz={meta.get('imgsz')}`, `conf={meta.get('conf')}`, `iou_nms={meta.get('iou_nms')}`, `letterbox_pad={meta.get('letterbox_pad')}`) |",
        "",
        "---",
        "",
        "## 2. Görev, Veri Kümesi ve Eğitim Ayarları / Task, Data & Training",
        "",
        f"- **Görev / Task:** 6 sınıflı PCB yüzey kusuru nesne tespiti ({classes_str}).",
        "- **Veri Kümesi & Kart Bazlı Bölme / Dataset & Board-Grouped Split:** "
        f"PKU-Market-PCB veri kümesi 10 farklı kart tasarımından (CV havuzu = {cv_total_imgs} görüntü) oluşur. "
        "Rastgele bölmede aynı kart arka planları eğitim ve test arasında sızıntı ürettiğinden katı kart-bazlı bölme uygulanmıştır: "
        f"**Val = Kart 05 + 11** ({int(final_val_last['görüntü_sayısı'])} görüntü, {int(final_val_last['kutu_sayısı'])} kutu), "
        f"**Tek Seferlik Kilitli Test = Kart 06 + 10** ({int(final_test['görüntü_sayısı'])} görüntü, {int(final_test['kutu_sayısı'])} kusur kutusu). "
        "Final eğitimindeki model test kartlarını hiç görmemiştir.",
        "- **Eğitim Ayarları (`g_y11s_1024_full`):** "
        f"Mimari = `{exp_row['model']}`, `imgsz={int(exp_row['imgsz'])}`, `batch={int(exp_row['batch'])}`, "
        f"`epochs={int(exp_row['epochs'])}`, `patience=0`, seçilen ağırlık = `last.pt` (val tabanlı erken durdurma yok), "
        f"eğitim süresi = {exp_row['eğitim_süresi_dk']:.2f} dk, tepe GPU belleği = {exp_row['gpu_bellek_tepe_gb']:.2f} GB.",
        "",
        "---",
        "",
        "## 3. Doğrulanmış Sonuçlar / Verified Evaluation Results",
        "",
        "> Tüm standart sapmalar örneklem standart sapmasıdır (`ddof=1`). Başlık sonucu 5-katlı kart bazlı CV'dir; "
        "kilitli test yalnız 2 farklı kart tasarımından (Kart 06 + 10) oluşur.",
        "",
        "### 3.1 Başlık Sonucu: Kart Bazlı 5-Katlı Çapraz Doğrulama & Sızıntı Kontrolü",
        "",
        "| Deney | Çözünürlük | Ağırlık | mAP50 (Ort ± std) | mAP50-95 (Ort ± std) | Katlar / Not |",
        "|---|---|---|---|---|---|",
        f"| **Nihai Mimari (Grouped CV)** | 1024 | `last.pt` | **{_fmt_mean_std(cv_1024['mAP50'])}** | **{_fmt_mean_std(cv_1024['mAP50-95'])}** | {cv_1024_folds} |",
        f"| Karşılaştırma (Grouped CV) | 640 | `last.pt` | {_fmt_mean_std(cv_640['mAP50'])} | {_fmt_mean_std(cv_640['mAP50-95'])} | Tüm 5 katta 1024 daha yüksek (+{(cv_1024['mAP50'].values - cv_640['mAP50'].values).mean():.4f} mAP50) |",
        f"| Sızıntılı Rastgele Bölme (640) | 640 | `last.pt` | {_fmt_mean_std(leak_rnd['mAP50'])} | {_fmt_mean_std(leak_rnd['mAP50-95'])} | Kart arka planı sızıntılı üst sınır |",
        f"| Aynı Görüntülerde Grouped CV | 640 | `last.pt` | {_fmt_mean_std(leak_grp['mAP50'])} | {_fmt_mean_std(leak_grp['mAP50-95'])} | Rastgeleden fark: **{leak_diff:.4f}** |",
        "",
        "### 3.2 Tek Seferlik Kilitli Test (`g_y11s_1024_full`, `last.pt`, Kart 06 + 10)",
        "",
        f"- **Eşikten Bağımsız Test mAP:** **mAP50 = {final_test['mAP50']:.4f}**, **mAP50-95 = {final_test['mAP50-95']:.4f}** "
        f"({int(final_test['görüntü_sayısı'])} görüntü, {int(final_test['kutu_sayısı'])} kutu).",
        "- *(Metodolojik not: Ultralytics çıktısındaki test F1-tepe oracle P/R değerleri işletme çalışma noktası olarak raporlanmaz; kilitli P/R/F1 aşağıda `conf = 0.395` üzerindedir.)*",
        "",
        "| Sınıf | Test mAP50 | Test mAP50-95 |",
        "|---|---|---|",
    ]

    for cls in CLASSES:
        lines.append(
            f"| `{cls}` | {final_test[f'{cls}_mAP50']:.4f} | {final_test[f'{cls}_mAP50-95']:.4f} |"
        )

    lines.extend([
        "",
        f"### 3.3 Kilitli Çalışma Noktası (`conf = {op_val['conf_threshold']:.3f}`, `IoU = {op_val['iou_threshold']:.2f}`)",
        "",
        "| Split | TP | FP | FN | Precision | Recall | F1 |",
        "|---|---|---|---|---|---|---|",
        f"| Val (Kart 05 + 11) | {int(op_val['TP'])} | {int(op_val['FP'])} | {int(op_val['FN'])} | {op_val['P']:.4f} | {op_val['R']:.4f} | {op_val['F1']:.4f} |",
        f"| **Test Kilitli (Kart 06 + 10)** | **{int(op_test['TP'])}** | **{int(op_test['FP'])}** | **{int(op_test['FN'])}** | **{op_test['P']:.4f}** | **{op_test['R']:.4f}** | **{op_test['F1']:.4f}** |",
        f"| *Duyarlılık Satırı (Kart 06 2 sabit şablon noktası hariç)* | {int(op_sens['TP'])} | {int(op_sens['FP'])} | {int(op_sens['FN'])} | {op_sens['P']:.4f} | {op_sens['R']:.4f} | {op_sens['F1']:.4f} |",
        "",
        f"- Temiz şablon kontrolünde Kart 06 şablon görüntüsü (`{tpl_06['dosya']}`) `mouse_bite` sınıfında {tpl_06['güven']:.4f} güvenle sabit yanlış alarm üretmektedir.",
        "",
        "### 3.4 Çıkarım Formatları, Nicemleme ve Tarayıcı Paritesi (Laptop Donanımı)",
        "",
        "*Not: Aşağıdaki hız ölçümleri dizüstü bilgisayar donanımında (Ryzen 7 8845HS CPU / RTX 4060 Laptop GPU) alınmıştır.*",
        "",
        "| Format | Donanım | Boyut (MB) | Medyan Model (ms) | Medyan Toplam (ms) | Uçtan Uca FPS | Val mAP50 |",
        "|---|---|---|---|---|---|---|",
    ])

    for _, er in e1024.iterrows():
        lines.append(
            f"| {er['format']} | {er['donanım']} | {er['boyut_mb']:.2f} | "
            f"{er['gecikme_inf_medyan_ms']:.2f} | {er['gecikme_toplam_medyan_ms']:.2f} | "
            f"{er['fps_on_son_islem_dahil']:.1f} | {er['mAP50']:.4f} |"
        )

    lines.extend([
        "",
        f"- **FP32 vs OpenVINO INT8 Nicemleme Paritesi (10 val görüntüsü):** "
        f"`conf=0.25` eşleşen = {int(q025['eslesen'])} (yalnız FP32: {int(q025['yalniz_fp32'])}, yalnız INT8: {int(q025['yalniz_int8'])}, ort IoU = {q025['ort_iou']:.4f}, ort |Δconf| = {q025['ort_dconf']:.4f}); "
        f"`conf=0.395` eşleşen = {int(q395['eslesen'])} (ort IoU = {q395['ort_iou']:.4f}, tüm farklar Kart 05 şablonundaki 2 sabit `spurious_copper` noktasında yoğunlaşmıştır).",
        f"- **Tarayıcı ORT WebGPU (`tracera_1024.onnx`) Paritesi (10 val, `conf=0.25`):** "
        f"İki yönlü eşleşme = **%{wp_sum['two_way_match_pct']:.2f}** ({int(wp_sum['matched_025'])} eşleşen), ort IoU = {wp_sum['mean_iou']:.4f}, ort |Δconf| = {wp_sum['mean_abs_dconf']:.4f}, "
        f"`0.395` sayım farkı = {int(wp_sum['diff_0395'])}; RTX 4060 Laptop WebGPU medyan model/toplam gecikmesi = {wp_med_model:.2f} / {wp_med_total:.2f} ms.",
        f"- **XAI Doğrulama (LayerCAM @ Katman 4, n={int(lc_row['n_tespit'])}):** "
        f"1× silme göreli düşüş = {lc_row['goreli_dusus_harita_ort_1x']:.4f} (kazanma %{lc_row['goreli_kazanma_orani_1x']*100:.1f}), "
        f"2× düşüş = {lc_row['goreli_dusus_harita_ort_2x']:.4f} (kazanma %{lc_row['goreli_kazanma_orani_2x']*100:.1f}), "
        f"pencere içi kutu enerjisi = %{lc_row['enerji_kutuda_pencere']*100:.1f}.",
        f"- **Endüstriyel PCB-AoI Temsil Transferi (3 Seed, null sonuç):** "
        f"PKU başlangıcı test mAP50 = {_fmt_mean_std(aoi_pku)}, COCO başlangıcı = {_fmt_mean_std(aoi_coco)}; "
        "fark seed varyansının içinde kalmıştır.",
        "",
        "---",
        "",
        "## 4. Kullanım / Quick Usage",
        "",
        "### Python (Ultralytics / PyTorch & OpenVINO INT8)",
        "",
        "```python",
        "from huggingface_hub import snapshot_download",
        "from ultralytics import YOLO",
        "",
        f'weights_dir = snapshot_download(repo_id="{repo_id}")',
        "",
        "# PyTorch (GPU/CPU)",
        'pt_model = YOLO(f"{weights_dir}/last.pt")',
        'pt_res = pt_model.predict("pcb.jpg", imgsz=1024, conf=0.395, iou=0.7)',
        "",
        "# OpenVINO INT8 (CPU)",
        'ov_model = YOLO(f"{weights_dir}/last_int8_openvino_model", task="detect")',
        'ov_res = ov_model.predict("pcb.jpg", imgsz=1024, conf=0.395, iou=0.7)',
        "```",
        "",
        "### Tarayıcı / Browser (ONNX Runtime Web + `tracera_1024.onnx`)",
        "",
        "- Girdi tensörü: `[1, 3, 1024, 1024]` RGB `float32` (`[0, 1]` normalize), en-boy oranını koruyan letterbox + sabit `114` (`114/255`) dolgu.",
        "- Çıktı tensörü: `[1, 10, 21504]` (`[cx, cy, w, h, 6 sınıf skoru]`); dahili NMS yoktur, istemci tarafında `conf >= 0.395` ve `iou = 0.7` NMS uygulanır (`meta.json`).",
        "",
        "### Ortam Değişkeniyle Otomatik İndirme (Tracera Repo)",
        "",
        "```bash",
        f"export TRACERA_MODEL_REPO={repo_id}",
        "pytest -v",
        "```",
        "",
        "---",
        "",
        "## 5. Sınırlılıklar / Known Limitations",
        "",
        "1. **Küçük Kart Çeşitliliği:** PKU-Market-PCB yalnızca 10 kart tasarımı içerir; kilitli test kümesi yalnız 2 kart tasarımına (Kart 06 ve 10) dayanır.",
        "2. **Çözünürlük Seçimi Bağımsızlığı:** `1024` vs `640` 5-katlı CV deneyi tüm 10 kartlık havuz üzerinde yürütüldüğünden çözünürlük kararı test kartlarından tam bağımsız değildir (nihai ağırlık test kartlarını görmemiştir).",
        "3. **Şablon Yanlış Alarmları:** Kart 06 temiz şablonunda `mouse_bite` yanlış alarmı sabit tekrarlanır; FP32–INT8 niceleme farkları da `conf >= 0.395` düzeyinde Kart 05 şablonundaki 2 sabit `spurious_copper` konumunda yoğunlaşır.",
        "4. **Çapraz Platform Farkları:** OpenVINO INT8 Windows vs Linux çalışmaları arasında eşik sınırındaki kutularda küçük (~0.01) güven farkı görülebilir.",
        "5. **Alan Boşluğu & Kamera:** Model yukarıdan çekilmiş yüksek çözünürlüklü sentetik kusurlu PKU görüntüleriyle eğitilmiştir. Gerçek AoI verisine transfer deneyi null sonuç vermiştir; el kamerasında açı/ışık değişimiyle doğruluk düşer.",
        "6. **Donanım Kapsamı:** Hız değerleri dizüstü donanımında (RTX 4060 Laptop / Ryzen 7 8845HS) ölçülmüştür; gömülü kart veya mobil tarayıcı hız garantisi verilmez.",
        "",
        "---",
        "",
        "## 6. Lisans ve Atıf / License & Citation",
        "",
        "- **Lisans / License:** GNU Affero General Public License v3.0 (`AGPL-3.0`).",
        "- **Veri Kümesi Atfı / Dataset Citation:**",
        "",
        "```bibtex",
        "@article{huang2019pcb,",
        '  title={A PCB Dataset for Defects Detection and Classification},',
        "  author={Huang, Weibo and Wei, Peng},",
        "  journal={arXiv preprint arXiv:1901.08204},",
        "  year={2019}",
        "}",
        "```",
        "",
    ])

    content = "\n".join(lines)
    lowered = content.lower()
    for term in FORBIDDEN_TERMS:
        if term in lowered:
            raise ValueError(f"Model kartinda yasakli ifade tespit edildi: {term!r}")
    return content


def build_model_repo(
    out_dir: Path = DEFAULT_OUT_DIR,
    weights_dir: Path = DEFAULT_WEIGHTS_DIR,
    onnx_path: Path = DEFAULT_ONNX_PATH,
    meta_path: Path = DEFAULT_META_PATH,
    repo_id: str = "ebboo/tracera-model",
) -> Path:
    pt_src = weights_dir / "last.pt"
    ov_src = weights_dir / "last_int8_openvino_model"

    for req in (pt_src, ov_src, onnx_path, meta_path):
        if not req.exists():
            raise FileNotFoundError(f"Gerekli kaynak dosya/dizin bulunamadi: {req}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(pt_src, out_dir / "last.pt")
    shutil.copytree(ov_src, out_dir / "last_int8_openvino_model")
    shutil.copy2(onnx_path, out_dir / "tracera_1024.onnx")
    shutil.copy2(meta_path, out_dir / "meta.json")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    card = build_model_card(out_dir=out_dir, meta=meta, repo_id=repo_id)
    (out_dir / "README.md").write_text(card, encoding="utf-8", newline="\n")
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Build HF Hub model_repo/ package")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--repo-id", default="ebboo/tracera-model")
    args = parser.parse_args()

    out_dir = build_model_repo(out_dir=args.out, repo_id=args.repo_id)

    print("MODEL REPO HAZIR")
    for p in sorted(out_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(out_dir).as_posix()
            mb = p.stat().st_size / (1024 * 1024)
            print(f"  - {rel:45s} {mb:6.2f} MB")
    print("\nYukleme komutu (manuel):")
    print(f"  huggingface-cli upload {args.repo_id} {out_dir.name}/ . --repo-type model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
