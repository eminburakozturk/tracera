"""README son hali taslagi uretici: docs/README_final_draft.md.

Mevcut README.md dosyasindaki Claude denetimli metni ve bolum numaralarini KORUR;
ust baglantilari, platform parite notunu, "Her yerde calisir: web, masaustu, Android"
bolumunu (CSV'lerden ve KARARLAR.md'den sayisal verilerle) ve web/Tauri calistirma
komutlarini ekler. README.md dosyasina DOKUNMAZ.
"""

from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
README_PATH = REPO_ROOT / "README.md"
KARARLAR_PATH = REPO_ROOT / "KARARLAR.md"
RESULTS_DIR = REPO_ROOT / "results"
WEB_DIR = REPO_ROOT / "web"
OUT_PATH = REPO_ROOT / "docs" / "README_final_draft.md"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Gerekli CSV bulunamadi: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return [{k.strip(): (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f)]


def _num(val: Any) -> float | None:
    try:
        v = float(str(val).strip())
        return v
    except (TypeError, ValueError):
        return None


def _measure_precache_mb() -> str:
    sw_path = WEB_DIR / "web_space" / "sw.js"
    if not sw_path.is_file():
        sw_path = WEB_DIR / "dist" / "sw.js"
    if not sw_path.is_file():
        return "olculmedi"
    base_dir = sw_path.parent
    sw_text = sw_path.read_text(encoding="utf-8")
    urls = {m.group(1).lstrip("/") for m in re.finditer(r'url:\s*"([^"]+)"', sw_text)}
    total_bytes = 0
    for rel in urls:
        target = base_dir / rel
        if target.is_file():
            total_bytes += target.stat().st_size
    return f"{total_bytes / (1024 * 1024):.2f} MB"


def _extract_desktop_metrics() -> dict[str, str]:
    """KARARLAR.md #65 ve #67'deki olculmus masaustu / APK paket metriklerini cikarir."""
    text = KARARLAR_PATH.read_text(encoding="utf-8")
    m65 = re.search(
        r"NSIS kurulum (\d+(?:\.\d+)? MiB), kurulu (\d+(?:\.\d+)? GiB), "
        r"pencere (\d+(?:\.\d+)? s), yerel motor hazır (~?\d+(?:\.\d+)? s) "
        r"\(soğuk ilk açılış (\d+(?:\.\d+)? s)\), /detect (\d+(?:\.\d+)? ms), "
        r"/explain (\d+(?:\.\d+)? s)",
        text,
    )
    if not m65:
        raise ValueError("KARARLAR.md #65 masaustu metrikleri bulunamadi")
    m67 = re.search(r"Android APK kabul:\s*(\d+(?:\.\d+)? MiB)", text)
    if not m67:
        raise ValueError("KARARLAR.md #67 APK boyutu bulunamadi")
    return {
        "nsis_mib": m65.group(1),
        "installed_gib": m65.group(2),
        "window_s": m65.group(3),
        "sidecar_ready_s": m65.group(4),
        "sidecar_cold_s": m65.group(5),
        "detect_ms": m65.group(6),
        "explain_s": m65.group(7),
        "apk_mib": m67.group(1),
    }


def _build_everywhere_section() -> str:
    web_parity_rows = _read_csv(RESULTS_DIR / "web_parity.csv")
    parity_images = [r for r in web_parity_rows if r["image"] != "SUMMARY"]
    parity_summary = next((r for r in web_parity_rows if r["image"] == "SUMMARY"), None)
    if parity_summary is None:
        raise ValueError("results/web_parity.csv icinde SUMMARY satiri bulunamadi")

    parity_med_model = statistics.median(
        v for v in (_num(r["model_ms"]) for r in parity_images) if v is not None
    )
    parity_med_total = statistics.median(
        v for v in (_num(r["total_ms"]) for r in parity_images) if v is not None
    )

    spike_rows = _read_csv(RESULTS_DIR / "web_spike.csv")
    wgpu_fp32 = next(r for r in spike_rows if r["backend"] == "ort-webgpu" and r["model"] == "FP32 ONNX")
    wgpu_int8 = next(r for r in spike_rows if r["backend"] == "ort-webgpu" and r["model"] == "INT8 Dynamic ONNX")
    wasm_fp32 = next(r for r in spike_rows if r["backend"] == "ort-wasm" and r["model"] == "FP32 ONNX")
    wasm_int8 = next(r for r in spike_rows if r["backend"] == "ort-wasm" and r["model"] == "INT8 Dynamic ONNX")

    video_rows = _read_csv(RESULTS_DIR / "web_video_fps.csv")
    vid_wgpu = next(r for r in video_rows if r["backend"] == "webgpu")
    vid_wasm = next(r for r in video_rows if r["backend"] == "wasm")

    quant_rows = _read_csv(RESULTS_DIR / "quant_parity.csv")
    qp025 = next(r for r in quant_rows if _num(r["conf_esik"]) == 0.25)
    qp395 = next(r for r in quant_rows if _num(r["conf_esik"]) == 0.395)

    edge_rows = _read_csv(RESULTS_DIR / "edge_benchmark.csv")
    gpu_hw = next(r["donanım"] for r in edge_rows if "GPU" in r["format"])
    cpu_hw = next(r["donanım"] for r in edge_rows if "CPU" in r["format"])

    analysis_json = json.loads((WEB_DIR / "public" / "analysis.json").read_text(encoding="utf-8"))
    b05_clean = [r for r in analysis_json["template_clean_rows"] if r["kart_no"] == "05"]
    b05_desc = " ve ".join(f"`conf={r['conf']}` @ ({r['cx']}, {r['cy']})" for r in b05_clean)

    onnx_bytes = (WEB_DIR / "public" / "models" / "tracera_1024.onnx").stat().st_size
    wasm_candidates = [
        WEB_DIR / "web_space" / "ort" / "ort-wasm-simd-threaded.jsep.wasm",
        WEB_DIR / "dist" / "ort" / "ort-wasm-simd-threaded.jsep.wasm",
        WEB_DIR / "node_modules" / "onnxruntime-web" / "dist" / "ort-wasm-simd-threaded.jsep.wasm",
    ]
    wasm_path = next((p for p in wasm_candidates if p.is_file()), None)
    if wasm_path is None:
        raise FileNotFoundError("ort-wasm-simd-threaded.jsep.wasm bulunamadi")
    onnx_mb = onnx_bytes / (1024 * 1024)
    wasm_mb = wasm_path.stat().st_size / (1024 * 1024)
    first_detect_mb = round(onnx_mb + wasm_mb)
    precache_mb = _measure_precache_mb()

    desk = _extract_desktop_metrics()
    phone_rows = _read_csv(RESULTS_DIR / "device_bench_phone.csv")
    phone = phone_rows[0]

    lines = [
        "## Her yerde çalışır: web, masaüstü, Android",
        "",
        "### Mimari",
        "",
        "Tracera, tek bir React + TypeScript (`web/`) arayüzünü üç çalıştırma ortamına ve dört dağıtım hedefine taşır:",
        "",
        "```mermaid",
        "flowchart LR",
        '    UI["web/ (Tek React + TypeScript Arayüzü)"]',
        '    UI -->|"Tarayıcı / PWA / Android WebView"| ORT["ORT-Web (FP32 ONNX: WebGPU / tek iş parçacıklı WASM)"]',
        '    UI -->|"Tauri 2 Masaüstü (Windows EXE)"| API["FastAPI Sidecar (OpenVINO INT8 + LayerCAM Katman 4)"]',
        '    UI -->|"Bulut Modu / Mobil Isı Haritası"| HF["Gradio Space (ebboo/tracera)"]',
        "```",
        "",
        "- **Statik Web / Hugging Face Static Space (`web_space/`):** Sunucu GPU'su gerektirmeden doğrudan kullanıcı tarayıcısında ONNX Runtime Web (WebGPU öncelikli, yoksa tek iş parçacıklı WASM) ile `1024×1024` FP32 ONNX çıkarımı yapar.",
        "- **PWA (Progressive Web App):** Uygulama kabuğunu Service Worker ile önbelleğe alır; ONNX modeli ve ORT çalışma zamanını yalnızca ilk tespitte indirerek sonraki açılışlarda çevrimdışı tespit sunar.",
        f"- **Tauri 2 Masaüstü (`desktop/`, Windows EXE):** Aynı arayüzü `isTauri()` köprüsüyle (`get_sidecar_status` / `get_api_base_url` / `get_platform` / `tracera://api-ready`) paketlenmiş Python FastAPI yan sunucusuna bağlar; varsayılan motor **Yerel sunucu**dur (OpenVINO INT8 tespit + LayerCAM Katman 4).",
        f"- **Android APK (Tauri 2 WebView, `{desk['apk_mib']}` debug imzalı):** `get_sidecar_status.available = false` ve `get_platform = \"android\"` bildirildiğinden yerel motor ekranı gösterilmez; tespit doğrudan **ORT-Web** üzerinde çalışır ve **Yerel sunucu** gizlenir, ısı haritası yalnız Bulut'a (`ebboo/tracera`) yönlenir.",
        "",
        "### Tarayıcı Paritesi (`OrtDetector` WebGPU vs. PyTorch FP32 Referans)",
        "",
        f"{len(parity_images)} val görüntüsü üzerinde (`results/web_parity.csv`, `conf = 0.25` iki yönlü IoU ≥ 0.5 eşleşmesi ve `conf = 0.395` sayım kıyası):",
        "",
        "| Kıyas | Eşleşen (≥0.25) | Yalnız ORT | Yalnız PyTorch | İki Yönlü Eşleşme | Ort. IoU | Ort. \\|Δconf\\| | Sayım Farkı (0.395) | Ort. Model (medyan) | Uçtan Uca (medyan) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {parity_summary['backend']} vs. PyTorch FP32 | {parity_summary['matched_025']} | {parity_summary['only_ort_025']} | {parity_summary['only_pt_025']} | %{parity_summary['two_way_match_pct']} | {float(parity_summary['mean_iou']):.4f} | {float(parity_summary['mean_abs_dconf']):.4f} | {parity_summary['pt_count_0395']} → {parity_summary['ort_count_0395']} ({parity_summary['diff_0395']}) | {parity_med_model:.1f} ms | {parity_med_total:.1f} ms |",
        "",
        f"> Ölçüm ortamı: Windows masaüstü Chrome, **{gpu_hw}** (ONNX Runtime Web `ort-webgpu`, FP32 ONNX). `0.395` çalışma noktasındaki 1 kutuluk fark eşik sınırındaki bir tahminden kaynaklanır.",
        "",
        "### Tarayıcı Hızları (Statik Görüntü & Canlı Video)",
        "",
        f"Aşağıdaki süreler dizüstü bilgisayar donanımında (**{gpu_hw}** / **{cpu_hw}**, Windows Desktop Chrome) ölçülmüştür:",
        "",
        "| Senaryo / Kaynak | Motor | Model | Medyan Model Süresi | Canlı Video FPS | Notlar |",
        "|---|---|---|---:|---:|---|",
        f"| Tekil görüntü, 10 val (`web_spike.csv`) | {wgpu_fp32['backend']} | {wgpu_fp32['model']} ({wgpu_fp32['boyut_mb']} MB) | {wgpu_fp32['medyan_ms']} ms | — | Ort. IoU {wgpu_fp32['ort_iou']}, \\|Δconf\\| {wgpu_fp32['ort_dconf']} |",
        f"| Tekil görüntü, 10 val (`web_spike.csv`) | {wgpu_int8['backend']} | {wgpu_int8['model']} ({wgpu_int8['boyut_mb']} MB) | zaman aşımı | — | Kullanımdan çıkarıldı |",
        f"| Tekil görüntü, 10 val (`web_spike.csv`) | {wasm_fp32['backend']} | {wasm_fp32['model']} ({wasm_fp32['boyut_mb']} MB) | {wasm_fp32['medyan_ms']} ms | — | WebGPU yokken yedek; tek iş parçacıklı WASM |",
        f"| Tekil görüntü, 10 val (`web_spike.csv`) | {wasm_int8['backend']} | {wasm_int8['model']} ({wasm_int8['boyut_mb']} MB) | {wasm_int8['medyan_ms']} ms | — | Eşleşme {wasm_int8['eşleşme_oranı']}; doğruluk kaybı nedeniyle elendi |",
        f"| Canlı video 5.2 s, `val_pan.mp4` (`web_video_fps.csv`) | {vid_wgpu['ort_ep']} | FP32 ONNX | {vid_wgpu['model_ms']} ms | {vid_wgpu['avg_fps']} FPS ({vid_wgpu['frames']} kare) | Kuyruk yok (meşgulken kare atlanır) |",
        f"| Canlı video 5.2 s, `val_pan.mp4` (`web_video_fps.csv`) | {vid_wasm['ort_ep']} | FP32 ONNX | {vid_wasm['model_ms']} ms | {vid_wasm['avg_fps']} FPS ({vid_wasm['frames']} kare) | WASM üzerinde akıcı video beklenmez |",
        "",
        "### Masaüstü Uygulaması (Windows Tauri 2 EXE + FastAPI Sidecar)",
        "",
        f"Windows kurulum ve ilk çalıştırma ölçümleri (`KARARLAR.md` #65, **{cpu_hw}**):",
        "",
        "| Ölçüm | Değer | Not |",
        "|---|---|---|",
        f"| NSIS kurulum dosyası | **{desk['nsis_mib']}** | İmzasız derleme (ilk açılışta Windows SmartScreen uyarısı gösterebilir) |",
        f"| Diskteki kurulu boyut | **{desk['installed_gib']}** | Web arayüzü + Python FastAPI CPU sidecar + OpenVINO INT8 / PyTorch ağırlıkları |",
        f"| Pencere açılış süresi | **{desk['window_s']}** | Tauri WebView2 penceresi |",
        f"| Yerel motor hazır süresi | **{desk['sidecar_ready_s']}** (soğuk ilk açılış **{desk['sidecar_cold_s']}**) | Rastgele yerel port üzerinde OpenVINO INT8 + LayerCAM yüklemesi |",
        f"| Tekil tespit (`/detect`) | **{desk['detect_ms']}** | OpenVINO INT8 CPU çıkarımı |",
        f"| Isı haritası (`/explain`) | **{desk['explain_s']}** | PyTorch LayerCAM @ Katman 4 gradyan hesabı |",
        "",
        "### Android Ölçümü (`results/device_bench_phone.csv` — Tek Cihaz)",
        "",
        "| Cihaz / Ortam | Motor & GPU Adaptör | Medyan Model | Medyan Toplam | Koşu Sayısı | Kaynak Notu |",
        "|---|---|---:|---:|---:|---|",
        f"| `{phone['cihaz']}` ({phone['os']}, {phone['webview']}) | `{phone['backend']}` (`{phone['gpu_adaptor']}`) | **{phone['medyan_model_ms']} ms** | **{phone['medyan_toplam_ms']} ms** | {phone['kosu_n']} (ısınma: {phone['isinma']}) | {phone['kaynak']} |",
        "",
        f"> **Tek Cihaz Notu:** Yukarıdaki mobil ölçüm yalnızca tek bir üst segment Android cihazda (`{phone['cihaz']}`, WebGPU aktif) alınmıştır; farklı telefon modellerine—özellikle WebGPU desteği bulunmayan cihazlarda WASM yedek yoluna düşen telefonlara—genellenemez. Telefon kullanıcıları kendi cihaz hızlarını **Sonuçlar → Bu cihazda hız testi** düğmesiyle ölçebilir.",
        "",
        "### FP32 / INT8 Farkı ve Şablon Noktası Notu",
        "",
        "Tarayıcı (`FP32 ONNX`) ile yerel masaüstü yan sunucusu (`OpenVINO INT8`) sınır değerdeki kutularda küçük farklar gösterebilir (`results/quant_parity.csv`, 10 val görüntüsü):",
        "",
        "| Güven Eşiği | Eşleşen | Yalnız FP32 | Yalnız INT8 | Ort. IoU | Ort. \\|Δconf\\| |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {qp025['conf_esik']} | {qp025['eslesen']} | {qp025['yalniz_fp32']} | {qp025['yalniz_int8']} | {float(qp025['ort_iou']):.4f} | {float(qp025['ort_dconf']):.4f} |",
        f"| {qp395['conf_esik']} | {qp395['eslesen']} | {qp395['yalniz_fp32']} | {qp395['yalniz_int8']} | {float(qp395['ort_iou']):.4f} | {float(qp395['ort_dconf']):.4f} |",
        "",
        f"`conf ≥ 0.395` noktasındaki tüm FP32–INT8 farkları rastgele dağılmamıştır; Kart 05 şablonunda temiz kartta da yanlış alarm üreten 2 sabit `spurious_copper` arka plan konumunda ({b05_desc}) yoğunlaşmıştır.",
        "",
        "### Çevrimdışı Kullanım, PWA ve İlk İndirme Boyutu",
        "",
        f"- **İlk açılış precache toplamı:** yalnızca uygulama kabuğu (**{precache_mb}**). Model açılışta kendiliğinden indirilmez.",
        f"- **İlk tespitte çalışma zamanı indirmesi:** FP32 ONNX (**{onnx_mb:.2f} MB**) + WebGPU/tek iş parçacıklı WASM JSEP çalışma zamanı (**{wasm_mb:.2f} MB**) olmak üzere yaklaşık **~{first_detect_mb} MB** CacheFirst stratejisiyle indirilir ve önbelleğe alınır. Kullanılmayan `tracera_1024_int8.onnx` üretim paketine (`dist/` ve `web_space/`) dahil edilmez.",
        "- **Çevrimdışı devam:** İlk tespit sonrasında internet kesilse bile tarayıcı içi tespit (\"Cihazda\") çevrimdışı çalışmaya devam eder.",
        "",
        "### Bilinen Sınırlılıklar (Tarayıcı & Mobil)",
        "",
        f"- **WASM canlı video performansı:** WebGPU desteği olmayan cihazlarda ORT WASM ile canlı video çıkarımı `< 1 FPS` düzeyindedir ({vid_wasm['avg_fps']} FPS @ {cpu_hw}); akıcı video akışı yerine tekil fotoğraf tespiti önerilir.",
        "- **Kamera modu deneyseldir:** Model PKU-Market-PCB veri setindeki yukarıdan çekilmiş yüksek çözünürlüklü kart fotoğraflarıyla eğitilmiştir; el kamerası perspektifi, parlamalar veya odak bozukluğunda doğruluk düşebilir.",
        "- **LayerCAM ısı haritası tarayıcıda hesaplanmaz:** Katman 4 aktivasyon × pozitif gradyan geri yayılımı gerektirdiği için XAI açıklaması tarayıcı içinde üretilmez; masaüstünde **Yerel sunucu** (`api/`), ulaşılamazsa veya Android'de **Bulut** (`ebboo/tracera` Space `/explain_api`) ucuna yönlendirilir. Space gizliyken bulut açıklaması kullanılamaz.",
        "",
        "---",
    ]
    return "\n".join(lines)


def _check_forbidden(markdown: str, everywhere_section: str) -> None:
    lower_all = markdown.lower()
    for term in ("mikron", "micron", "gerçek zamanlı", "gercek zamanli"):
        if term in lower_all:
            raise ValueError(f"Yasak ifade tespit edildi: {term!r}")

    # Oracle P/R degerleri sadece Metodolojik Not (uyari) icinde gecmeli, baslik maddelerinde yer almamali
    headline_part = markdown.split("## 2.", 1)[0].split("> **Metodolojik Not:**", 1)[0]
    for oracle_val in ("0.9462", "0.8577"):
        if oracle_val in headline_part:
            raise ValueError(f"Oracle P/R degeri baslik metninde yer alamaz: {oracle_val}")

    # Yeni eklenen bolumde "telefonda X FPS" genellemesi olmamali
    if re.search(r"telefonda.*\bfps\b", everywhere_section.lower()):
        raise ValueError("Yeni bolumde 'telefonda ... FPS' genellemesi tespit edildi")


def build_final_readme_draft() -> str:
    base_lines = README_PATH.read_text(encoding="utf-8").splitlines()

    top_links_block = [
        "[![CI](https://github.com/<OWNER>/<REPO>/actions/workflows/ci.yml/badge.svg)](https://github.com/<OWNER>/<REPO>/actions/workflows/ci.yml) *(CI rozeti yer tutucusu)*",
        "",
        "> - **Canlı Statik Web Demosu (Tarayıcıda ONNX):** `https://huggingface.co/spaces/ebboo/tracera-web` *(yer tutucu — statik web paketi)*",
        "> - **Bulut Demo & LayerCAM API (Hugging Face Spaces, Gradio):** [huggingface.co/spaces/ebboo/tracera](https://huggingface.co/spaces/ebboo/tracera)",
        "> - **Model Deposu (Hugging Face Hub):** [huggingface.co/ebboo/tracera-model](https://huggingface.co/ebboo/tracera-model) *(last.pt, OpenVINO INT8, FP32 ONNX, meta.json)*",
        "> - **Kurulum Paketleri (GitHub Releases — Windows EXE & Android APK):** `[GitHub Releases (<OWNER>/<REPO>) — Windows EXE + Android APK](https://github.com/<OWNER>/<REPO>/releases)` *(yer tutucu)*",
    ]

    platform_note_block = [
        "",
        "> **Platform Paritesi Notu:** Sonuçlar Windows'ta üretildi; Linux'ta ~0.01 güven kayması gözlendi (CI raporu).",
    ]

    everywhere_block = _build_everywhere_section()

    app_build_cmds = [
        "",
        "### Web, Masaüstü (EXE) ve Android (APK) Çalıştırma / Derleme",
        "",
        "**Web / PWA / Statik Space (`web/`):**",
        "```bash",
        "cd web",
        "npm ci",
        "npm run dev          # http://localhost:5173 (COOP/COEP aktif)",
        "npm run build        # web/dist/ üretim paketi (Tauri masaüstü & APK girdisi)",
        "npm run build:space  # web/web_space/ HF Static Space paketi",
        "```",
        "",
        "**Masaüstü Windows Kurulumu (Tauri 2 EXE, `desktop/`):**",
        "```bash",
        "cd web && npm run build",
        "cd ../desktop && npm ci",
        "npx tauri build --config src-tauri/tauri.windows.conf.json   # dist/tracera-api/ sidecar hazır olmalı",
        "```",
        "",
        "**Android APK (Tauri 2 Android, `desktop/`):**",
        "```bash",
        "cd web && npm run build",
        "cd ../desktop",
        "npx tauri android build",
        "```",
    ]

    out_lines: list[str] = [
        "<!-- Bu taslak scripts/make_readme_final.py tarafından README.md + results/*.csv + KARARLAR.md kaynağından otomatik üretilmiştir. -->",
        "",
    ]

    replaced_demo_line = False
    inserted_platform_note = False
    inserted_everywhere = False
    inserted_build_cmds = False

    for line in base_lines:
        if not replaced_demo_line and line.startswith("> 🌐 **Canlı Web Demosu"):
            out_lines.extend(top_links_block)
            replaced_demo_line = True
            continue

        out_lines.append(line)

        if not inserted_platform_note and line.startswith("> **Metodolojik Not:**"):
            out_lines.extend(platform_note_block)
            inserted_platform_note = True

        if not inserted_everywhere and line.startswith("- Tüm doğruluk sayıları val setindendir"):
            # Wait for the '---' separator right after Section 6 before inserting
            pass

        if not inserted_everywhere and inserted_platform_note and line.strip() == "---":
            # Detect right before Section 7
            pass

    # Clean structural insertion on joined text without modifying a single preserved sentence
    draft = "\n".join(out_lines)

    sec7_marker = "## 7. Endüstriyel PCB-AoI Transfer Deneyi (Null Sonuç)"
    if sec7_marker not in draft:
        raise ValueError(f"Beklenen baslik bulunamadi: {sec7_marker}")
    draft = draft.replace(sec7_marker, f"{everywhere_block}\n\n{sec7_marker}", 1)

    sec10_marker = "## 10. Veri Kaynakları ve Lisans"
    if sec10_marker not in draft:
        raise ValueError(f"Beklenen baslik bulunamadi: {sec10_marker}")
    cmds_text = "\n".join(app_build_cmds)
    draft = draft.replace(
        f"---\n\n{sec10_marker}",
        f"{cmds_text}\n\n---\n\n{sec10_marker}",
        1,
    )

    _check_forbidden(draft, everywhere_block)
    return draft.rstrip() + "\n"


def main() -> int:
    content = build_final_readme_draft()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(content, encoding="utf-8", newline="\n")
    print(f"[make_readme_final] Yazildi: {OUT_PATH} ({len(content.splitlines())} satir)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
