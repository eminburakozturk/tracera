"""README son hali taslagi uretici: docs/README_final_draft.md.

Mevcut README.md dosyasindaki Claude denetimli ana metni KORUR; ust baglantilari,
platform parite notunu, 7. bolum olarak "Her yerde calisir: web, masaustu, Android"
bolumunu (results/*.csv, release/SHA256SUMS.txt, docs/RELEASE_NOTES_v1.0.0.md ve
KARARLAR.md verilerinden), sonraki bolum numaralarinin (8-11) ve Bolum atiflarinin
kaydirilmasini, web/EXE/Android derleme komutlarini ve English Summary platform
maddesini ekler. README.md dosyasina DOKUNMAZ.
"""

from __future__ import annotations

import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
README_PATH = REPO_ROOT / "README.md"
KARARLAR_PATH = REPO_ROOT / "KARARLAR.md"
RESULTS_DIR = REPO_ROOT / "results"
RELEASE_DIR = REPO_ROOT / "release"
RELEASE_NOTES_PATH = REPO_ROOT / "docs" / "RELEASE_NOTES_v1.0.0.md"
WEB_DIR = REPO_ROOT / "web"
OUT_PATH = REPO_ROOT / "docs" / "README_final_draft.md"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Gerekli CSV bulunamadi: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return [{k.strip(): (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f)]


def _num(val: Any) -> float | None:
    try:
        return float(str(val).strip())
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
    """KARARLAR.md (ve varsa RELEASE_NOTES) icindeki olculmus masaustu metriklerini cikarir."""
    text = KARARLAR_PATH.read_text(encoding="utf-8")
    m65 = re.search(
        r"NSIS kurulum (\d+(?:\.\d+)? MiB), kurulu (\d+(?:\.\d+)? GiB), "
        r"pencere (\d+(?:\.\d+)? s), yerel motor hazır (~?\d+(?:\.\d+)? s) "
        r"\(soğuk ilk açılış (\d+(?:\.\d+)? s)\), /detect (\d+(?:\.\d+)? ms), "
        r"/explain (\d+(?:\.\d+)? s)",
        text,
    )
    if not m65:
        raise ValueError("KARARLAR.md 13 Kasim masaustu metrikleri bulunamadi")

    installed_gib = m65.group(2)
    latest_installed = re.findall(r"son derlemede\s+(\d+(?:\.\d+)? GiB)", text)
    if latest_installed:
        installed_gib = latest_installed[-1]
    elif RELEASE_NOTES_PATH.is_file():
        rn_text = RELEASE_NOTES_PATH.read_text(encoding="utf-8")
        m_rn = re.search(r"kurulum sonrası disk kullanımı yaklaşık\s+`(\d+(?:\.\d+)? GiB)`", rn_text)
        if m_rn:
            installed_gib = m_rn.group(1)

    return {
        "nsis_mib": m65.group(1),
        "installed_gib": installed_gib,
        "window_s": m65.group(3),
        "sidecar_ready_s": m65.group(4),
        "sidecar_cold_s": m65.group(5),
        "detect_ms": m65.group(6),
        "explain_s": m65.group(7),
    }


def _extract_release_artifacts() -> dict[str, str]:
    """APK/EXE adini, boyutunu ve SHA-256 ozetini release/ veya RELEASE_NOTES'tan okur."""
    sums_path = RELEASE_DIR / "SHA256SUMS.txt"
    apk_name = ""
    apk_mib = ""
    apk_sha = ""
    exe_name = ""
    exe_mib = ""
    exe_sha = ""

    if sums_path.is_file():
        for raw_line in sums_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            sha, fname = parts[0], parts[1].strip()
            fpath = RELEASE_DIR / fname
            if fname.endswith(".apk"):
                apk_name = fname
                apk_sha = sha
                if fpath.is_file():
                    apk_mib = f"{fpath.stat().st_size / (1024 * 1024):.2f} MiB"
            elif fname.endswith(".exe"):
                exe_name = fname
                exe_sha = sha
                if fpath.is_file():
                    exe_mib = f"{fpath.stat().st_size / (1024 * 1024):.2f} MiB"

    if (not apk_mib or not exe_mib) and RELEASE_NOTES_PATH.is_file():
        rn_text = RELEASE_NOTES_PATH.read_text(encoding="utf-8")
        if not apk_mib:
            m_apk = re.search(
                r"\|\s*`([^`]+\.apk)`\s*\|[^|]+\|\s*([0-9]+(?:\.[0-9]+)?\s*MiB)[^|]*\|\s*`([0-9a-f]{64})`",
                rn_text,
            )
            if m_apk:
                apk_name = apk_name or m_apk.group(1)
                apk_mib = m_apk.group(2)
                apk_sha = apk_sha or m_apk.group(3)
        if not exe_mib:
            m_exe = re.search(
                r"\|\s*`([^`]+\.exe)`\s*\|[^|]+\|\s*([0-9]+(?:\.[0-9]+)?\s*MiB)[^|]*\|\s*`([0-9a-f]{64})`",
                rn_text,
            )
            if m_exe:
                exe_name = exe_name or m_exe.group(1)
                exe_mib = m_exe.group(2)
                exe_sha = exe_sha or m_exe.group(3)

    if not apk_mib:
        raise ValueError("APK boyutu release/SHA256SUMS.txt veya docs/RELEASE_NOTES_v1.0.0.md icinden okunamadi")

    return {
        "apk_name": apk_name or "Tracera_android_arm64.apk",
        "apk_mib": apk_mib,
        "apk_sha": apk_sha,
        "exe_name": exe_name or "Tracera_x64-setup.exe",
        "exe_mib": exe_mib,
        "exe_sha": exe_sha,
    }


def _collect_data() -> dict[str, Any]:
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

    phone_rows = _read_csv(RESULTS_DIR / "device_bench_phone.csv")
    if not phone_rows:
        raise ValueError("results/device_bench_phone.csv bos")

    return {
        "parity_images": parity_images,
        "parity_summary": parity_summary,
        "parity_med_model": parity_med_model,
        "parity_med_total": parity_med_total,
        "wgpu_fp32": wgpu_fp32,
        "wgpu_int8": wgpu_int8,
        "wasm_fp32": wasm_fp32,
        "wasm_int8": wasm_int8,
        "vid_wgpu": vid_wgpu,
        "vid_wasm": vid_wasm,
        "qp025": qp025,
        "qp395": qp395,
        "gpu_hw": gpu_hw,
        "cpu_hw": cpu_hw,
        "b05_desc": b05_desc,
        "onnx_mb": onnx_mb,
        "wasm_mb": wasm_mb,
        "first_detect_mb": round(onnx_mb + wasm_mb),
        "precache_mb": _measure_precache_mb(),
        "desk": _extract_desktop_metrics(),
        "rel": _extract_release_artifacts(),
        "phone": phone_rows[0],
    }


def _build_everywhere_section(data: dict[str, Any]) -> str:
    parity_images = data["parity_images"]
    parity_summary = data["parity_summary"]
    parity_med_model = data["parity_med_model"]
    parity_med_total = data["parity_med_total"]
    wgpu_fp32 = data["wgpu_fp32"]
    wgpu_int8 = data["wgpu_int8"]
    wasm_fp32 = data["wasm_fp32"]
    wasm_int8 = data["wasm_int8"]
    vid_wgpu = data["vid_wgpu"]
    vid_wasm = data["vid_wasm"]
    qp025 = data["qp025"]
    qp395 = data["qp395"]
    gpu_hw = data["gpu_hw"]
    cpu_hw = data["cpu_hw"]
    b05_desc = data["b05_desc"]
    onnx_mb = data["onnx_mb"]
    wasm_mb = data["wasm_mb"]
    first_detect_mb = data["first_detect_mb"]
    precache_mb = data["precache_mb"]
    desk = data["desk"]
    rel = data["rel"]
    phone = data["phone"]

    int8_note = "INT8 dinamik ONNX elendi (WebGPU'da çalışmadı, WASM'da doğruluk düştü)"
    exe_size_label = rel["exe_mib"] or desk["nsis_mib"]

    lines = [
        "## 7. Her yerde çalışır: web, masaüstü, Android",
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
        f"- **Tauri 2 Masaüstü (`desktop/`, Windows EXE — `{rel['exe_name']}`, `{exe_size_label}`):** Aynı arayüzü `isTauri()` köprüsüyle (`get_sidecar_status` / `get_api_base_url` / `get_platform` / `tracera://api-ready`) paketlenmiş Python FastAPI yan sunucusuna bağlar; varsayılan motor **Yerel sunucu**dur (OpenVINO INT8 tespit + LayerCAM Katman 4).",
        f"- **Android APK (Tauri 2 WebView — `{rel['apk_name']}`, `{rel['apk_mib']}` debug imzalı):** `get_sidecar_status.available = false` ve `get_platform = \"android\"` bildirildiğinden yerel motor ekranı gösterilmez; tespit doğrudan **ORT-Web** üzerinde çalışır ve **Yerel sunucu** gizlenir, ısı haritası yalnız Bulut'a (`ebboo/tracera`) yönlenir.",
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
        f"| Tekil görüntü, 10 val (`web_spike.csv`) | {wgpu_fp32['backend']} | {wgpu_fp32['model']} ({wgpu_fp32['boyut_mb']} MB) | {wgpu_fp32['medyan_ms']} ms | — | Birincil tarayıcı motoru (parite yukarıda `web_parity.csv` bölümünde) |",
        f"| Tekil görüntü, 10 val (`web_spike.csv`) | {wgpu_int8['backend']} | {wgpu_int8['model']} ({wgpu_int8['boyut_mb']} MB) | zaman aşımı | — | {int8_note} |",
        f"| Tekil görüntü, 10 val (`web_spike.csv`) | {wasm_fp32['backend']} | {wasm_fp32['model']} ({wasm_fp32['boyut_mb']} MB) | {wasm_fp32['medyan_ms']} ms | — | WebGPU yokken yedek; tek iş parçacıklı WASM |",
        f"| Tekil görüntü, 10 val (`web_spike.csv`) | {wasm_int8['backend']} | {wasm_int8['model']} ({wasm_int8['boyut_mb']} MB) | {wasm_int8['medyan_ms']} ms | — | {int8_note} |",
        f"| Canlı video 5.2 s, `val_pan.mp4` (`web_video_fps.csv`) | {vid_wgpu['ort_ep']} | FP32 ONNX | {vid_wgpu['model_ms']} ms | {vid_wgpu['avg_fps']} FPS ({vid_wgpu['frames']} kare) | Kuyruk yok (meşgulken kare atlanır) |",
        f"| Canlı video 5.2 s, `val_pan.mp4` (`web_video_fps.csv`) | {vid_wasm['ort_ep']} | FP32 ONNX | {vid_wasm['model_ms']} ms | {vid_wasm['avg_fps']} FPS ({vid_wasm['frames']} kare) | WASM üzerinde akıcı video beklenmez |",
        "",
        "### Masaüstü Uygulaması (Windows Tauri 2 EXE + FastAPI Sidecar)",
        "",
        f"Windows kurulum ve ilk çalıştırma ölçümleri (`KARARLAR.md`, 13–14 Kasım kayıtları, **{cpu_hw}**):",
        "",
        "| Ölçüm | Değer | Not |",
        "|---|---|---|",
        f"| NSIS kurulum dosyası (`{rel['exe_name']}`) | **{exe_size_label}** | İmzasız derleme (ilk açılışta Windows SmartScreen uyarısı gösterebilir) |",
        f"| Diskteki kurulu boyut | **{desk['installed_gib']}** | Web arayüzü + Python FastAPI CPU sidecar + OpenVINO INT8 / PyTorch ağırlıkları |",
        f"| Pencere açılış süresi | **{desk['window_s']}** | Tauri WebView2 penceresi |",
        f"| Yerel motor hazır süresi | **{desk['sidecar_ready_s']}** (soğuk ilk açılış **{desk['sidecar_cold_s']}**) | Rastgele yerel port üzerinde OpenVINO INT8 + LayerCAM yüklemesi |",
        f"| Tekil tespit (`/detect`) | **{desk['detect_ms']}** | OpenVINO INT8 CPU çıkarımı |",
        f"| Isı haritası (`/explain`) | **{desk['explain_s']}** | PyTorch LayerCAM @ Katman 4 gradyan hesabı |",
        "",
        "### Android Ölçümü (`results/device_bench_phone.csv` — Tek Cihaz)",
        "",
        "| Cihaz / Ortam | Motor & GPU Adaptör | Medyan Model | Medyan Toplam | Koşu Sayısı | APK Paketi | Kaynak Notu |",
        "|---|---|---:|---:|---:|---|---|",
        f"| `{phone['cihaz']}` ({phone['os']}, {phone['webview']}) | `{phone['backend']}` (`{phone['gpu_adaptor']}`) | **{phone['medyan_model_ms']} ms** | **{phone['medyan_toplam_ms']} ms** | {phone['kosu_n']} (ısınma: {phone['isinma']}) | `{rel['apk_name']}` (**{rel['apk_mib']}**) | {phone['kaynak']} |",
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
        "- **LayerCAM ısı haritası tarayıcıda hesaplanmaz:** Katman 4 aktivasyon × pozitif gradyan geri yayılımı gerektirdiği için XAI açıklaması tarayıcı içinde üretilmez; masaüstünde **Yerel sunucu** (`api/`), ulaşılamazsa veya Android'de **Bulut** (`ebboo/tracera` Space `/explain_api`) ucuna yönlendirilir.",
        "",
        "---",
    ]
    return "\n".join(lines)


def _check_forbidden(markdown: str, everywhere_section: str) -> None:
    lower_all = markdown.lower()
    for term in (
        "mikron",
        "micron",
        "gerçek zamanlı",
        "gercek zamanli",
        "64.3 mib",
        "space gizliyken bulut",
        "eşleşme 0.7",
        "kararlar.md` #65",
        "kararlar.md #65",
        "<owner>",
        "<repo>",
        "yer tutucu",
    ):
        if term in lower_all:
            raise ValueError(f"Yasak veya guncel olmayan ifade tespit edildi: {term!r}")

    if "linux'ta ~0.01 güven kayması gözlendi (ci raporu)" in lower_all:
        raise ValueError("Eski Platform Paritesi Notu cumlesi kalmis")

    # Oracle P/R degerleri sadece Metodolojik Not (uyari) icinde gecmeli, baslik maddelerinde yer almamali
    headline_part = markdown.split("## 2.", 1)[0].split("> **Metodolojik Not:**", 1)[0]
    for oracle_val in ("0.9462", "0.8577"):
        if oracle_val in headline_part:
            raise ValueError(f"Oracle P/R degeri baslik metninde yer alamaz: {oracle_val}")

    # Yeni eklenen bolumde "telefonda X FPS" genellemesi olmamali
    if re.search(r"telefonda.*\bfps\b", everywhere_section.lower()):
        raise ValueError("Yeni bolumde 'telefonda ... FPS' genellemesi tespit edildi")

    # Eski bolum 7-10 numaralari kalmamis olmali; yeni siralama 1..11 kesintisiz olmali
    headings = re.findall(r"^## (\d+)\.", markdown, flags=re.MULTILINE)
    if headings != [str(i) for i in range(1, 12)]:
        raise ValueError(f"Bolum numaralandirmasi 1..11 degil: {headings}")

    if "(Bölüm 8)" in markdown:
        raise ValueError("Eski '(Bölüm 8)' atifi '(Bölüm 9)' olarak guncellenmemis")


def build_final_readme_draft() -> str:
    data = _collect_data()
    raw_base = README_PATH.read_text(encoding="utf-8")
    # Zaten v1.0.0 yapisina donusturulmus README.md uzerinde yeniden calisilirsa
    # yalnizca veri odakli bolumleri (Bölüm 7 ve English Summary) tazele.
    everywhere_block = _build_everywhere_section(data)
    two_way_pct_1dp = f"{float(data['parity_summary']['two_way_match_pct']):.1f}%"
    n_parity_imgs = len(data["parity_images"])
    phone_model_ms_int = round(float(data["phone"]["medyan_model_ms"]))
    english_platform_bullet = (
        f"- **Runs everywhere from one React/TypeScript codebase:** in-browser inference with "
        f"ONNX Runtime Web (WebGPU, WASM fallback; {two_way_pct_1dp} two-way box match with PyTorch "
        f"on {n_parity_imgs} val images), offline PWA, Windows desktop app (Tauri 2 + bundled "
        f"FastAPI/OpenVINO engine, LayerCAM offline) and Android APK (on-device WebGPU, measured "
        f"on one phone: {phone_model_ms_int} ms model median; heatmaps via cloud)."
    )

    draft_comment = "<!-- Bu taslak scripts/make_readme_final.py tarafından README.md + results/*.csv + release/SHA256SUMS.txt + KARARLAR.md kaynağından otomatik üretilmiştir. -->\n\n"

    if "## 7. Her yerde çalışır: web, masaüstü, Android" in raw_base:
        body = re.sub(r"^<!-- Bu taslak .*?-->\s*", "", raw_base, count=1, flags=re.DOTALL)
        sec7_re = re.compile(
            r"## 7\. Her yerde çalışır: web, masaüstü, Android.*?\n---(?=\n\n## 8\.)",
            flags=re.DOTALL,
        )
        if not sec7_re.search(body):
            raise ValueError("Mevcut README.md icinde Bölüm 7 sinirlari bulunamadi")
        body = sec7_re.sub(lambda _m: everywhere_block, body, count=1)

        eng_bullet_re = re.compile(
            r"^- \*\*Runs everywhere from one React/TypeScript codebase:\*\*.*$",
            flags=re.MULTILINE,
        )
        if eng_bullet_re.search(body):
            body = eng_bullet_re.sub(lambda _m: english_platform_bullet, body, count=1)
        else:
            body = body.replace("\nCopyright (C)", f"\n{english_platform_bullet}\n\nCopyright (C)", 1)

        draft = f"{draft_comment}{body}"
        _check_forbidden(draft, everywhere_block)
        return draft.rstrip() + "\n"

    base_lines = raw_base.splitlines()

    demo_banner_block = [
        "",
        "![Tracera demo](results/demo/tracera_demo.gif)",
        "",
        "Tam demo (50 sn): [results/demo/tracera_demo.mp4](results/demo/tracera_demo.mp4)",
    ]

    top_links_block = [
        "[![CI](https://github.com/eminburakozturk/tracera/actions/workflows/ci.yml/badge.svg)](https://github.com/eminburakozturk/tracera/actions/workflows/ci.yml)",
        "",
        "> - **Canlı Statik Web Demosu (Tarayıcıda ONNX):** [huggingface.co/spaces/ebboo/tracera-web](https://huggingface.co/spaces/ebboo/tracera-web) *(yayında — tarayıcıda ONNX)*",
        "> - **Bulut Demo & LayerCAM API (Hugging Face Spaces, Gradio):** [huggingface.co/spaces/ebboo/tracera](https://huggingface.co/spaces/ebboo/tracera)",
        "> - **Model Deposu (Hugging Face Hub):** [huggingface.co/ebboo/tracera-model](https://huggingface.co/ebboo/tracera-model) *(last.pt, OpenVINO INT8, FP32 ONNX, meta.json)*",
        "> - **Kurulum Paketleri (GitHub Releases — Windows EXE & Android APK):** [GitHub Releases (eminburakozturk/tracera) — Windows EXE + Android APK](https://github.com/eminburakozturk/tracera/releases)",
    ]

    platform_note_block = [
        "",
        "> **Platform Paritesi Notu:** Sonuçlar Windows'ta üretildi; Linux Docker testlerinde aynı görüntüde ~0.01 güven farkı gözlendi (ör. 0.8312 → 0.8408); CI Linux parite raporu bu farkı ölçmek için eklendi.",
    ]

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
        "conda activate tracera-build && python api/build_exe.py   # dist/tracera-api/ sidecar üretir",
        "cd web && npm run build                                   # web/dist/ hazırlar",
        "cd ../desktop && npm ci",
        "npx tauri build --config src-tauri/tauri.windows.conf.json",
        "```",
        "",
        "**Android APK (Tauri 2 Android, `desktop/`):**",
        "- Gereksinimler: Android Studio + SDK, Android NDK, `JAVA_HOME` / `ANDROID_HOME` / `NDK_HOME` ortam değişkenleri ve Windows Geliştirici Modu.",
        "```bash",
        "cd web && npm run build",
        "cd ../desktop",
        "npx tauri android build --apk --target aarch64",
        "```",
    ]

    out_lines: list[str] = [
        "<!-- Bu taslak scripts/make_readme_final.py tarafından README.md + results/*.csv + release/SHA256SUMS.txt + KARARLAR.md kaynağından otomatik üretilmiştir. -->",
        "",
    ]

    inserted_demo_banner = False
    replaced_demo_line = False
    inserted_platform_note = False

    for line in base_lines:
        if not replaced_demo_line and line.startswith("> 🌐 **Canlı Web Demosu"):
            out_lines.extend(top_links_block)
            replaced_demo_line = True
            continue

        out_lines.append(line)

        if not inserted_demo_banner and line.startswith("# Tracera:"):
            out_lines.extend(demo_banner_block)
            inserted_demo_banner = True

        if not inserted_platform_note and line.startswith("> **Metodolojik Not:**"):
            out_lines.extend(platform_note_block)
            inserted_platform_note = True

    draft = "\n".join(out_lines)

    # Bolum 7-10'u 8-11'e kaydir (sondan basa dogru cakismanin onune gecmek icin)
    heading_shifts = [
        ("## 10. Veri Kaynakları ve Lisans", "## 11. Veri Kaynakları ve Lisans"),
        ("## 9. Projeyi Çalıştırma", "## 10. Projeyi Çalıştırma"),
        ("## 8. Sınırlılıklar ve Açık Noktalar", "## 9. Sınırlılıklar ve Açık Noktalar"),
        ("## 7. Endüstriyel PCB-AoI Transfer Deneyi (Null Sonuç)", "## 8. Endüstriyel PCB-AoI Transfer Deneyi (Null Sonuç)"),
    ]
    for old_h, new_h in heading_shifts:
        if old_h not in draft:
            raise ValueError(f"Beklenen baslik bulunamadi: {old_h}")
        draft = draft.replace(old_h, new_h, 1)

    # Bolum atiflarini yeni numaralamaya gore guncelle (eski Bolum 8 -> Bolum 9)
    if "(Bölüm 8)" not in draft:
        raise ValueError("Beklenen '(Bölüm 8)' atifi bulunamadi")
    draft = draft.replace("(Bölüm 8)", "(Bölüm 9)", 1)

    sec8_marker = "## 8. Endüstriyel PCB-AoI Transfer Deneyi (Null Sonuç)"
    draft = draft.replace(sec8_marker, f"{everywhere_block}\n\n{sec8_marker}", 1)

    sec11_marker = "## 11. Veri Kaynakları ve Lisans"
    cmds_text = "\n".join(app_build_cmds)
    draft = draft.replace(
        f"---\n\n{sec11_marker}",
        f"{cmds_text}\n\n---\n\n{sec11_marker}",
        1,
    )

    english_transfer_line = "- **Transfer to PCB-AoI**"
    if english_transfer_line not in draft:
        raise ValueError(f"Beklenen English Summary satiri bulunamadi: {english_transfer_line}")
    draft = draft.replace(
        "\nCopyright (C)",
        f"\n{english_platform_bullet}\n\nCopyright (C)",
        1,
    )

    _check_forbidden(draft, everywhere_block)
    return draft.rstrip() + "\n"


def strip_draft_comment(markdown: str) -> str:
    return re.sub(r"^<!-- Bu taslak .*?-->\s*", "", markdown, count=1, flags=re.DOTALL)


def main() -> int:
    import sys

    write_readme = "--write-readme" in sys.argv[1:]
    content = build_final_readme_draft()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(content, encoding="utf-8", newline="\n")
    print(f"[make_readme_final] Yazildi: {OUT_PATH} ({len(content.splitlines())} satir)")
    if write_readme:
        readme_text = strip_draft_comment(content)
        README_PATH.write_text(readme_text, encoding="utf-8", newline="\n")
        print(f"[make_readme_final] Yazildi: {README_PATH} ({len(readme_text.splitlines())} satir)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
