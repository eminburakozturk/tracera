# Tracera v1.1.0 - Surum Derleme Gunlugu (release_build_log.md)

- **Tarih / Saat:** 2026-11-17 03:52:54
- **Git Dal / Commit SHA:** `main` / `76cd832` (`76cd832ed4365b98e84e0ac9981c12748cdbe4ea`)
- **Uygulama Surumu (Tek Kaynak):** `v1.1.0`
- **Model Surumu ve ONNX SHA256:** `v1.0.0` / `aa8e1f42702c823426df778c9ff298293e8661010dce1d410ea89b6caccbb58c`
- **Genel Durum:** **BASARILI** (Toplam sure: `1272.6 s`)
- **Android APK Imza Durumu:** UYARI: Release keystore tanimli degil; debug imzali APK uretilecek

## 1. Surum Tutarliligi (g)

| Dosya | Alan | Deger | Durum |
|---|---|---|---|
| `web/package.json` | `version` | `1.1.0` | OK |
| `desktop/package.json` | `version` | `1.1.0` | OK |
| `desktop/src-tauri/tauri.conf.json` | `version` | `1.1.0` | OK |
| `desktop/src-tauri/Cargo.toml` | `version` | `1.1.0` | OK |
| `web/public/models/meta.json` | `version` | `1.1.0` | OK |
| `api/main.py` | `APP_VERSION` | `1.1.0` | OK |
| `tracera_core/config.py` | `APP_VERSION` | `1.1.0` | OK |
| `web/public/models/meta.json` | `model_version` | `1.0.0` | OK |
| `web/public/models/meta.json` | `onnx_sha256` | `aa8e1f42702c8234...` | OK |

## 2. Derleme Adimlari (a-f)

| Adim | Aciklama | Durum | Sure (s) | Cikti / Boyut |
|---|---|---|---:|---|
| **(a)** | Web birim testleri + uretim build (`web/dist`) | **BASARILI** | 27.3 | `web/dist/`: 78,52 MiB (82.335.404 B) (56 dosya) |
| **(b)** | Statik Web Space paketi (`build_web_space.mjs`) | **BASARILI** | 20.6 | `web/web_space/`: 78,52 MiB (82.335.819 B) (57 dosya) |
| **(c)** | Gradio Space paketi ve izole testi (`build_space.py --test`) | **BASARILI** | 20.8 | `space/`: 32,68 MiB (34.268.862 B) (23 dosya) |
| **(d)** | Sidecar PyInstaller + canli v1.3 API dogrulamasi + Tauri NSIS EXE | **BASARILI** | 1052 | `Tracera_1.1.0_x64-setup.exe`: 316,24 MiB (331.600.758 B) (health.version=1.1.0, energy_in_box=0.0576, peak_in_box=False) |
| **(e)** | Tauri Android APK (arm64) | **BASARILI** | 150.6 | `app-universal-release.apk`: 59,29 MiB (62.170.347 B) [debug imzali (UYARI)] |
| **(f)** | `dist_release/` paketleme ve SHA256SUMS.txt | **BASARILI** | 1.1 | EXE: 316,24 MiB (331.600.758 B), APK: 59,29 MiB (62.170.347 B), `SHA256SUMS.txt` hazir |
