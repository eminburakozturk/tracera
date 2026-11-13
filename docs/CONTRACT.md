# Tracera — Arayüz / Model / API Sözleşmesi (v1)

Bu dosya iki paralel çalışma hattının (web arayüzü ve model/API) ortak sözleşmesidir.
Değişiklik yalnızca Claude onayıyla yapılır; değişirse sürüm numarası artar ve KARARLAR.md'ye yazılır.

## 1. Model sabitleri (tek kaynak: `web/public/models/meta.json`)
```json
{
  "version": "1",
  "model": "yolo11s",
  "imgsz": 1024,
  "conf": 0.395,
  "iou_nms": 0.7,
  "letterbox_pad": 114,
  "classes": ["missing_hole","mouse_bite","open_circuit","short","spur","spurious_copper"],
  "colors": ["#38bdf8","#f472b6","#facc15","#f87171","#a78bfa","#34d399"],
  "onnx": "tracera_1024.onnx"
}
```
- `conf` = val'de F1-maks ile seçilen KİLİTLİ eşik. Arayüzde kaydırıcı olabilir ama varsayılan bu.
- Ön işleme Ultralytics ile birebir: letterbox (oran korunur, ortalanır, dolgu 114), BGR->RGB, /255, NCHW float32.
- Son işleme: sınıf bazlı NMS (IoU 0.7), sonra conf filtresi; kutular orijinal görüntü koordinatına geri ölçeklenir.

## 2. Ortak tipler (TypeScript ve Pydantic aynı alan adları)
```ts
type Detection = { id: number; cls: string; cls_id: number; conf: number; box: [number, number, number, number] }; // x1,y1,x2,y2 orijinal piksel
type Timing = { pre_ms: number; model_ms: number; post_ms: number; total_ms: number };
type DetectResult = { width: number; height: number; detections: Detection[]; timing: Timing; backend: string }; // backend: "ort-webgpu" | "ort-wasm" | "openvino-int8" | "pytorch"
type ExplainResult = { overlay_png: string; zoom_png: string; method: "layercam"; layer: 4; timing: Timing }; // png'ler base64 (data URL değil, ham base64)
```

## 3. Yerel API (FastAPI, `api/`, varsayılan `http://127.0.0.1:8765`)
| Uç | Girdi | Çıktı |
|---|---|---|
| `GET /health` | — | `{status:"ok", model, version, device, backends:[...]}` |
| `POST /detect` | multipart: `image`, ops. `conf` (float) | `DetectResult` (backend = "openvino-int8") |
| `POST /explain` | multipart: `image`, `box` ("x1,y1,x2,y2"), `cls_id` | `ExplainResult` (LayerCAM katman 4; hedef anchor kutuya en iyi IoU ile seçilir, scripts/gradcam.py mantığı) |
| `GET /metrics` | — | results/*.csv'den özet JSON: `operating_point`, `edge_benchmark`, `cv`, `xai`, `aoi` |
- CORS: `http://localhost:5173`, `tauri://localhost`, `http://tauri.localhost`, `https://tauri.localhost`.
- Hata biçimi: `{error: string}` + uygun HTTP kodu. Görüntü sınırı 25 MB.

## 4. Kurallar
- Sayılar elle yazılmaz: arayüzdeki tüm metrikler `/metrics` veya `web/public/metrics.json` (build sırasında csv'den üretilir).
- Test görüntüleri (kart 06, 10) örnek olarak kullanılmaz; örnekler yalnız val kartları (05, 11).
- Tarayıcı çıkarımı, Python çıktısıyla karşılaştırması geçmeden "doğru" sayılmaz (bkz. results/web_spike.csv).
