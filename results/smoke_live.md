# Tracera — Canlı Hugging Face Space Duman Testi (`smoke_live.md`)

- **Tarih / Saat:** `2026-11-14 12:38:41`
- **Hedef Space:** `https://huggingface.co/spaces/ebboo/tracera` (`ebboo/tracera`)
- **Test Görüntüsü:** `app/examples/val_05_missing_hole_01.jpg`
- **Güven Eşiği (`conf`):** `0.395`
- **Genel Sonuç:** **BASARILI (PASS)**

## 1. İstek Süreleri ve Deneme Sayıları

| Aşama | Süre (s) | Deneme | Durum |
|---|---:|---:|---|
| Space bağlantısı (`Client`) | 2.36 | 1 | OK |
| `/detect_api` | 11.07 | 1 | OK |
| `/explain_api` | 16.67 | 1 | OK |

## 2. `/detect_api` Çıktı Alanları

- **Boyut (`width × height`):** `2544 × 2156`
- **Backend:** `openvino-int8`
- **Tespit Sayısı:** `4`
- **Sunucu Zamanlaması (`timing`):** `{"pre_ms": 11.521, "model_ms": 13.878, "post_ms": 6.742, "total_ms": 32.142}`

| id | cls | cls_id | conf | box [x1, y1, x2, y2] |
|---:|---|---:|---:|---|
| 0 | `missing_hole` | 0 | 0.8322 | `[1337.8, 1657.8, 1382.4, 1710.9]` |
| 1 | `missing_hole` | 0 | 0.8152 | `[1527.0, 970.4, 1573.5, 1020.9]` |
| 2 | `missing_hole` | 0 | 0.7994 | `[1103.1, 568.0, 1169.4, 619.3]` |
| 3 | `spurious_copper` | 5 | 0.5221 | `[1507.2, 1147.8, 1568.0, 1217.7]` |

## 3. `/explain_api` Çıktı Alanları (CONTRACT v1.3)

- **Açıklanan Kutu (`box`, `cls_id`):** `1337.836,1657.829,1382.399,1710.921` (`cls_id=0`)
- **Dönen Anahtarlar:** `energy_in_box, layer, method, overlay_png, peak_in_box, peak_xy, timing, zoom_png`
- **`method` / `layer`:** `layercam` / `4`
- **`overlay_png` uzunluğu:** `7854596 karakter`
- **`zoom_png` uzunluğu:** `220884 karakter`
- **`energy_in_box`:** `0.785114`
- **`peak_in_box`:** `True`
- **`peak_xy`:** `[1362, 1685]`
- **Sunucu Zamanlaması (`timing`):** `{"pre_ms": 11.403, "model_ms": 1538.929, "post_ms": 145.017, "total_ms": 1695.349}`
