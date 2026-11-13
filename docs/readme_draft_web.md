<!-- Bu dosya web/scripts/make_readme_web.mjs tarafından results/*.csv ve derleme çıktılarından otomatik üretilir. Elle sayı düzenlemeyin. -->

## Tarayıcıda / Cihazda Çalıştırma

### Mimari

Tracera, tek bir React + TypeScript (`web/`) arayüzünü dört dağıtım hedefine taşır:

- **Statik Web / Hugging Face Static Space (`web_space/`)**: Sunucu tarafı GPU gerektirmeden doğrudan kullanıcı tarayıcısında ONNX Runtime Web (WebGPU öncelikli, yoksa tek iş parçacıklı WASM) ile `1024×1024` FP32 ONNX çıkarımı yapar.
- **PWA (Progressive Web App)**: Uygulama kabuğunu Service Worker ile önbelleğe alır; ONNX modeli ve ORT çalışma zamanını yalnızca ilk tespitte indirerek sonraki açılışlarda çevrimdışı tespit sunar.
- **Tauri 2 Masaüstü (`desktop/`, Windows EXE)**: Aynı web arayüzünü `isTauri()` köprüsüyle (`get_sidecar_status` / `get_api_base_url` / `tracera://api-ready`) paketlenmiş Python FastAPI yan sunucusuna bağlar; varsayılan motor **Yerel sunucu** (OpenVINO INT8 tespit + LayerCAM Katman 4) olarak açılır.
- **Android APK Hazırlığı (Tauri WebView)**: Aynı WebGPU → tek iş parçacıklı WASM düşüşüyle Android WebView üzerinde tarayıcı içi çıkarım uyumu hedeflenir (ayrıntılar: `docs/android_webview_notes.md`).

### Tarayıcı Paritesi (`OrtDetector` WebGPU vs. PyTorch FP32 Referans)

10 val görüntüsü üzerinde (`results/web_parity.csv`, `conf = 0.25` iki yönlü IoU ≥ 0.5 eşleşmesi ve `conf = 0.395` sayım kıyası):

| Kıyas | Eşleşen (≥0.25) | Yalnız ORT | Yalnız PyTorch | İki Yönlü Eşleşme | Ort. IoU | Ort. \|Δconf\| | Sayım Farkı (0.395) | Ort. Model (medyan) | Uçtan Uca (medyan) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ort-webgpu vs. PyTorch FP32 | 59 | 1 | 3 | %96.72 | 0.9885 | 0.0065 | 45 → 44 (-1) | 51.0 ms | 114.3 ms |

> Ölçüm ortamı: Windows masaüstü Chrome, **NVIDIA GeForce RTX 4060 Laptop GPU** (ONNX Runtime Web `ort-webgpu`, FP32 ONNX). `0.395` çalışma noktasındaki 1 kutuluk fark eşik sınırındaki bir tahminden kaynaklanır.

### Tarayıcı Hızları (Statik Görüntü & Canlı Video)

Aşağıdaki süreler dizüstü bilgisayar donanımında (**NVIDIA GeForce RTX 4060 Laptop GPU** / **AMD Ryzen 7 8845HS w/ Radeon 780M Graphics**, Windows Desktop Chrome) ölçülmüştür:

| Senaryo / Kaynak | Motor | Model | Medyan Model Süresi | Canlı Video FPS | Notlar |
|---|---|---|---:|---:|---|
| Tekil görüntü, 10 val (`web_spike.csv`) | ort-webgpu | FP32 ONNX (36.43 MB) | 36.05 ms | — | Ort. IoU 0.9908, \|Δconf\| 0.001589 |
| Tekil görüntü, 10 val (`web_spike.csv`) | ort-webgpu | INT8 Dynamic ONNX (9.64 MB) | zaman aşımı | — | Kullanımdan çıkarıldı |
| Tekil görüntü, 10 val (`web_spike.csv`) | ort-wasm | FP32 ONNX (36.43 MB) | 689.41 ms | — | WebGPU yokken yedek; tek iş parçacıklı WASM |
| Tekil görüntü, 10 val (`web_spike.csv`) | ort-wasm | INT8 Dynamic ONNX (9.64 MB) | 607.61 ms | — | Eşleşme 0.7; doğruluk kaybı nedeniyle elendi |
| Canlı video 5.2 s, `val_pan.mp4` (`web_video_fps.csv`) | ort-webgpu | FP32 ONNX | 36 ms | 16.09 FPS (84 kare) | Kuyruk yok (meşgulken kare atlanır) |
| Canlı video 5.2 s, `val_pan.mp4` (`web_video_fps.csv`) | ort-wasm | FP32 ONNX | 2241.8 ms | 0.38 FPS (2 kare) | WASM üzerinde akıcı video beklenmez |

### FP32 / INT8 Farkı ve Şablon Noktası Notu

Tarayıcı (`FP32 ONNX`) ile yerel masaüstü yan sunucusu (`OpenVINO INT8`) sınır değerdeki kutularda küçük farklar gösterebilir (`results/quant_parity.csv`, 10 val görüntüsü):

| Güven Eşiği | Eşleşen | Yalnız FP32 | Yalnız INT8 | Ort. IoU | Ort. \|Δconf\| |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 59 | 3 | 1 | 0.9537 | 0.0320 |
| 0.395 | 43 | 2 | 5 | 0.9530 | 0.0160 |

`conf ≥ 0.395` noktasındaki tüm FP32–INT8 farkları rastgele dağılmamıştır; Kart 05 şablonunda temiz kartta da yanlış alarm üreten 2 sabit `spurious_copper` arka plan konumunda (`conf=0.5108` @ (1606.5, 1072.9) ve `conf=0.4475` @ (1537.1, 1183.2)) yoğunlaşmıştır.

### Çevrimdışı Kullanım, PWA ve İlk İndirme Boyutu

- **İlk açılış precache toplamı**: yalnızca uygulama kabuğu (**1.09 MB**). Model açılışta kendiliğinden indirilmez.
- **İlk tespitte çalışma zamanı indirmesi**: FP32 ONNX (**36.43 MB**) + WebGPU/tek iş parçacıklı WASM JSEP çalışma zamanı (**28.15 MB**) olmak üzere yaklaşık **~65 MB** CacheFirst stratejisiyle indirilir ve önbelleğe alınır.
- **Çevrimdışı devam**: İlk tespit sonrasında internet kesilse bile tarayıcı içi tespit ("Cihazda") çevrimdışı çalışmaya devam eder.

### Bilinen Sınırlılıklar (Tarayıcı & Mobil)

- **WASM canlı video performansı**: WebGPU desteği olmayan cihazlarda ORT WASM ile canlı video çıkarımı `< 1 FPS` düzeyindedir (0.38 FPS @ AMD Ryzen 7 8845HS w/ Radeon 780M Graphics); akıcı video akışı yerine tekil fotoğraf tespiti önerilir.
- **Kamera modu deneyseldir**: Model PKU-Market-PCB veri setindeki yukarıdan çekilmiş yüksek çözünürlüklü kart fotoğraflarıyla eğitilmiştir; el kamerası perspektifi, parlamalar veya odak bozukluğunda doğruluk düşebilir.
- **LayerCAM ısı haritası tarayıcıda hesaplanmaz**: Katman 4 aktivasyon × pozitif gradyan geri yayılımı gerektirdiği için XAI açıklaması tarayıcı içinde üretilmez; "Cihazda" tespit modundayken açıklama isteği sırasıyla **Yerel sunucu**ya (`api/`), ulaşılamazsa **Bulut** (`ebboo/tracera` Space `/explain_api`) ucuna yönlendirilir.
- **Telefon hızları ayrıca ölçülür**: Bu belgede herhangi bir telefon modeli için ölçülmemiş FPS/gecikme iddiası verilmez; kullanıcılar **Sonuçlar → Bu cihazda hız testi** üzerinden kendi cihazlarında `4 val görüntüsü × 5 tekrar` medyanını yerel olarak ölçebilir.
