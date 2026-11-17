# Tracera v1.0.0 Sürüm Notları

Tracera; PKU-Market-PCB veri seti üzerinde eğitilmiş YOLO11s (`1024×1024`) modeliyle 6 PCB üretim hatasını (`missing_hole`, `mouse_bite`, `open_circuit`, `short`, `spur`, `spurious_copper`) tespit eden masaüstü ve Android uygulamasıdır.

## 1. Sürüm Paketi İçeriği

| Dosya | Platform | Boyut | SHA-256 |
|---|---|---:|---|
| `Tracera_1.0.0_x64-setup.exe` | Windows (`x86_64`) | 315.03 MiB (330,336,072 bayt) | `23d1c34cf4160b5b7bec8b7293588d0126ee7e82dddf3fb48ea268f81af9d6c8` |
| `Tracera_1.0.0_android_arm64.apk` | Android (`arm64-v8a`) | 58.29 MiB (61,123,275 bayt) | `ecadf956d04065a335b8d577e07a296f41c2781f26848b0570389a58bbe17559` |
| `SHA256SUMS.txt` | Tümü | 194 bayt | — |

- **Windows Masaüstü (`Tracera_1.0.0_x64-setup.exe`)**: Tauri 2 arayüzü + yerleşik CPU FastAPI yan sunucusu (`tracera-api`, OpenVINO INT8 tespit + Katman 4 LayerCAM ısı haritası) ve isteğe bağlı tarayıcı (`ort-webgpu` / `ort-wasm`) çıkarımı. Kurulum sonrası disk kullanımı yaklaşık `1.02 GiB`.
- **Android (`Tracera_1.0.0_android_arm64.apk`)**: Tauri 2 WebView içinde doğrudan cihazda ONNX Runtime Web (`ort-webgpu` öncelikli, yoksa `ort-wasm`) ile `1024×1024` FP32 ONNX (`tracera_1024.onnx`) çıkarımı, galeri/kamera girişi ve uygulama içi hız testi.

Bütünlük doğrulaması için:

```powershell
Get-FileHash .\Tracera_1.0.0_x64-setup.exe, .\Tracera_1.0.0_android_arm64.apk -Algorithm SHA256
```

```bash
sha256sum -c SHA256SUMS.txt
```

## 2. Kurulum ve Güvenlik Uyarıları

### Windows (`Tracera_1.0.0_x64-setup.exe`)
- Kurulum dosyası kod imzalama sertifikasıyla **imzalanmamıştır**.
- Çalıştırdığınızda **Windows Defender SmartScreen** *"Windows bilgisayarınızı korudu"* uyarısı gösterebilir. Devam etmek için **"Daha fazla bilgi"** bağlantısına, ardından **"Yine de çalıştır"** düğmesine tıklayın.
- İlk açılışta gömülü yerel motorun başlatılması birkaç saniye sürebilir (yerel API hazır olana kadar arayüzde başlatılıyor bilgisi görünür).

### Android (`Tracera_1.0.0_android_arm64.apk`)
- APK yalnızca geliştirici **debug anahtarıyla imzalanmıştır** (`C=US, O=Android, CN=Android Debug`); Google Play üzerinden dağıtılmaz.
- Yükleme sırasında Android ayarlarından kurulumu yaptığınız uygulama (dosya yöneticisi / tarayıcı) için **"Bilinmeyen kaynaklardan uygulama yükleme"** (*Bilinmeyen uygulamalara izin ver*) iznini vermeniz gerekir. Google Play Protect ek onay isterse *"Yine de yükle"* seçeneğini kullanın.

## 3. Sistem Gereksinimleri

- **Windows Masaüstü**: Windows 10 veya 11 (`x86_64`), en az `1.5 GB` boş disk alanı (kurulu paket `~1.02 GiB`), `4 GB+` RAM (önerilen `8 GB+`).
- **Android**: `arm64-v8a` işlemcili Android cihaz, güncel Chrome / Android System WebView (WebGPU hızlandırması için güncel WebView gereklidir).

## 4. Ölçülen Performans ve Doğruluk (CSV Kayıtlarından)

- **Android Telefon (`results/device_bench_phone.csv`)**:
  - Cihaz: `SM-S948B` (`Android 17`, `Chrome 157` WebView, GPU: `qualcomm adreno-8xx`)
  - Backend: `ort-webgpu` (`20` zamanlanmış koşu + `1` ısınma)
  - Medyan model süresi: **`134.15 ms`** | Medyan toplam süre: **`220.55 ms`**
  - Kaynak notu: Uygulama içi hız testi arayüz tablosundan alınmıştır (ham JSON panodan çekilememiştir).
- **Dizüstü / Tarayıcı (`results/edge_benchmark.csv`, `results/web_parity.csv`, `results/web_spike.csv`, `results/web_video_fps.csv`)**:
  - Masaüstü yan sunucu (`OpenVINO INT8 (CPU)`, `Ryzen 7 8845HS`, `1024`): medyan model `36.7 ms` (`27.3 FPS`), uçtan uca `43.73 ms` (`22.9 FPS`), val `mAP50 = 0.943`, `mAP50-95 = 0.4569`.
  - Tarayıcı WebGPU (`ort-webgpu`, FP32 ONNX, `RTX 4060 Laptop GPU`, `web_parity.csv`): PyTorch FP32 referansıyla `conf ≥ 0.25` iki yönlü eşleşme **`%96.72`** (`ort. IoU = 0.9885`, `ort. |Δconf| = 0.0065`); `web_spike.csv` tekil medyan model `36.05 ms`; `web_video_fps.csv` canlı video `16.09 FPS` (model `36 ms`).
  - Tarayıcı WASM yedeği (`ort-wasm`, FP32 ONNX, `Ryzen 7 8845HS`): `web_spike.csv` tekil medyan **`689.41 ms`**; `web_video_fps.csv` canlı video **`0.38 FPS`** (model `2241.8 ms`).
- **FP32 vs INT8 Farkı (`results/quant_parity.csv`)**: `conf ≥ 0.395` noktasında `43` eşleşen, `2` yalnızca FP32, `5` yalnızca INT8 (`ort. IoU = 0.9530`, `ort. |Δconf| = 0.0160`). Farklar eşik sınırındaki Kart 05 şablon konumlarında yoğunlaşır.

## 5. Bilinen Sınırlılıklar

1. **Telefonda ısı haritası (LayerCAM) yalnız buluttur**: Katman 4 LayerCAM aktivasyon × pozitif gradyan geri yayılımı gerektirdiği için tarayıcı ve Android WebView içinde yerel olarak hesaplanmaz. Telefonda XAI ısı haritası isteği yalnızca çevrimiçi **Bulut** modunda (HF Space erişilebilir olduğunda) çalışır.
2. **WebGPU olmayan cihazlarda yavaştır**: WebGPU desteği sunmayan telefonlarda veya WebView sürümlerinde çıkarım tek iş parçacıklı `ort-wasm` yedeğine düşer (`web_spike.csv`: `689.41 ms`, `web_video_fps.csv`: `0.38 FPS`). Bu cihazlarda canlı kamera/video akışı yerine tekil fotoğraf tespiti önerilir.
3. **Tek cihazda ölçülmüş mobil hız**: `results/device_bench_phone.csv` yalnızca bir cihazda (`SM-S948B`, Android 17, `ort-webgpu`) ölçülmüştür; farklı telefonlarda GPU/WebView desteğine bağlı olarak daha yavaş veya farklı sonuçlar görülebilir. Kullanıcılar **Sonuçlar → Bu cihazda hız testi** ile kendi cihazlarını yerel olarak ölçebilir.
4. **Kamera modu deneyseldir**: Model PKU-Market-PCB veri setindeki yukarıdan çekilmiş kart görüntüleriyle eğitilmiştir; el kamerası perspektifi, yansıma veya odak bozukluğunda tespit doğruluğu düşebilir.
5. **Paket imzaları**: Masaüstü EXE imzasız, Android APK debug imzalıdır.
