# Tracera v1.1.0 Sürüm Notları

v1.1.0 arayüz sürümüdür (derleme: `main` @ `76cd832`): **model, eşik ve raporlanan metrikler v1.0.0 ile aynıdır** (model sürümü `1.0.0`, ONNX SHA-256 `aa8e1f42702c823426df778c9ff298293e8661010dce1d410ea89b6caccbb58c`, güven eşiği `0,395`). Değişen şey uygulamanın kendisi ve açıklama katmanıdır.

## 1. Sürüm Paketi İçeriği

| Dosya | Platform | Boyut | SHA-256 |
|---|---|---:|---|
| `Tracera_1.1.0_x64-setup.exe` | Windows (`x86_64`) | 316.24 MiB (331,600,758 bayt) | `2be6699d8854b3841c4b8abe56cab96ff5e6f7f0edac896b35e611802da39634` |
| `Tracera_1.1.0_android_arm64.apk` | Android (`arm64-v8a`) | 59.29 MiB (62,170,347 bayt) | `9b5b2f9be721c60fc34be61f77244ff2aacb9ec340113494079279b01070f469` |
| `SHA256SUMS.txt` | Tümü | 194 bayt | — |

```powershell
Get-FileHash .\Tracera_1.1.0_x64-setup.exe, .\Tracera_1.1.0_android_arm64.apk -Algorithm SHA256
```

## 2. Yenilikler

**Yeni arayüz (web, Windows, Android aynı kod):** daraltılabilir kenar çubuğu (masaüstü), alt sekme çubuğu (mobil), koyu ve açık tema (iki temada da WCAG AA kontrastı, `axe` ile denetlendi), hareketi azaltma tercihine uyum, Türkçe ve İngilizce.

**Ekranlar:**
- **Ana Sayfa:** sürükle-bırak, doğrulama kartı örnekleri (yalnız val kartları 05 ve 11), CV / kilitli test / model süresi kartları (`metrics.json`'dan), son muayeneler.
- **Muayene:** GEÇTİ/KALDI bandı (yazılım kuralı, standart değil), eşik kaydırıcısı ve val verisinden Kesinlik–Duyarlılık mini grafiği, sınıf filtreleri, kusur kartları.
- **Açıklama:** LayerCAM (Katman 4) ısı haritası ve **"Bu karar nasıl verildi?"** kartı: güven ve eşik farkı, aynı güven aralığında **val ve test** doğru tespit oranı (n ile), ısı haritası enerjisinin kutu içindeki payı.
- **Muayene raporu (PDF):** cihazda üretilir, Türkçe karakterler gömülü yazı tipiyle; altbilgi: *"Bu rapor Tracera yazılımının otomatik çıktısıdır; herhangi bir standarda uygunluk belgesi değildir."*
- **Toplu muayene:** çoklu dosya veya klasör, sıralı kuyruk, GEÇTİ/KALDI özeti, sınıf dağılımı grafiği, CSV ve birleşik PDF.
- **Geçmiş:** cihazda (IndexedDB) saklanır; açma, silme, iki muayeneyi yan yana karşılaştırma.
- **Kusur Rehberi:** 6 sınıf için val kartlarından gerçek kırpımlar ve sınıf bazında val/test metrikleri.
- **Sonuçlar:** sızıntı, çözünürlük, hata analizi ve platform hızları grafiklerle.
- **Geliştirici, Ayarlar, ilk açılış tanıtımı.**

**"Düşük güven: gözden geçirin" etiketi:** güveni 0,6'nın altındaki tespitler işaretlenir. Kilitli test tahminlerinde 0,395–0,6 aralığındaki tespitlerin doğru oranı düşük çıktı (0,395–0,5: %12,8, n=78; 0,5–0,6: %14,7, n=68; büyük kısmı kart 06'daki 2 şablon noktası). Etiket **değerlendirmeden sonra eklenmiş bir kullanılabilirlik özelliğidir**; eşik, GEÇTİ/KALDI kuralı ve raporlanan metrikler değişmedi.

**API (CONTRACT v1.3, geriye uyumlu):** `/explain` ve `/explain_api` yanıtına `energy_in_box` (0–1), `peak_in_box`, `peak_xy` eklendi. Isı haritasının kendisi değişmedi; v1.0.0 istemcileri etkilenmez.

**Altyapı:** CI artık pytest başarısız olunca gerçekten kırmızı oluyor (v1.0.0'daki CI yanlış yeşil veriyordu). Linux ile Windows arasındaki OpenVINO INT8 farkı her CI koşusunda ölçülüp raporlanıyor. Sürüm tek komutla (`scripts/release_build.ps1`) derleniyor: yerel API her seferinde yeniden derlenip canlı test ediliyor ve çalışma ağacında değişiklik varken derleme yapılmıyor.

## 3. Ölçümler (değişmedi; kaynak: `results/*.csv`)

v1.0.0 sürüm notlarındaki tüm hız ve doğruluk sayıları geçerlidir. v1.1.0'a özgü yeni ölçüm:

- **Platform farkı (CI #2, GitHub Linux sunucusu vs Windows, OpenVINO INT8, 9 val görüntüsü):** ortalama |Δgüven| 0,020, en büyük 0,136, ortalama IoU 0,982; 0,395 eşiğinde tespit sayısı 42 → 45. Eşiğe yakın tespitlerde karar platforma göre değişebilir. Olası açıklama (doğrulanmadı): INT8 çekirdeklerinin işlemci komut setine göre farklı çalışması.

## 4. Kurulum Uyarıları

v1.0.0 ile aynı: Windows kurulum dosyası **imzasızdır** (SmartScreen → "Ek bilgi" → "Yine de çalıştır"); Android APK **debug anahtarıyla imzalıdır** ("Bilinmeyen kaynaklardan yükleme" izni gerekir). v1.0.0 APK'sı aynı bilgisayarda derlendiyse üzerine güncellenir; farklı imzalıysa önce kaldırın.

## 5. Bilinen Sınırlılıklar

1. Telefonda ısı haritası yalnız bulut modunda (HF Space) üretilir.
2. WebGPU olmayan cihazlarda çıkarım yavaştır (tek iş parçacıklı WASM).
3. Mobil hız tek cihazda ölçülmüştür (SM-S948B).
4. Test seti 2 karttan oluşur; güven aralığı oranları genel bir kalibrasyon iddiası değildir.
5. Kamera modu deneyseldir (model yukarıdan çekilmiş çıplak kart görüntüleriyle eğitildi).
