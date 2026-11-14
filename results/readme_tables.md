# Tracera Projesi — Doğrulanmış Deney Tabloları

> [!NOTE]
> Bu dosya `scripts/make_readme_tables.py` tarafından `results/*.csv` dosyalarından tam otomatik olarak üretilmiştir.
> README.md dosyasındaki hiçbir sayı elle yazılmamış; doğrudan buradaki çıktılardan aktarılmıştır.
> Tüm standart sapmalar örneklem standart sapmasıdır (`ddof=1`).

### Tablo 1: Veri Sızıntısı Kanıtı — Rastgele vs. Kart Bazlı (Grouped) 5-Katlı Çapraz Doğrulama

| Bölme Şeması | Kat | Doğrulama Kartları | Eğitim / Val Görüntü | mAP50 | mAP50-95 | missing_hole | mouse_bite | open_circuit | short | spur | spurious_copper |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Rastgele (Random Split) | Kat 1 | 01;04;05;06;07;08;09;10;11;12 | 498 / 125 | 0.9416 | 0.5022 | 0.5870 | 0.4556 | 0.5005 | 0.5192 | 0.4492 | 0.5015 |
| Rastgele (Random Split) | Kat 2 | 01;04;05;06;07;08;09;10;11;12 | 498 / 125 | 0.9507 | 0.5088 | 0.5928 | 0.4792 | 0.4567 | 0.5609 | 0.4934 | 0.4695 |
| Rastgele (Random Split) | Kat 3 | 01;04;05;06;07;08;09;10;11;12 | 498 / 125 | 0.9414 | 0.4931 | 0.5493 | 0.5284 | 0.4443 | 0.4794 | 0.4609 | 0.4960 |
| Rastgele (Random Split) | Kat 4 | 01;04;05;06;07;08;09;10;11;12 | 499 / 124 | 0.9185 | 0.4889 | 0.6418 | 0.4503 | 0.4996 | 0.4349 | 0.4428 | 0.4638 |
| Rastgele (Random Split) | Kat 5 | 01;04;05;06;07;08;09;10;11;12 | 499 / 124 | 0.9300 | 0.5009 | 0.6193 | 0.4895 | 0.4850 | 0.4963 | 0.4124 | 0.5029 |
| **Rastgele (Random Split) Ortalama** | **Özet** | **10 Kart (Tümü)** | **623 Toplam** | **0.9364 ± 0.0124** | **0.4988 ± 0.0078** | 0.5980 | 0.4806 | 0.4772 | 0.4981 | 0.4517 | 0.4867 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Kart Bazlı (Grouped Split) | Kat 1 | 01 | 509 / 114 | 0.5005 | 0.2001 | 0.4456 | 0.1000 | 0.2528 | 0.2163 | 0.0308 | 0.1550 |
| Kart Bazlı (Grouped Split) | Kat 2 | 04;11 | 466 / 157 | 0.8805 | 0.4182 | 0.5331 | 0.4960 | 0.4409 | 0.2381 | 0.3940 | 0.4071 |
| Kart Bazlı (Grouped Split) | Kat 3 | 05;12 | 513 / 110 | 0.8525 | 0.4363 | 0.5571 | 0.4130 | 0.4666 | 0.4960 | 0.4751 | 0.2100 |
| Kart Bazlı (Grouped Split) | Kat 4 | 08;09 | 514 / 109 | 0.8998 | 0.4437 | 0.4765 | 0.3511 | 0.4568 | 0.4380 | 0.4384 | 0.5016 |
| Kart Bazlı (Grouped Split) | Kat 5 | 06;07;10 | 490 / 133 | 0.8974 | 0.4801 | 0.5844 | 0.4613 | 0.4746 | 0.4532 | 0.4475 | 0.4593 |
| **Kart Bazlı (Grouped Split) Ortalama** | **Özet** | **10 Kart (Tümü)** | **623 Toplam** | **0.8061 ± 0.1719** | **0.3957 ± 0.1116** | 0.5193 | 0.3643 | 0.4183 | 0.3683 | 0.3572 | 0.3466 |
|---|---|---|---|---|---|---|---|---|---|---|---|

> **Önemli Bulgular:**
> - Rastgele bölmede ortalama mAP50 **0.9364** iken, aynı görüntüler kart bazında ayrıldığında **0.8061** seviyesine düşmektedir (**-0.1303** puan / -13.0 puan düşüş).
> - Kat 1'de (yalnız kart 01) model mAP50 = 0.5005 seviyesinde kalmaktadır. Bu fark, rastgele bölmenin kartın arka plan dokusunu ezberleyerek aşırı iyimser (sızıntılı) sonuç ürettiğini deneysel olarak kanıtlar.

### Tablo 2: Giriş Çözünürlüğü ve Ağırlık Seçimi (5-Katlı Kart Bazlı CV, 80 Epoch, Patience 0)

| Çözünürlük | Ağırlık Kuralı | Kat 1 | Kat 2 | Kat 3 | Kat 4 | Kat 5 | Ortalama ± std (mAP50) | Ortalama ± std (mAP50-95) | Ortalama Süre (dk) |
|---|---|---|---|---|---|---|---|---|---|
| **640x640** | last.pt (Önceden Kilitli) | 0.5155 | 0.8402 | 0.8565 | 0.8192 | 0.9031 | **0.7869 ± 0.1548** | 0.3760 ± 0.1080 | 11.0 dk |
| **640x640** | best.pt (Bilgi Amaçlı Val Seçimi) | 0.5069 | 0.8716 | 0.8921 | 0.9258 | 0.9031 | **0.8199 ± 0.1761** | 0.4040 ± 0.1080 | 11.0 dk |
| **1024x1024** | last.pt (Önceden Kilitli) | 0.5885 | 0.9080 | 0.8952 | 0.9388 | 0.9569 | **0.8575 ± 0.1523** | 0.4240 ± 0.1188 | 24.8 dk |
| **1024x1024** | best.pt (Bilgi Amaçlı Val Seçimi) | 0.6442 | 0.9248 | 0.9295 | 0.9637 | 0.9546 | **0.8834 ± 0.1347** | 0.4508 ± 0.0938 | 24.8 dk |

> **Karar Gerekçesi:**
> - 1024 çözünürlük, 5 katın 5'inde de 640'ın önündedir (eşleştirilmiş kat farkları: Kat 1: +0.0730, Kat 2: +0.0678, Kat 3: +0.0387, Kat 4: +0.1196, Kat 5: +0.0538; ortalama mAP50 farkı: **+0.0706**; mAP50-95 farkı: **+0.0480**).
> - Büyük std (±0.152), zor olan Kat 1'den (sadece kart 01) kaynaklanmaktadır; katlar arası fark tutarlıdır.
> - CV protokolüyle tutarlılık açısından önceden kilitlenen **last.pt** temel ağırlık olarak seçilmiştir (val'e göre erken durdurma veya model seçimi yapılmamıştır).

### Tablo 3: Kilitli Nihai Test Seti Performansı (yolo11s @ 1024, last.pt, Kart 06 + 10)

- **Test Kümesi Boyutu:** 93 görüntü, 469 kusur kutusu
- **Genel Sonuçlar:** Precision = **0.9462**, Recall = **0.8577**, mAP50 = **0.9225**, mAP50-95 = **0.4740**

| Kusur Sınıfı | Precision (Oracle Eşik: 0.569) | Recall (Oracle Eşik: 0.569) | mAP50 | mAP50-95 |
|---|---|---|---|---|
| **missing_hole** | 0.9833 | 0.9605 | 0.9705 | 0.4791 |
| **mouse_bite** | 0.9194 | 0.9103 | 0.9442 | 0.4651 |
| **open_circuit** | 0.9442 | 0.8375 | 0.9207 | 0.5158 |
| **short** | 0.9479 | 0.6828 | 0.8903 | 0.4451 |
| **spur** | 0.9226 | 0.8649 | 0.8527 | 0.4259 |
| **spurious_copper** | 0.9601 | 0.8905 | 0.9564 | 0.5128 |

> **Metodolojik Not:**
> - Bu tablodaki mAP değerleri eşikten bağımsızdır ve test setinde **tek bir kez** hesaplanıp kilitlenmiştir.
> - Tablodaki P (0.9462) ve R (0.8577) değerleri, Ultralytics'in test kümesi F1 eğrisinin tepe noktasından (conf = 0.569) seçtiği **oracle eşik (üst sınır)** değerleridir; işletme çalışma noktası olarak val'de seçilen sabit eşik (0.395) kullanılır (bkz. Tablo 4).

### Tablo 4: Sabit Çalışma Noktası Doğrulaması ve Hata Analizi (Sabit Eşik = 0.395, IoU = 0.50)

| Veri Kümesi Spliti | Güven Eşiği | TP | Toplam FP | Arka Plan FP (IoU<0.1) | Konum Hatası FP | Çift Tespit FP | Yanlış Sınıf FP | FN | Precision | Recall | F1 Skoru |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Doğrulama (Val: Kart 05, 11) | 0.395 | 514 | 16 | 7 | 7 | 2 | 0 | 48 | 0.9698 | 0.9146 | **0.9414** |
| Test (Kilitli: Kart 06, 10) | 0.395 | 416 | 142 | 125 | 8 | 4 | 5 | 53 | 0.7455 | 0.8870 | **0.8101** |
| **Test (Duyarlılık: 2 Şablon Noktası Hariç)** | 0.395 | 416 | 28 | 11 | 8 | 4 | 5 | 53 | 0.9369 | 0.8870 | **0.9113** |

> **Hata Analizi ve Teşhis Özeti:**
> - Val setinde F1'i maksimize eden eşik (0.395) test setine doğrudan uygulandığında: Precision = **0.7455**, Recall = **0.8870**, F1 = **0.8101** elde edilir. Kilitli sonuç budur.
> - Test setindeki 142 FP'nin 138'i Kart 06'da toplanmaktadır (Kart 10'da sadece 4 FP vardır).
> - Kart 06'daki 114 FP, şablon fotoğrafındaki **2 sabit koordinatta** (mouse_bite: 1001, 1380 ve spur: 1708, 1050) hemen her görüntüde tekrarlanan yol geometrisi özellikleridir.
> - Bu 2 sabit nokta tekil sayıldığında / çıkarıldığında test setinde kalan FP sayısı 28'e düşer; Precision = **0.9369**, F1 = **0.9113** seviyesine ulaşır.

### Tablo 5: Temiz / Hatasız Şablon Kontrolü (PCB_USED/05, 06, 10, 11 @ conf=0.395)

| Kart No | Şablon Dosyası | Yanlış Alarm Sınıfı | Model Güveni | Koordinatlar (cx, cy) | Kutu [x1, y1, x2, y2] | Teşhis |
|---|---|---|---|---|---|---|
| Kart 5 | `05.JPG` | spurious_copper | 0.5108 | (1606.5, 1072.9) | [1550.5, 1015.6, 1662.5, 1130.2] | Arka Plan Dokusu Yanlış Alarmı |
| Kart 5 | `05.JPG` | spurious_copper | 0.4475 | (1537.1, 1183.2) | [1506.1, 1147.5, 1568.1, 1219.0] | Arka Plan Dokusu Yanlış Alarmı |
| Kart 6 | `06.JPG` | mouse_bite | 0.5569 | (1000.6, 1380.1) | [983.2, 1359.3, 1018.0, 1400.9] | Arka Plan Dokusu Yanlış Alarmı |

> **Bulgular:**
> - 4 temiz şablon görüntüsünde toplam 3 yanlış alarm üretilmiştir (Kart 10 ve Kart 11'de sıfır yanlış alarm).
> - Kart 06 temiz şablonunda tespit edilen mouse_bite (güven: 0.5569, cx=1000.6, cy=1380.1), test görüntülerindeki 57 mouse_bite FP'sinin konumuyla tam olarak örtüşmektedir. Bu tespit, hatanın sentetik bozulmadan değil, kartın kendi tasarım şablonundan kaynaklandığını **kesin** olarak doğrular.

### Güven aralığına göre kesinlik (val ve test)

| aralık | val n | val kesinlik | test n | test TP | test FP | test kesinlik |
|---|---:|---:|---:|---:|---:|---:|
| 0.395–0.5 | 4 | 1.0000 (%100.0) | 78 | 10 | 68 | 0.1282 (%12.8) |
| 0.5–0.6 | 13 | 0.8462 (%84.6) | 68 | 10 | 58 | 0.1471 (%14.7) |
| 0.6–0.7 | 20 | 0.9000 (%90.0) | 41 | 32 | 9 | 0.7805 (%78.0) |
| 0.7–0.8 | 253 | 0.9763 (%97.6) | 211 | 206 | 5 | 0.9763 (%97.6) |
| 0.8–0.9 | 240 | 0.9750 (%97.5) | 160 | 158 | 2 | 0.9875 (%98.8) |
| 0.9–1.0 | 0 | — | 0 | — | — | — |
| **Toplam (≥ 0.395)** | **530** | **0.9698 (%97.0)** | **558** | **416** | **142** | **0.7455 (%74.6)** |

Güven eşiği (`conf = 0.395`) doğrulama (`val`) kümesinde seçilmiş ve sabit tutulmuştur; test sonuçlarına bakılarak eşik, GEÇTİ/KALDI kuralı veya raporlanan metrikler değiştirilmemiştir. Testteki düşük güven bandı (`0.395–0.6`) yanlış pozitiflerinin büyük bölümü (`142` FP'nin `126`'sı `< 0.6` aralığındadır) Kart 06'daki 2 sabit şablon noktasından gelmektedir. Test kümesi yalnızca 2 karttan (`06` ve `10`) oluştuğu için genel bir kalibrasyon iddiası yapılmaz (arayüzde `0.395–0.6` aralığına eklenen *"Düşük güven: gözden geçirin"* etiketi test sonrası eklenmiş bir kullanılabilirlik özelliğidir, metrik iddiası değildir).

### Tablo 6a: XAI Pointing Game, Rastgelelik ve Enerji Oranı (yolo11s @ 1024, n=150 Doğru Tespit)

| Hedef Katman | Yöntem | Kendi Kutu Pointing (Eğitilmiş) | Pointing (Rastgele Model) | Rastgele Şans Oranı | Pencere Alan Oranı | Kutu İçi Enerji (Pencere İçi) | Taban Kutu Alan Oranı |
|---|---|---|---|---|---|---|---|
| Katman 4 | gradcam | 6.7% | 0.7% | 0.106% | 100.0% | 0.5% | 0.110% |
| Katman 4 | hirescam | 88.0% | 62.0% | 0.106% | 100.0% | 33.7% | 0.110% |
| Katman 4 | layercam | 90.7% | 68.0% | 0.106% | 100.0% | 61.0% | 0.110% |
| Katman 16 | layercam | 100.0% | 68.0% | 0.106% | 0.2% | 92.9% | 51.960% |

### Tablo 6b: XAI Mutlak Bütçeli Yapısal Silme Testi (Deletion Fidelity — Asıl Sadakat Ölçüsü)

| Hedef Katman | Yöntem | 1x GT Alanı Harita Göreli Düşüş (Ort / Med) | 1x Kontrol Düşüşü (Ort / Med) | 1x Kazanma Oranı | 2x GT Alanı Harita Göreli Düşüş (Ort / Med) | 2x Kontrol Düşüşü (Ort / Med) | 2x Kazanma Oranı |
|---|---|---|---|---|---|---|---|
| Katman 4 | gradcam | 0.0433 / 0.0007 | 0.0132 / 0.0001 | **62.0%** | 0.0849 / 0.0016 | 0.0255 / 0.0004 | **65.3%** |
| Katman 4 | hirescam | 0.7725 / 0.9858 | 0.0009 / 0.0000 | **98.7%** | 0.8728 / 0.9999 | 0.0054 / 0.0000 | **99.3%** |
| Katman 4 | layercam | 0.7004 / 0.9752 | 0.0023 / 0.0000 | **97.3%** | 0.7654 / 0.9978 | 0.0172 / 0.0001 | **98.7%** |
| Katman 16 | layercam | 0.7623 / 1.0000 | 0.7538 / 0.9324 | **58.7%** | 0.8750 / 1.0000 | 0.8344 / 0.9999 | **60.7%** |

> **Açıklanabilirlik ve Yöntem Kararı (Neden LayerCAM @ Katman 4?):**
> 1. **Klasik Grad-CAM Başarısızlığı:** Grad-CAM kanal ağırlıklarını uzamsal ortalama ile hesapladığından, YOLO gibi tek evrişim katmanında birden fazla nesnenin kodlandığı mimarilerde sinyali seyreltir. 1x silme testinde bağıl skor düşüşü yalnızca 0.0433 seviyesinde kalmakta ve kazanma oranı %62'ye düşmektedir.
> 2. **HiResCAM vs. LayerCAM:** HiResCAM silme testinde yüksek sadakat gösterse de (düşüş 0.7725, kazanma %98.7), aktivasyon enerjisinin sadece %33.7'si GT kutusuna odaklanabilmektedir. LayerCAM ise gradyanları pozitif yönde süzerek (%61.0 kutu içi enerji) çok daha net yerelleşme sunar; silme düşüşü 0.7004 ve kazanma oranı %97.3'tür.
> 3. **Katman Seçimi (Katman 4 vs. Katman 16):** Katman 16 (Detect başlığı girdisi) receptive field olarak tüm görüntüyü kapsar ancak uzamsal çözünürlüğü kabadır; silme kontrolünde bile düşüş yüksektir (kaydırılmış maske hâlâ nesneye etki eder). Katman 4 (backbone stride 8) mikron düzeyindeki PCB hatalarında doğru uzamsal hassasiyeti veren tek katmandır.
> 4. **Model Randomizasyon Testi (Adebayo):** Rastgele ağırlıklı modelde pointing oranının %68 çıkması, pointing oyununun eğitilmiş ağırlıklardan ziyade evrişimli mimarinin türevlenebilir uzamsal yapısını ölçtüğünü göstermiştir. Bu nedenle birincil sadakat kanıtı olarak silme testi kabul edilmiştir.

### Tablo 7: Çıkarım Formatı ve Donanım Kıyaslaması (Val Seti, Laptop Ortamı)

*Donanım Ortamı: AMD Ryzen 7 8845HS CPU (8 thread) & NVIDIA GeForce RTX 4060 Laptop GPU (8 GB, TGP 140W)*

| Model & Çözünürlük | Format / Çalışma Zamanı | Donanım | Model Boyutu (MB) | Model Çıkarım Gecikmesi (Ort / Med ms) | Uçtan Uca Gecikme (Ort / Med ms) | FPS (Uçtan Uca) | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|---|---|
| `g_y11s_640_full` | PyTorch FP32 (GPU) | GPU (RTX 4060 Laptop) | 18.3 MB | 8.89 / 8.78 ms | 12.10 / 11.97 ms | **83.5** | 0.9005 | 0.4214 |
| `g_y11s_640_full` | ONNX Runtime (CPU) | CPU (Ryzen 7 8845HS) | 36.2 MB | 54.74 / 54.68 ms | 58.49 / 58.44 ms | **17.1** | 0.8995 | 0.4173 |
| `g_y11s_640_full` | OpenVINO FP32 (CPU) | CPU (Ryzen 7 8845HS) | 36.3 MB | 37.84 / 36.48 ms | 41.15 / 39.87 ms | **25.1** | 0.8995 | 0.4173 |
| `g_y11s_640_full` | OpenVINO INT8 (CPU) | CPU (Ryzen 7 8845HS) | 9.8 MB | 14.08 / 13.60 ms | 17.40 / 16.83 ms | **59.4** | 0.8975 | 0.4164 |
| `g_y11s_1024_full` | PyTorch FP32 (GPU) | GPU (RTX 4060 Laptop) | 18.4 MB | 9.74 / 9.80 ms | 15.02 / 14.83 ms | **67.4** | 0.9396 | 0.4434 |
| `g_y11s_1024_full` | ONNX Runtime (CPU) | CPU (Ryzen 7 8845HS) | 36.4 MB | 157.45 / 157.18 ms | 165.31 / 165.00 ms | **6.1** | 0.9372 | 0.4378 |
| `g_y11s_1024_full` | OpenVINO FP32 (CPU) | CPU (Ryzen 7 8845HS) | 36.5 MB | 105.18 / 103.06 ms | 111.68 / 109.70 ms | **9.1** | 0.9372 | 0.4378 |
| `g_y11s_1024_full` | OpenVINO INT8 (CPU) | CPU (Ryzen 7 8845HS) | 10.0 MB | 37.32 / 36.70 ms | 43.89 / 43.73 ms | **22.9** | 0.9430 | 0.4569 |
| `g_y11s_640_full` | TensorRT FP16 (GPU) | GPU (RTX 4060 Laptop) | 21.5 MB | 1.86 / 1.86 ms | 5.18 / 5.18 ms | **193.0** | 0.9003 | 0.4155 |
| `g_y11s_1024_full` | TensorRT FP16 (GPU) | GPU (RTX 4060 Laptop) | 22.1 MB | 3.36 / 3.34 ms | 12.19 / 12.29 ms | **81.4** | 0.9372 | 0.4382 |

> **Bulgular ve Yorum:**
> - **OpenVINO INT8 Başarımı (CPU):** 1024x1024 çözünürlükte OpenVINO INT8 kuantizasyonu, CPU üzerinde model çıkarımını 105.2 ms'den **37.3 ms**'ye indirmiş (~2.8x hızlanma), boyutu 36.5 MB'tan **9.97 MB**'a düşürmüştür. mAP50 kaybı yaşanmamıştır (120 görüntülük val setinde +0.6 puanlık değişim ölçüm gürültüsüdür; 'kayıp yok' şeklinde yorumlanır).
> - **TensorRT FP16 (GPU):** Model çıkarımını 1024'te **3.34 ms**'ye indirmiştir. Ancak uçtan uca sürede (12.19 ms) darboğaz modelden ziyade PyTorch/CPU ön ve son işleme (letterbox + NMS) kaymaktadır.
> - **Kapsam:** Tüm değerler dizüstü bilgisayar donanımında ölçülmüştür; Jetson veya gömülü kenar cihaz iddiası yapılmamaktadır.

### Platform Kayması (Linux CI #2 vs. Windows OpenVINO INT8 Referansı)

GitHub Actions Linux runner (`CI #2`, `33037a1`) üzerinde OpenVINO INT8 çıkarımı ile Windows referansı karşılaştırılmıştır (Windows'ta aynı makinedeki yerel kıyas farkı `0.000`):

| Kapsam | Ort. \|Δconf\| | Maks. \|Δconf\| | Ort. IoU | Sayım (`conf ≥ 0.395`) | Not |
|---|---:|---:|---:|---:|---|
| GitHub runner (`9/10` val görüntüsü, `conf = 0.25` iki yönlü eşleştirme) | **0.020** | **0.136** | **0.982** | **42 → 45 (+3)** | Eşiğe yakın kutularda GEÇTİ/KALDI kararını değiştirebilir |
| Tek görüntü testi (`val_05_missing_hole_01.jpg` maks. güven) | **0.0034** | **0.0034** | — | 3 → 3 (0) | Linux `0.827784` vs. Windows `0.831160` |

> Olası açıklama (doğrulanmadı): INT8 çekirdeklerinin CPU komut setine göre farklı çalışması.

### Tablo 8: Endüstriyel PCB-AoI Veri Kümesine Transfer Öğrenme Deneyi (Test Seti, 3 Seed, last.pt)

| Başlangıç Modeli | Seed | Eğitim Süresi (dk) | Test mAP50 | Test mAP50-95 | Bad_qiaojiao mAP50 | Bad_podu mAP50 |
|---|---|---|---|---|---|---|
| COCO (yolo11s.pt) | Seed 0 | 5.97 dk | 0.6035 | 0.2082 | 0.3750 | 0.8321 |
| COCO (yolo11s.pt) | Seed 1 | 15.62 dk | 0.6107 | 0.2133 | 0.3793 | 0.8422 |
| COCO (yolo11s.pt) | Seed 2 | 5.27 dk | 0.6108 | 0.2099 | 0.4193 | 0.8023 |
| PKU (g_y11s_1024_full) | Seed 0 | 5.11 dk | 0.6283 | 0.2212 | 0.4134 | 0.8433 |
| PKU (g_y11s_1024_full) | Seed 1 | 5.19 dk | 0.6402 | 0.2267 | 0.4531 | 0.8273 |
| PKU (g_y11s_1024_full) | Seed 2 | 5.80 dk | 0.5831 | 0.2084 | 0.3425 | 0.8237 |
| **COCO (yolo11s.pt) Ortalama** | **Özet (ddof=1)** | **8.95 dk** | **0.6083 ± 0.0042** | **0.2105 ± 0.0026** | **0.3912 ± 0.0244** | **0.8255 ± 0.0207** |
| **PKU (g_y11s_1024_full) Ortalama** | **Özet (ddof=1)** | **5.37 dk** | **0.6172 ± 0.0301** | **0.2188 ± 0.0094** | **0.4030 ± 0.0560** | **0.8314 ± 0.0104** |

> **Bilimsel Sonuç (Null Sonuç):**
> - PKU ön eğitimli modelin test mAP50 ortalaması **0.6172 ± 0.0301**, standart COCO başlangıcının ise **0.6083 ± 0.0042** olmuştur.
> - Eşleştirilmiş fark: Seed 0 (+0.0248), Seed 1 (+0.0295), Seed 2 (-0.0277); ortalama fark **+0.0089 ± 0.0318** seviyesindedir.
> - **Yorum:** PKU ön eğitiminin gerçek endüstriyel AoI verisi üzerinde istatistiksel olarak anlamlı bir faydası gösterilememiştir (fark seed gürültüsünün içindedir, n=3). Bu deney bir **null sonuç** olarak raporlanır; sentetik PKU temsillerinin farklı optik ve kusur dağılımına sahip gerçek endüstriyel ortama doğrudan genellenmediğini gösterir.
