# Kararlar

Format: `- YYYY-MM-DD [ajan] karar — gerekçe`

- 2026-11-11 [claude] Ana veri PKU-Market-PCB (693 görüntü). Risk: aynı kartlar train ve test setinde olabilir -> rastgele bölme mAP'ı şişirebilir. Çözüm: kart numarasına göre gruplu bölme ile karşılaştır, DeepPCB ile cross-dataset test.
- 2026-11-11 [claude] DeepPCB plandan çıkarıldı (o da sentetik). Opsiyonel ikinci veri: PCB-AoI; uygunsa gerçek veri fine-tune deneyi.
- 2026-11-11 [claude] PKU: `PCB_DATASET/rotation` klasörü KULLANILMAZ (aynı görüntülerin döndürülmüş kopyaları -> sızıntı). Sadece `images/` + `Annotations/` (VOC XML).
- 2026-11-11 [claude] PCB-AoI: VOC XML, gerçek SPI-AOI görüntüleri (2017 tarihli). Sınıflar: Bad_qiaojiao (kalkık bacak), Bad_podu. `ChaojieZhu` etiketi sınıf değil (muhtemelen etiketleyici adı) -> atlanır. Sadece `train_data` (173) + `test_data` (60) kullanılır; `train_data_augmentation` KULLANILMAZ (aynı kartlardan türetilmiş).
- 2026-11-11 [claude] PKU veri incelemesi: 693 görüntü/693 XML eşleşiyor, etiket-klasör uyuşmazlığı yok, sınıflar dengeli (482–503 kutu). Her görüntüde tek hata tipi, 1–6 kutu. 10 kart: 01 ve 04'te 120'şer, 10'da 32, diğerlerinde ~60 görüntü. Görüntü boyutu karta göre sabit (~2240–3056 px).
- 2026-11-11 [claude] Rastgele bölme hata tipine (klasöre) göre stratified yapılır — her split'te 6 sınıf dengeli olsun.
- 2026-11-11 [claude] Kutular küçük: medyan ~64 px; 640'a küçültünce medyan 13–24 px, en küçük ~5 px. 640 sadece baseline; tiling/yüksek imgsz asıl deney. Tiling YAPILACAKSA bölmeden SONRA, her split içinde yapılır (aynı görüntünün parçaları farklı split'e düşmesin).
