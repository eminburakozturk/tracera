# Tracera — Açıklanabilirlik (XAI) Doğrulama ve Sadakat Özeti

- **Kilitlenen Varsayılan Konfigürasyon:** `LayerCAM @ Katman 4` (`--method layercam --layer 4`)
- **Kaynak Veri:** `results/gradcam_sanity.csv` (`model=g_y11s_1024_full`, `imgsz=1024`, `split=val`, `n_tespit=150`, `seed=42`)

## 1. Pointing Game Üçlüsü (Eğitilmiş / Rastgele Model / Şans) ve Yerel Pencere Metrikleri

| Katman | Yöntem | $n$ | Eğitilmiş Model Pointing (`kendi_kutu`) | Rastgele Model Pointing (`pointing_rastgele_model`) | Şans Tabanı (`şans_kendi_kutu`) | Pencere Alan Oranı | Pencere İçi Randomizasyon Spearman (Medyan) | Pencere İçi Enerji Kutuda | Taban (`Kutu ∩ Pencere / Pencere`) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **4** | **`layercam` (Varsayılan)** | 150 | 0.9067 (%90.67) | 0.6800 (%68.00) | 0.001061 (%0.11) | 1.0000 (%100.00) | 0.2054 | 0.6104 (%61.04) | 0.0011 (%0.11) |
| **4** | `hirescam` (Karşılaştırma) | 150 | 0.8800 (%88.00) | 0.6200 (%62.00) | 0.001061 (%0.11) | 1.0000 (%100.00) | 0.2280 | 0.3366 (%33.66) | 0.0011 (%0.11) |
| **4** | `gradcam` (Klasik CAM) | 150 | 0.0667 (%6.67) | 0.0067 (%0.67) | 0.001061 (%0.11) | 1.0000 (%100.00) | 0.0228 | 0.0052 (%0.52) | 0.0011 (%0.11) |
| **16** | `layercam` (Referans) | 150 | 1.0000 (%100.00) | 0.6800 (%68.00) | 0.001061 (%0.11) | 0.0021 (%0.21) | 0.0904 | 0.9289 (%92.89) | 0.5196 (%51.96) |

## 2. Mutlak Bütçeli ($1\times$ ve $2\times$ GT Kutu Alanı) Yapısal Kaydırma Kontrollü Silme Testi — Göreli Skor Düşüşü $(s_0 - s_1)/s_0$

| Katman | Yöntem | $1\times$ Harita Düşüş (Ort / Med) | $1\times$ Kontrol Düşüş (Ort / Med) | $1\times$ Kazanma Oranı | $2\times$ Harita Düşüş (Ort / Med) | $2\times$ Kontrol Düşüş (Ort / Med) | $2\times$ Kazanma Oranı |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **4** | **`layercam` (Varsayılan)** | 0.7004 / 0.9752 | 0.0023 / 0.0000 | **0.9733 (%97.33)** | 0.7654 / 0.9978 | 0.0172 / 0.0001 | **0.9867 (%98.67)** |
| **4** | `hirescam` (Karşılaştırma) | 0.7725 / 0.9858 | 0.0009 / 0.0000 | **0.9867 (%98.67)** | 0.8728 / 0.9999 | 0.0054 / 0.0000 | **0.9933 (%99.33)** |
| **4** | `gradcam` (Klasik CAM) | 0.0433 / 0.0007 | 0.0132 / 0.0001 | **0.6200 (%62.00)** | 0.0849 / 0.0016 | 0.0255 / 0.0004 | **0.6533 (%65.33)** |
| **16** | `layercam` (Referans) | 0.7623 / 1.0000 | 0.7538 / 0.9324 | **0.5867 (%58.67)** | 0.8750 / 1.0000 | 0.8344 / 0.9999 | **0.6067 (%60.67)** |

## 3. Bulgular ve Raporlama Kuralları Özeti (5 Cümle)

1. **Mutlak Bütçeli Yapısal Silme Sadakati (Kural a):** Geniş gradyan desteğine sahip backbone Katman 4'te (pencere alanı görüntünün %100.00'i), GT kutu alanının $1\times$ ve $2\times$ bütçeleriyle haritanın en yüksek pikselleri bulanıklaştırıldığında `layercam` göreli skor düşüşü ortalamada **0.7004** ($1\times$, medyan **0.9752**) ve **0.7654** ($2\times$, medyan **0.9978**) olurken, aynı maskenin pencere içinde rastgele kaydırıldığı yapısal kontrolde düşüş sıfıra yakın (**0.0023** ve **0.0172**) kalmakta; böylece sıfıra yakın payda nedeniyle yapay oranlar yerine doğrudan raporlanan kazanma oranı `layercam` için **%97.33** ($1\times$) / **%98.67** ($2\times$) ve `hirescam` için **%98.67** ($1\times$) / **%99.33** ($2\times$) düzeyine ulaşmaktadır.
2. **Varsayılan Yöntem Seçimi (`LayerCAM @ Katman 4`):** Katman 4 silme testinde `layercam` ve `hirescam` sadakat açısından çok yakın performans gösterse de (`1x` harita düşüş ortalaması **0.7004** vs. **0.7725**), `layercam` pozitif gradyan kapılaması sayesinde pencere içi harita enerjisinin **%61.04**'unu doğrudan GT kutusu içine odaklamakta (`hirescam`: **%33.66**, düzgün harita tabanı: **%0.11**) ve daha keskin yerelleşme sağladığı için varsayılan XAI yöntemi olarak kilitlenmiştir.
3. **Pointing Game Üçlüsü ve Mimari Katkı (Kural b):** Katman 4'te `layercam` kendi GT kutusuna pointing testinde **%90.67** (`hirescam`: **%88.00**) başarı sağlarken, rastgele ağırlıklı aynı mimarinin **%68.00** (`hirescam`: **%62.00**) ve şans tabanının **%0.11** çıkması, anchor merkezli etkin alıcı alanın (effective receptive field) mimari bir temel oluşturduğunu ancak öğrenilmiş ağırlıkların isabet oranını **1.33 katına** çıkardığını (pencere içi randomizasyon Spearman medyanı **0.2054**) dürüstçe göstermektedir.
4. **Katman 16 Referans Bulgusu ve Dar Alıcı Alan Kısıtı (Kural c):** Boyun çıkışındaki referans Katman 16'da tek bir anchor'ın gradyan desteği görüntünün yalnızca **%0.21**'ünü ($5\times 5$ hücre) kapsadığından, pencere içinde kaydırılan kontrol maskesi de GT kutusuyla örtüşerek yüksek skor düşüşüne ($1\times$ kontrol ort. **0.7538**, $2\times$ kontrol ort. **0.8344**) neden olmakta ve bu sığ alıcı alan yüzünden Katman 16 sadakat testi için uygun bulunmamaktadır.
5. **Klasik Grad-CAM Analizi (Kural d):** Katman 4'te klasik `gradcam`, gradyanların uzamsal ortalamasını alarak kanal ağırlığı ürettiği için çoklu kusur barındıran PCB kartlarında kendi kutusuna pointing oranında **%6.67** ile şans tabanının (**%0.11**) **62.9 katına** ulaşsa da (herhangi bir GT kutusuna pointing: **%40.00**), mutlak bütçeli silme testinde harita düşüşü $1\times$'te ort. **0.0433** (medyan **0.0007**) vs. kontrol düşüşü ort. **0.0132** (medyan **0.0001**, kazanma oranı **%62.00**) ve $2\times$'te harita düşüşü ort. **0.0849** (medyan **0.0016**) vs. kontrol düşüşü ort. **0.0255** (medyan **0.0004**, kazanma oranı **%65.33**) olarak ölçülmüştür; dolayısıyla yapısal kaydırma kontrolünün üzerinde kısmi sadakat gösterse de hem silme düşüşü/kazanma oranında (`layercam`: $1\times$ ort. **0.7004**, kazanma **%97.33**) hem de pencere içi kutu enerjisinde (**%0.52** vs. taban **%0.11**; `layercam`: **%61.04**) eleman bazlı yöntemlerin belirgin biçimde gerisinde kaldığından tespit bazlı açıklama için tercih edilmemiştir.
