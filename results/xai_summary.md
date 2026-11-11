# Tracera — Açıklanabilirlik (XAI) Doğrulama ve Sadakat Özeti

- **Kilitlenen Varsayılan Konfigürasyon:** `LayerCAM @ Katman 4` (`--method layercam --layer 4`)
- **Kaynak Veri:** `results/gradcam_sanity.csv` (`model=g_y11s_640`, `imgsz=640`, `split=val`, `n_tespit=150`, `seed=42`)

## 1. Pointing Game Üçlüsü (Eğitilmiş / Rastgele Model / Şans) ve Yerel Pencere Metrikleri

| Katman | Yöntem | $n$ | Eğitilmiş Model Pointing (`kendi_kutu`) | Rastgele Model Pointing (`pointing_rastgele_model`) | Şans Tabanı (`şans_kendi_kutu`) | Pencere Alan Oranı | Pencere İçi Randomizasyon Spearman (Medyan) | Pencere İçi Enerji Kutuda | Taban (`Kutu ∩ Pencere / Pencere`) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **4** | **`layercam` (Varsayılan)** | 150 | 0.8600 (%86.00) | 0.4333 (%43.33) | 0.001061 (%0.11) | 0.9991 (%99.91) | 0.1209 | 0.6419 (%64.19) | 0.0011 (%0.11) |
| **4** | `hirescam` | 150 | 0.8000 (%80.00) | 0.3933 (%39.33) | 0.001061 (%0.11) | 0.9991 (%99.91) | 0.1933 | 0.3973 (%39.73) | 0.0011 (%0.11) |
| **4** | `gradcam` | 150 | 0.0400 (%4.00) | 0.0000 (%0.00) | 0.001061 (%0.11) | 0.9991 (%99.91) | 0.0561 | 0.0049 (%0.49) | 0.0011 (%0.11) |
| **16** | `layercam` | 150 | 0.9933 (%99.33) | 0.4133 (%41.33) | 0.001061 (%0.11) | 0.0043 (%0.43) | 0.0811 | 0.7714 (%77.14) | 0.2426 (%24.26) |
| **16** | `hirescam` | 150 | 0.9133 (%91.33) | 0.3533 (%35.33) | 0.001061 (%0.11) | 0.0043 (%0.43) | 0.0581 | 0.6592 (%65.92) | 0.2426 (%24.26) |
| **16** | `gradcam` | 150 | 0.0667 (%6.67) | 0.0000 (%0.00) | 0.001061 (%0.11) | 0.0043 (%0.43) | 0.0000 | 0.4532 (%45.32) | 0.2426 (%24.26) |

## 2. Mutlak Bütçeli ($1\times$ ve $2\times$ GT Kutu Alanı) Yapısal Kaydırma Kontrollü Silme Testi — Göreli Skor Düşüşü $(s_0 - s_1)/s_0$

| Katman | Yöntem | $1\times$ Harita Düşüş (Ort / Med) | $1\times$ Kontrol Düşüş (Ort / Med) | $1\times$ Kazanma Oranı | $2\times$ Harita Düşüş (Ort / Med) | $2\times$ Kontrol Düşüş (Ort / Med) | $2\times$ Kazanma Oranı |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **4** | **`layercam` (Varsayılan)** | 0.7142 / 0.9826 | 0.0041 / 0.0000 | **0.9267 (%92.67)** | 0.7921 / 0.9977 | 0.0210 / 0.0000 | **0.9467 (%94.67)** |
| **4** | `hirescam` (Karşılaştırma) | 0.7264 / 0.9884 | 0.0016 / 0.0000 | **0.9600 (%96.00)** | 0.8654 / 0.9993 | 0.0017 / 0.0000 | **0.9667 (%96.67)** |
| **16** | `layercam` (Referans) | 0.8618 / 0.9989 | 0.6551 / 0.6880 | **0.7733 (%77.33)** | 0.9816 / 1.0000 | 0.9019 / 0.9947 | **0.8133 (%81.33)** |

## 3. Bulgular ve Raporlama Kuralları Özeti (5 Cümle)

1. **Mutlak Bütçeli Yapısal Silme Sadakati (Kural a):** Geniş gradyan desteğine sahip backbone Katman 4'te (pencere alanı görüntünün %99.91'i), GT kutu alanının $1\times$ ve $2\times$ bütçeleriyle haritanın en yüksek pikselleri bulanıklaştırıldığında `layercam` göreli skor düşüşü ortalamada **0.7142** ($1\times$, medyan **0.9826**) ve **0.7921** ($2\times$, medyan **0.9977**) olurken, aynı maskenin pencere içinde rastgele kaydırıldığı yapısal kontrolde düşüş sıfıra yakın (**0.0041** ve **0.0210**) kalmakta; böylece sıfıra yakın payda nedeniyle yapay oranlar yerine doğrudan raporlanan kazanma oranı `layercam` için **%92.67** ($1\times$) / **%94.67** ($2\times$) ve `hirescam` için **%96.00** ($1\times$) / **%96.67** ($2\times$) düzeyine ulaşmaktadır.
2. **Varsayılan Yöntem Seçimi (`LayerCAM @ Katman 4`):** Katman 4 silme testinde `layercam` ve `hirescam` sadakat açısından eşdeğer performans gösterse de (`1x` harita düşüş ortalaması **0.7142** vs. **0.7264**), `layercam` pozitif gradyan kapılaması sayesinde pencere içi harita enerjisinin **%64.19**'unu doğrudan GT kutusu içine odaklamakta (`hirescam`: **%39.73**, düzgün harita tabanı: **%0.11**) ve daha keskin yerelleşme sağladığı için varsayılan XAI yöntemi olarak kilitlenmiştir.
3. **Pointing Game Üçlüsü ve Mimari Katkı (Kural b):** Katman 4'te `layercam` kendi GT kutusuna pointing testinde **%86.00** (`hirescam`: **%80.00**) başarı sağlarken, rastgele ağırlıklı aynı mimarinin **%43.33** (`hirescam`: **%39.33**) ve şans tabanının **%0.11** çıkması, anchor merkezli etkin alıcı alanın (effective receptive field) mimari bir temel oluşturduğunu ancak öğrenilmiş ağırlıkların isabet oranını iki katına çıkardığını (pencere içi randomizasyon Spearman medyanı **0.1209**) dürüstçe göstermektedir.
4. **Katman 16 Referans Bulgusu ve Dar Alıcı Alan Kısıtı (Kural c):** Boyun çıkışındaki referans Katman 16'da tek bir anchor'ın gradyan desteği görüntünün yalnızca **%0.43**'ünü ($5\times 5$ hücre) kapsadığından, pencere içinde kaydırılan kontrol maskesi de GT kutusuyla örtüşerek yüksek skor düşüşüne ($1\times$ kontrol ort. **0.6551**, $2\times$ kontrol ort. **0.9019**) neden olmakta ve bu sığ alıcı alan yüzünden Katman 16 sadakat testi için uygun bulunmamaktadır.
5. **Neden Klasik Grad-CAM Değil? (Kural d):** Klasik `gradcam`, gradyanların uzamsal ortalamasını alarak konum bilgisini yok ettiği için çoklu kusur barındıran PCB kartlarında kendi kutusuna pointing oranında (**%4.00** @ Katman 4, **%6.67** @ Katman 16) ve pencere içi kutu enerjisinde (Katman 4'te **%0.49** vs. taban **%0.11**) rastgele düzeyde kalmakta, dolayısıyla tespit bazlı açıklama için kullanılamamaktadır.
