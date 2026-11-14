# Tracera v1.1 arayüz kuralları (Claude, 2026-11-14)

Kaynak tasarım: `design/stitch/` (Google Stitch çıktısı). **Yalnız görsel dil alınır** (yerleşim, bileşenler, renk/tipografi tokenları `precision_optics_xai/DESIGN.md`, boşluklar, hareket). **Stitch'in yazdığı içerik, sayı, etiket ve iddiaların hiçbiri kullanılmaz.** Tüm içerik repodaki verilerden gelir.

## Stitch'te uydurulmuş ve KESİNLİKLE kullanılmayacaklar
- Standart/sertifika iddiaları: IPC-A-610 (Sınıf 1/2/3), ISO 9001, ISO/IEC 17025, "Sertifika", "Sertifikalı", "Kabul uyumluluğu".
- Fiziksel ölçüler: mikron, µm, mm, µm/px, "Optik çözünürlük 48 MPix", FOV, Δ…mm, derinlik/alan değerleri, "45 mikronluk köprü".
- Yanlış teknik: YOLOv8-PCB, Conv5_3, "WebGPU FP16" (tarayıcı modeli FP32), "v2.4", 12.4 TFLOPS, VRAM tahsisi, 58 FPS, 18 ms, 38.4 ms, oto-odak/optik filtre/zoom 5x, RGB+NIR multispektral.
- Uydurma metrikler: mAP 97.4/98.1, sınıf başına F1 0.964 vb., "rastgele 0.942 / Δ-8.9%", "1.420 PCB serisi", "%99,92 eşleşme güveni", P 0,84 / R 0,79 (eşik %50), çözünürlük–recall tablosu.
- Uydurma kayıtlar: PCB-9842-X, LOT-4089, SMT-HAT-04, Rev A2, pin R14, "Baş Kalite Güvence Müh.", tarih/saatli sahte geçmiş satırları, "Golden Board 4,8 kat gradyan".
- Veri setinde olmayan kusurlar: köprü, soğuk lehim, eksik bileşen, boşluk. Dizgili (komponentli) kart fotoğrafları — bizim veri çıplak kart.
- Yanlış kişi bilgisi: "Bilgisayar Müh.", "YZ Araştırmacısı". Doğrusu aşağıda.

## Sınıf adları (tek sözlük)
| kod | TR | EN |
|---|---|---|
| missing_hole | Eksik delik | Missing hole |
| mouse_bite | Fare ısırığı (kenar çentiği) | Mouse bite |
| open_circuit | Açık devre | Open circuit |
| short | Kısa devre | Short |
| spur | Çıkıntı | Spur |
| spurious_copper | İstenmeyen bakır | Spurious copper |
Diğer terimler: kusur, tespit, güven, eşik, ısı haritası, muayene, muayene raporu (ASLA "sertifika"), geçti/kaldı (karar kuralı: eşik üstünde ≥1 kusur = KALDI; bu bir yazılım kararıdır, standart değildir — raporda yazılır).

## Veri kaynakları (sayılar yalnız buradan)
results/*.csv, web/public/metrics.json, analysis.json, ve v1.1 için Sohbet 1'in üreteceği web/public/data/*.json (pr_curve_val, reliability_val, class_metrics, guide crops, timeline). Elle sayı yazılmaz. Demo kaydındaki FPS/ms kullanılmaz.

## Geliştirici sekmesi
Emin Burak Öztürk · Yapay Zeka Operatörlüğü (Önlisans), Balıkesir Üniversitesi Balıkesir MYO · linkler: github.com/eminburakozturk, https://www.linkedin.com/in/emin-burak-%C3%B6zt%C3%BCrk-85a674395/ (tam adres, kısaltma yok), huggingface.co/ebboo. Telefon numarası YOK. E-posta: eminburak1410@gmail.com (kullanıcı onayladı, 2026-11-14; mailto bağlantısı). Fotoğraf yok: baş harf avatarı "EB".
Zaman çizelgesi: KARARLAR.md'den Claude'un onayladığı kısa başlıklar (timeline.json).

## Kalite
Tüm mevcut testler (birim, e2e, parite) geçmeye devam eder; tespit/LayerCAM/parite mantığı değişmez. Açık ve koyu tema WCAG AA (axe). prefers-reduced-motion'a uyulur. Mobil 390 px'te yatay taşma yok.

## Kusur rehberi metinleri (Claude onaylı; değiştirilmeden kullanılır)
- **Eksik delik:** Kartta bulunması gereken bir deliğin (bileşen ya da geçiş deliği) olmaması.
- **Fare ısırığı:** İletken yolun kenarında küçük, düzensiz çentikler; yol yerel olarak incelir.
- **Açık devre:** İletken yolun kopması; iki nokta arasındaki elektriksel bağlantı kesilir.
- **Kısa devre:** Ayrı kalması gereken iki iletken arasında istenmeyen bakır bağlantı.
- **Çıkıntı:** İletken yolun kenarından dışarı taşan ince, sivri bakır parçası.
- **İstenmeyen bakır:** Bakır olmaması gereken bir bölgede kalmış bakır parçası.
Rehberin altında sabit not: "Bu veri setinde kusurlar, kusursuz kart fotoğraflarına sentetik olarak eklenmiştir (PKU-Market-PCB). Gerçek üretim hattı görüntülerinde performans farklı olabilir."

## Muayene raporu (PDF) zorunlu altbilgi
"Bu rapor Tracera yazılımının otomatik çıktısıdır; herhangi bir standarda uygunluk belgesi değildir. Karar kuralı: güven eşiği (varsayılan 0,395) üzerinde en az bir kusur → KALDI."
