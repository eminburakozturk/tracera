# Antigravity talimatları — Tracera (PCB Hata Tespiti)

**Rolün: ANA GELİŞTİRİCİ.** Bu projenin kodunu sen yazar ve çalıştırırsın. Önce `PLAN.md` ve
`KARARLAR.md`'yi oku. Görevleri PLAN.md'deki takvim sırasıyla yap.

- Eğitimi yerel GPU'da (RTX 4060, 8 GB) çalıştır. Bellek hatasında önce batch'i düşür.
- Her deneyi MLflow'a kaydet; özet satırını `results/experiments.csv`'ye ekle.
- Kritik bir noktaya geldiğinde (aşağıdaki protokol) kendin karar verme, `>>> CLAUDE'A SOR` bloğu üret.
- Özellikle şunlarda mutlaka Claude'a sor: test setinin ayrılma şekli, tiling stratejisi,
  beklenmedik mAP, TensorRT/INT8 kalibrasyonu, README'deki sonuç yorumları.

## İşbölümü protokolü (iki ajan için ortak)

Bu repoda iki yapay zeka ajanı çalışır: **Claude (Claude Code)** ve **Antigravity (Gemini 4 Argon)**.
Ajanlar birbirine doğrudan ulaşamaz. Aradaki köprü kullanıcıdır (Burak). Amaç: Claude'un sınırlı
kredisini sadece kritik işlere harcamak, rutin işleri Antigravity'ye yaptırmak.

### Kritik işler → Claude
- Mimari karar (yeni bileşen, kütüphane seçimi, veri akışını değiştirmek)
- Değerlendirme tasarımı: metrik seçimi, test seti, karşılaştırma deneyi kurgusu
- İki denemede çözülemeyen hata
- Beklenmedik sonuç: metrik aniden çok iyi veya çok kötü, veri sızıntısı şüphesi
- Sonuçların yorumu ve README'deki teknik iddiaların doğruluğu
- Güvenlik: anahtarlar, `.env`, dışarıya açılan API

### Rutin işler → Antigravity
- Boilerplate, dosya/klasör oluşturma, requirements.txt
- Dockerfile, docker-compose, Gradio/Streamlit arayüzü
- Grafik çizimi, tablo biçimlendirme, lint/format
- README taslağı, docstring, yorum satırları
- Tekrar eden scriptler ve küçük, net tanımlı değişiklikler

### Devretme formatı
Rolün dışında kalan bir iş gelirse, işi yapma; kullanıcıya şu bloklardan birini ver:

```
>>> CLAUDE'A SOR
Bağlam: (ne yapılıyordu, 2-3 cümle)
Soru / karar: (tek net soru)
İlgili dosyalar: (yollar)
Denenenler: (varsa)
```

```
>>> ANTIGRAVITY'YE VER
Görev: (tek cümle)
Dosyalar: (oluşturulacak / değişecek yollar)
Bitti kriteri: (nasıl kontrol edilecek)
Not: (PLAN.md'deki ilgili satır)
```

### Ortak hafıza
- Her önemli karar `KARARLAR.md`'ye eklenir: `- YYYY-MM-DD [ajan] karar — gerekçe`.
- Her oturuma başlarken `PLAN.md` ve `KARARLAR.md` okunur.
- Rakamlar sadece gerçek çıktılardan yazılır. Uydurma metrik, uydurma kaynak yok.
- Push'u kullanıcı yapar. Ajan sadece commit atar (küçük, açıklayıcı mesajlarla).
