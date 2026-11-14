# Tracera — Sürüm Günü Kontrol Listesi (`RELEASE_CHECKLIST.md`)

Bu kontrol listesi, yeni bir Tracera sürümü (ör. `v1.1.0`) yayınlanırken derleme sonrasında **elle yapılması gereken** doğrulama, dağıtım ve yayın adımlarını sırasıyla tanımlar.

---

## 1. Sürüm Öncesi Ön Kontrol

- [ ] **Dal ve çalışma ağacı kontrolü:** Tüm özellik ve arayüz commit'lerinin ilgili dalda (`feat/v1.1-ui`) tamamlandığını ve `git status` çıktısında beklenmeyen/geçici dosya bulunmadığını doğrula.
- [ ] **Sürüm tutarlılığı ve araç zinciri kontrolü (`--dry-run`):**
  ```powershell
  powershell -ExecutionPolicy Bypass -File scripts/release_build.ps1 --dry-run
  ```
  - `web/package.json`, `desktop/package.json`, `desktop/src-tauri/tauri.conf.json`, `desktop/src-tauri/Cargo.toml`, `web/public/models/meta.json`, `api/main.py` ve `tracera_core/config.py` sürümlerinin birebir aynı (`1.1.0`) olduğunu gör.
  - `node`, `npm`, `cargo`, `rustc` (`aarch64-linux-android`), `JAVA_HOME`, `ANDROID_HOME`, `NDK_HOME` ve `tracera` Python ortamının `[HAZIR]` olduğunu doğrula.
- [ ] **Python ve Web testleri:**
  ```powershell
  C:\Users\Burak\anaconda3\envs\tracera\python.exe -m pytest -q
  cd web && npm test
  ```

---

## 2. Tek Komutla Sürüm Derlemesi

- [ ] **Tam sürüm derlemesini çalıştır:**
  ```powershell
  powershell -ExecutionPolicy Bypass -File scripts/release_build.ps1
  ```
  *(Python FastAPI sidecar'ında değişiklik yapıldıysa `-RebuildSidecar` bayrağını ekle.)*
- [ ] **Derleme günlüğünü kontrol et (`results/release_build_log.md`):**
  - `(a)` Web testleri + `web/dist/` üretimi başarılı mı? (`tracera_1024_int8.onnx` dışarıda bırakıldı mı?)
  - `(b)` `web/web_space/` statik Space paketi üretildi mi?
  - `(c)` `space/` Gradio Space paketi üretildi ve izole CPU testi (`--test`) geçti mi?
  - `(d)` `Tracera_1.1.0_x64-setup.exe` üretildi mi?
  - `(e)` `Tracera_1.1.0_android_arm64.apk` üretildi mi? *(Release keystore yoksa debug imzalı olduğu günlükte belirtilir.)*
  - `(f)` `dist_release/` altında `Tracera_1.1.0_x64-setup.exe`, `Tracera_1.1.0_android_arm64.apk` ve `SHA256SUMS.txt` oluştu mu?

---

## 3. `README.md` ve Sürüm Notları Sayı Kontrolü

- [ ] **Tabloları ve taslak README'yi güncelle:**
  ```powershell
  C:\Users\Burak\anaconda3\envs\tracera\python.exe scripts/make_readme_final.py
  ```
- [ ] **Sayıları kaynak dosyalarla karşılaştır:**
  - `results/operating_point.csv`: val (`TP=514, FP=16, FN=48, P=0.9698, R=0.9146, F1=0.9414`) ve kilitli test (`TP=416, FP=142, FN=53, P=0.7455, R=0.8870, F1=0.8101`).
  - `web/public/data/reliability_val.json` ve `web/public/data/reliability_test.json`: **Güven aralığına göre kesinlik (val ve test)** tablosundaki satırların ve `0.9–1.0` (`—`) satırının birebir tuttuğunu doğrula.
  - `dist_release/SHA256SUMS.txt`: Yeni `EXE` ve `APK` dosya boyutlarının (`MiB`) ve SHA-256 özetlerinin sürüm notları ve `README.md` ile uyuştuğunu kontrol et.
- [ ] **Onay sonrası `README.md` dosyasını yaz:**
  ```powershell
  C:\Users\Burak\anaconda3\envs\tracera\python.exe scripts/make_readme_final.py --write-readme
  ```
  - `git diff README.md` ile yalnızca beklenen bölümlerin güncellendiğini doğrula ve commit'le.

---

## 4. Hugging Face Spaces ve Model Deposu Yüklemeleri

- [ ] **1) Gradio Bulut Space (`ebboo/tracera`) yüklemesi:**
  - `space/` klasöründeki dosyaları (`app.py`, `tracera_core/`, `tracera_xai/`, `weights/`, `examples/`, `results/`, `requirements.txt`, `README.md`) `https://huggingface.co/spaces/ebboo/tracera` deposuna yükle (`git push` veya `huggingface-cli upload`).
  - Space'in `Running` durumuna geçtiğini ve `/explain_api` ucunun CONTRACT v1.3 alanlarını (`energy_in_box`, `peak_in_box`, `peak_xy`) döndürdüğünü doğrula.
- [ ] **2) Statik Web Space (`ebboo/tracera-web`) yüklemesi:**
  - `web/web_space/` içeriğini `https://huggingface.co/spaces/ebboo/tracera-web` deposuna yükle.
  - Tarayıcıda açarak `v1.1` sekmelerinin (Ana Sayfa, İncele, Sonuçlar & PR/Kalibrasyon grafikleri, Kusur Rehberi, Zaman Çizelgesi) ve WebGPU/WASM çıkarımının çalıştığını kontrol et.
- [ ] **3) Model Deposu (`ebboo/tracera-model`) kontrolü:**
  - `web/public/models/meta.json` (`version: "1.1.0"`) güncellendiyse `https://huggingface.co/ebboo/tracera-model` deposundaki `meta.json` dosyasını senkronize et.
- [ ] **4) Space yüklemesinden sonra `smoke_live.py` çalıştır:**
  ```powershell
  C:\Users\Burak\anaconda3\envs\tracera\python.exe scripts/smoke_live.py
  ```
  - `results/smoke_live.md` raporunda `/detect_api` tespit listesinin ve `/explain_api` CONTRACT v1.3 alanlarının (`energy_in_box`, `peak_in_box`, `peak_xy`) başarıyla doğrulandığını kontrol et.

---

## 5. Git Dal Birleştirme, Etiketleme (`git tag`) ve Push

- [ ] **`main` dalına birleştirme:**
  ```powershell
  git switch main
  git merge --no-ff feat/v1.1-ui -m "chore(release): merge feat/v1.1-ui for v1.1.0"
  ```
- [ ] **Açıklamalı sürüm etiketi (`annotated tag`) oluşturma:**
  ```powershell
  git tag -a v1.1.0 -m "Tracera v1.1.0 — Kalibrasyon analizi, CONTRACT v1.3 XAI metrikleri, Kusur Rehberi ve yeni arayüz"
  ```
- [ ] **Uzak depoya (`origin`) gönderme:**
  ```powershell
  git push origin main
  git push origin v1.1.0
  ```
- [ ] **GitHub Actions CI kontrolü:** `main` dalında tetiklenen CI iş akışının (`ci.yml`) yeşil geçtiğini doğrula.

---

## 6. GitHub Releases Yayınlama

- [ ] GitHub üzerinde `https://github.com/eminburakozturk/tracera/releases/new` sayfasını aç.
- [ ] **Tag:** `v1.1.0` seç (Hedef: `main`).
- [ ] **Başlık:** `Tracera v1.1.0 — Web, Windows Masaüstü ve Android Sürümü`
- [ ] **APK debug imzalı; sürüm notunda belirt:** Android APK paketinin portföy projesi kapsamında `debug` anahtarı ile imzalandığını, kurulumda Android *"Bilinmeyen kaynaklardan yükleme / Yine de yükle"* onayı gerektiğini sürüm notunda açıkça belirt.
- [ ] **Açıklama:** İlgili sürüm notlarını (`docs/RELEASE_NOTES_v1.1.0.md` / `v1.0.0` şablonu) yapıştır; Windows SmartScreen (*"Ek bilgi → Yine de çalıştır"*) ve Android (*"Bilinmeyen kaynaklardan yükleme"*) kurulum notlarının yer aldığından emin ol.
- [ ] **İkili dosyaları (`dist_release/` altından) ekle:**
  1. `Tracera_1.1.0_x64-setup.exe`
  2. `Tracera_1.1.0_android_arm64.apk`
  3. `SHA256SUMS.txt`
- [ ] **Publish release** düğmesiyle yayına al.

---

## 7. Sürüm Sonrası Canlı Doğrulama (Smoke Test)

- [ ] GitHub Releases sayfasından indirilen `EXE` ve `APK` dosyalarının SHA-256 özetlerini `SHA256SUMS.txt` ile doğrula:
  ```powershell
  Get-FileHash dist_release\Tracera_1.1.0_x64-setup.exe, dist_release\Tracera_1.1.0_android_arm64.apk -Algorithm SHA256
  ```
- [ ] Canlı bağlantıları (`README.md` üst bilgi kutusu) gizli sekmede açarak erişilebilir olduğunu teyit et.
