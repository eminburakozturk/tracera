# Tracera — Android WebView (APK) Uyumluluk Notları

Bu belge `web/` uygulamasının Android System WebView (Tauri 2 Android kabuğu veya mobil Chromium WebView) altında sorunsuz çalışması için mevcut uygulamadaki teknik kararları ve native Android tarafında gereken izinleri özetler (`desktop/` klasörüne dokunmadan).

## 1. WebGPU Yoksa WASM Düşüşü

- [ort.worker.ts](file:///c:/Users/Burak/Documents/Projeler/tracera/web/src/inference/ort/ort.worker.ts) içinde `hasWebGPU()`, `navigator.gpu?.requestAdapter()` çağrısıyla WebGPU desteğini kontrol eder. Android cihazlarda veya System WebView sürümlerinde WebGPU adaptörü bulunmazsa veya oturum derlemesi başarısız olursa çıkarım otomatik olarak `wasm` execution provider'ına (`ort-wasm`) düşer.
- [LiveView.tsx](file:///c:/Users/Burak/Documents/Projeler/tracera/web/src/components/LiveView.tsx) aynı `navigator.gpu?.requestAdapter()` kontrolünü açılışta çalıştırır; WebGPU olmadığında kullanıcıya `"WebGPU yok; canlı mod bu cihazda çok yavaş (<1 FPS) çalışır."` uyarısını (`data-testid="live-wasm-warning"`) gösterir.

## 2. `SharedArrayBuffer` Olmadan Tek İş Parçacıklı WASM

- Statik barındırıcılarda (HF Static Space, PWA önbelleği) ve Android WebView'de `Cross-Origin-Opener-Policy: same-origin` / `Cross-Origin-Embedder-Policy: require-corp` (`window.crossOriginIsolated === true`) başlıkları zorunlu değildir.
- [ort.worker.ts](file:///c:/Users/Burak/Documents/Projeler/tracera/web/src/inference/ort/ort.worker.ts) oturum açılışında açıkça şu ayarları uygular:
  - `ort.env.wasm.numThreads = 1`
  - `ort.env.wasm.proxy = false`
- Bu sayede `SharedArrayBuffer` gereksinimi ve iç içe `pthread` worker kilitlenmesi olmadan, tek iş parçacıklı JSEP WASM ikilisi (`ort-wasm-simd-threaded.jsep.mjs` + `ort-wasm-simd-threaded.jsep.wasm`, `web_space/ort/`) doğrudan Web Worker içinde çalışır.

## 3. Kamera (`getUserMedia`) ve Dosya Seçici Gereksinimleri

Web arayüzü Android'de üç giriş yöntemini kullanır:
1. Dosya yükleme: `<input type="file" accept="image/*" />` ([DetectView.tsx](file:///c:/Users/Burak/Documents/Projeler/tracera/web/src/components/DetectView.tsx)) ve `<input type="file" accept="video/mp4,video/webm,video/*" />` ([LiveView.tsx](file:///c:/Users/Burak/Documents/Projeler/tracera/web/src/components/LiveView.tsx)).
2. Doğrudan fotoğraf çekimi: `<input type="file" accept="image/*" capture="environment" />` ([DetectView.tsx](file:///c:/Users/Burak/Documents/Projeler/tracera/web/src/components/DetectView.tsx)).
3. Canlı video akışı: `navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" } }, audio: false })` ([LiveView.tsx](file:///c:/Users/Burak/Documents/Projeler/tracera/web/src/components/LiveView.tsx)).

Android APK / WebView paketlemesinde bunların çalışması için gerekenler:

| Bileşen | Gereksinim |
|---|---|
| **Güvenli Bağlam (Secure Context)** | `getUserMedia` yalnız `https://...`, `http://localhost` veya Tauri Android özel kökeni (`http://tauri.localhost` / `https://tauri.localhost`) içinde aktiftir; düz LAN `http://192.168.x.x` adresinde WebView tarafından engellenir. |
| **Android Manifest İzinleri** | `<uses-permission android:name="android.permission.CAMERA" />` (hem `getUserMedia` hem `capture="environment"` için zorunlu). Galeri/dosya seçimi için Android 13+ sistem Photo Picker ek izinsiz çalışabilir; Android ≤12 hedefi için `READ_EXTERNAL_STORAGE` (`maxSdkVersion="32"`) gerekir. `<uses-feature android:name="android.hardware.camera.any" android:required="false" />` kamerasız cihazlarda yüklemeyi korumak için önerilir. |
| **Çalışma Zamanı İzinleri & `WebChromeClient`** | Android WebView `WebChromeClient.onPermissionRequest` çağrısında `PermissionRequest.RESOURCE_VIDEO_CAPTURE` isteğini uygulama izni alındıktan sonra onaylamalıdır; aksi halde `LiveView` `live.cameraError` yakalar. Dosya ve `capture="environment"` seçicileri için `WebChromeClient.onShowFileChooser` (Tauri Android'de `wry` tarafından sağlanır) aktif olmalıdır. |
| **Çıkarım & Bellek Bütçesi** | `tracera_1024.onnx` (36.43 MB) + WASM çalışma zamanı (28.15 MB) ilk tespitte RAM/Cache Storage içinde tutulur (`~65 MB`); düşük RAM'li cihazlarda Android WebView'in `android:largeHeap="true"` ayarı olmadan bellek baskısı yaşanabilir (cihazda test edilmesi gerekir). |
