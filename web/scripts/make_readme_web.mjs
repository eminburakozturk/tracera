import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const WEB_DIR = path.resolve(__dirname, "..");
const REPO_ROOT = path.resolve(WEB_DIR, "..");
const RESULTS_DIR = path.join(REPO_ROOT, "results");
const OUT_PATH = path.join(REPO_ROOT, "docs", "readme_draft_web.md");

function parseCsv(filePath) {
  const raw = fs.readFileSync(filePath, "utf8").trim();
  const lines = raw.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length < 2) return [];
  const headers = lines[0].split(",").map((h) => h.trim());
  return lines.slice(1).map((line) => {
    const cols = line.split(",");
    const row = {};
    headers.forEach((h, i) => {
      row[h] = (cols[i] ?? "").trim();
    });
    return row;
  });
}

function num(v) {
  const n = Number.parseFloat(String(v));
  return Number.isFinite(n) ? n : null;
}

function median(vals) {
  const arr = vals.filter((v) => typeof v === "number" && Number.isFinite(v)).sort((a, b) => a - b);
  if (arr.length === 0) return null;
  const mid = Math.floor(arr.length / 2);
  return arr.length % 2 === 1 ? arr[mid] : (arr[mid - 1] + arr[mid]) / 2;
}

function main() {
  const webParityRows = parseCsv(path.join(RESULTS_DIR, "web_parity.csv"));
  const parityImages = webParityRows.filter((r) => r.image !== "SUMMARY");
  const paritySummary = webParityRows.find((r) => r.image === "SUMMARY");
  if (!paritySummary) {
    throw new Error("SUMMARY row not found in results/web_parity.csv");
  }

  const parityMedianModelMs = median(parityImages.map((r) => num(r.model_ms)));
  const parityMedianTotalMs = median(parityImages.map((r) => num(r.total_ms)));

  const webSpikeRows = parseCsv(path.join(RESULTS_DIR, "web_spike.csv"));
  const webGpuFp32Spike = webSpikeRows.find(
    (r) => r.backend === "ort-webgpu" && r.model === "FP32 ONNX",
  );
  const webGpuInt8Spike = webSpikeRows.find(
    (r) => r.backend === "ort-webgpu" && r.model === "INT8 Dynamic ONNX",
  );
  const wasmFp32Spike = webSpikeRows.find(
    (r) => r.backend === "ort-wasm" && r.model === "FP32 ONNX",
  );
  const wasmInt8Spike = webSpikeRows.find(
    (r) => r.backend === "ort-wasm" && r.model === "INT8 Dynamic ONNX",
  );

  const videoFpsRows = parseCsv(path.join(RESULTS_DIR, "web_video_fps.csv"));
  const videoWebGpu = videoFpsRows.find((r) => r.backend === "webgpu");
  const videoWasm = videoFpsRows.find((r) => r.backend === "wasm");

  const quantParityRows = parseCsv(path.join(RESULTS_DIR, "quant_parity.csv"));
  const qp025 = quantParityRows.find((r) => num(r.conf_esik) === 0.25);
  const qp0395 = quantParityRows.find((r) => num(r.conf_esik) === 0.395);

  const edgeRows = parseCsv(path.join(RESULTS_DIR, "edge_benchmark.csv"));
  const gpuHardware =
    edgeRows.find((r) => r.format.includes("GPU"))?.donanım ||
    "NVIDIA GeForce RTX 4060 Laptop GPU";
  const cpuHardware =
    edgeRows.find((r) => r.format.includes("CPU"))?.donanım ||
    "AMD Ryzen 7 8845HS w/ Radeon 780M Graphics";

  const analysisJson = JSON.parse(
    fs.readFileSync(path.join(WEB_DIR, "public", "analysis.json"), "utf8"),
  );
  const b05Clean = analysisJson.template_clean_rows.filter((r) => r.kart_no === "05");

  const onnxSizeBytes = fs.statSync(
    path.join(WEB_DIR, "public", "models", "tracera_1024.onnx"),
  ).size;
  const wasmCandidates = [
    path.join(WEB_DIR, "web_space", "ort", "ort-wasm-simd-threaded.jsep.wasm"),
    path.join(WEB_DIR, "dist", "ort", "ort-wasm-simd-threaded.jsep.wasm"),
    path.join(
      WEB_DIR,
      "node_modules",
      "onnxruntime-web",
      "dist",
      "ort-wasm-simd-threaded.jsep.wasm",
    ),
  ];
  const wasmPath = wasmCandidates.find((p) => fs.existsSync(p));
  if (!wasmPath) {
    throw new Error("ort-wasm-simd-threaded.jsep.wasm bulunamadı");
  }
  const wasmSizeBytes = fs.statSync(wasmPath).size;
  const onnxMb = onnxSizeBytes / (1024 * 1024);
  const wasmMb = wasmSizeBytes / (1024 * 1024);
  const firstDetectMb = Math.round(onnxMb + wasmMb);

  let precacheMbText = "ölçülmedi";
  const swPath = path.join(WEB_DIR, "web_space", "sw.js");
  if (fs.existsSync(swPath)) {
    const swText = fs.readFileSync(swPath, "utf8");
    const urls = new Set();
    for (const m of swText.matchAll(/url:\s*"([^"]+)"/g)) {
      urls.add(m[1].replace(/^\/+/, ""));
    }
    let totalBytes = 0;
    for (const rel of urls) {
      const target = path.join(WEB_DIR, "web_space", rel);
      if (fs.existsSync(target) && fs.statSync(target).isFile()) {
        totalBytes += fs.statSync(target).size;
      }
    }
    precacheMbText = `${(totalBytes / (1024 * 1024)).toFixed(2)} MB`;
  }

  const lines = [
    "<!-- Bu dosya web/scripts/make_readme_web.mjs tarafından results/*.csv ve derleme çıktılarından otomatik üretilir. Elle sayı düzenlemeyin. -->",
    "",
    "## Tarayıcıda / Cihazda Çalıştırma",
    "",
    "### Mimari",
    "",
    "Tracera, tek bir React + TypeScript (`web/`) arayüzünü dört dağıtım hedefine taşır:",
    "",
    "- **Statik Web / Hugging Face Static Space (`web_space/`)**: Sunucu tarafı GPU gerektirmeden doğrudan kullanıcı tarayıcısında ONNX Runtime Web (WebGPU öncelikli, yoksa tek iş parçacıklı WASM) ile `1024×1024` FP32 ONNX çıkarımı yapar.",
    "- **PWA (Progressive Web App)**: Uygulama kabuğunu Service Worker ile önbelleğe alır; ONNX modeli ve ORT çalışma zamanını yalnızca ilk tespitte indirerek sonraki açılışlarda çevrimdışı tespit sunar.",
    "- **Tauri 2 Masaüstü (`desktop/`, Windows EXE)**: Aynı web arayüzünü `isTauri()` köprüsüyle (`get_sidecar_status` / `get_api_base_url` / `tracera://api-ready`) paketlenmiş Python FastAPI yan sunucusuna bağlar; varsayılan motor **Yerel sunucu** (OpenVINO INT8 tespit + LayerCAM Katman 4) olarak açılır.",
    "- **Android APK Hazırlığı (Tauri WebView)**: Aynı WebGPU → tek iş parçacıklı WASM düşüşüyle Android WebView üzerinde tarayıcı içi çıkarım uyumu hedeflenir (ayrıntılar: `docs/android_webview_notes.md`).",
    "",
    "### Tarayıcı Paritesi (`OrtDetector` WebGPU vs. PyTorch FP32 Referans)",
    "",
    `${parityImages.length} val görüntüsü üzerinde (\`results/web_parity.csv\`, \`conf = 0.25\` iki yönlü IoU ≥ 0.5 eşleşmesi ve \`conf = 0.395\` sayım kıyası):`,
    "",
    "| Kıyas | Eşleşen (≥0.25) | Yalnız ORT | Yalnız PyTorch | İki Yönlü Eşleşme | Ort. IoU | Ort. \\|Δconf\\| | Sayım Farkı (0.395) | Ort. Model (medyan) | Uçtan Uca (medyan) |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    `| ${paritySummary.backend} vs. PyTorch FP32 | ${paritySummary.matched_025} | ${paritySummary.only_ort_025} | ${paritySummary.only_pt_025} | %${paritySummary.two_way_match_pct} | ${num(paritySummary.mean_iou)?.toFixed(4)} | ${num(paritySummary.mean_abs_dconf)?.toFixed(4)} | ${paritySummary.pt_count_0395} → ${paritySummary.ort_count_0395} (${paritySummary.diff_0395}) | ${parityMedianModelMs?.toFixed(1)} ms | ${parityMedianTotalMs?.toFixed(1)} ms |`,
    "",
    `> Ölçüm ortamı: Windows masaüstü Chrome, **${gpuHardware}** (ONNX Runtime Web \`ort-webgpu\`, FP32 ONNX). \`0.395\` çalışma noktasındaki 1 kutuluk fark eşik sınırındaki bir tahminden kaynaklanır.`,
    "",
    "### Tarayıcı Hızları (Statik Görüntü & Canlı Video)",
    "",
    `Aşağıdaki süreler dizüstü bilgisayar donanımında (**${gpuHardware}** / **${cpuHardware}**, Windows Desktop Chrome) ölçülmüştür:`,
    "",
    "| Senaryo / Kaynak | Motor | Model | Medyan Model Süresi | Canlı Video FPS | Notlar |",
    "|---|---|---|---:|---:|---|",
    `| Tekil görüntü, 10 val (\`web_spike.csv\`) | ${webGpuFp32Spike?.backend} | ${webGpuFp32Spike?.model} (${webGpuFp32Spike?.boyut_mb} MB) | ${webGpuFp32Spike?.medyan_ms} ms | — | Ort. IoU ${webGpuFp32Spike?.ort_iou}, \\|Δconf\\| ${webGpuFp32Spike?.ort_dconf} |`,
    `| Tekil görüntü, 10 val (\`web_spike.csv\`) | ${webGpuInt8Spike?.backend} | ${webGpuInt8Spike?.model} (${webGpuInt8Spike?.boyut_mb} MB) | zaman aşımı | — | Kullanımdan çıkarıldı |`,
    `| Tekil görüntü, 10 val (\`web_spike.csv\`) | ${wasmFp32Spike?.backend} | ${wasmFp32Spike?.model} (${wasmFp32Spike?.boyut_mb} MB) | ${wasmFp32Spike?.medyan_ms} ms | — | WebGPU yokken yedek; tek iş parçacıklı WASM |`,
    `| Tekil görüntü, 10 val (\`web_spike.csv\`) | ${wasmInt8Spike?.backend} | ${wasmInt8Spike?.model} (${wasmInt8Spike?.boyut_mb} MB) | ${wasmInt8Spike?.medyan_ms} ms | — | Eşleşme ${wasmInt8Spike?.eşleşme_oranı}; doğruluk kaybı nedeniyle elendi |`,
    `| Canlı video 5.2 s, \`val_pan.mp4\` (\`web_video_fps.csv\`) | ${videoWebGpu?.ort_ep} | FP32 ONNX | ${videoWebGpu?.model_ms} ms | ${videoWebGpu?.avg_fps} FPS (${videoWebGpu?.frames} kare) | Kuyruk yok (meşgulken kare atlanır) |`,
    `| Canlı video 5.2 s, \`val_pan.mp4\` (\`web_video_fps.csv\`) | ${videoWasm?.ort_ep} | FP32 ONNX | ${videoWasm?.model_ms} ms | ${videoWasm?.avg_fps} FPS (${videoWasm?.frames} kare) | WASM üzerinde akıcı video beklenmez |`,
    "",
    "### FP32 / INT8 Farkı ve Şablon Noktası Notu",
    "",
    `Tarayıcı (\`FP32 ONNX\`) ile yerel masaüstü yan sunucusu (\`OpenVINO INT8\`) sınır değerdeki kutularda küçük farklar gösterebilir (\`results/quant_parity.csv\`, 10 val görüntüsü):`,
    "",
    `| Güven Eşiği | Eşleşen | Yalnız FP32 | Yalnız INT8 | Ort. IoU | Ort. \\|Δconf\\| |`,
    `|---:|---:|---:|---:|---:|---:|`,
    `| ${qp025?.conf_esik} | ${qp025?.eslesen} | ${qp025?.yalniz_fp32} | ${qp025?.yalniz_int8} | ${num(qp025?.ort_iou)?.toFixed(4)} | ${num(qp025?.ort_dconf)?.toFixed(4)} |`,
    `| ${qp0395?.conf_esik} | ${qp0395?.eslesen} | ${qp0395?.yalniz_fp32} | ${qp0395?.yalniz_int8} | ${num(qp0395?.ort_iou)?.toFixed(4)} | ${num(qp0395?.ort_dconf)?.toFixed(4)} |`,
    "",
    `\`conf ≥ 0.395\` noktasındaki tüm FP32–INT8 farkları rastgele dağılmamıştır; Kart 05 şablonunda temiz kartta da yanlış alarm üreten 2 sabit \`spurious_copper\` arka plan konumunda (${b05Clean.map((r) => `\`conf=${r.conf}\` @ (${r.cx}, ${r.cy})`).join(" ve ")}) yoğunlaşmıştır.`,
    "",
    "### Çevrimdışı Kullanım, PWA ve İlk İndirme Boyutu",
    "",
    `- **İlk açılış precache toplamı**: yalnızca uygulama kabuğu (**${precacheMbText}**). Model açılışta kendiliğinden indirilmez.`,
    `- **İlk tespitte çalışma zamanı indirmesi**: FP32 ONNX (**${onnxMb.toFixed(2)} MB**) + WebGPU/tek iş parçacıklı WASM JSEP çalışma zamanı (**${wasmMb.toFixed(2)} MB**) olmak üzere yaklaşık **~${firstDetectMb} MB** CacheFirst stratejisiyle indirilir ve önbelleğe alınır.`,
    "- **Çevrimdışı devam**: İlk tespit sonrasında internet kesilse bile tarayıcı içi tespit (\"Cihazda\") çevrimdışı çalışmaya devam eder.",
    "",
    "### Bilinen Sınırlılıklar (Tarayıcı & Mobil)",
    "",
    `- **WASM canlı video performansı**: WebGPU desteği olmayan cihazlarda ORT WASM ile canlı video çıkarımı \`< 1 FPS\` düzeyindedir (${videoWasm?.avg_fps} FPS @ ${cpuHardware}); akıcı video akışı yerine tekil fotoğraf tespiti önerilir.`,
    "- **Kamera modu deneyseldir**: Model PKU-Market-PCB veri setindeki yukarıdan çekilmiş yüksek çözünürlüklü kart fotoğraflarıyla eğitilmiştir; el kamerası perspektifi, parlamalar veya odak bozukluğunda doğruluk düşebilir.",
    "- **LayerCAM ısı haritası tarayıcıda hesaplanmaz**: Katman 4 aktivasyon × pozitif gradyan geri yayılımı gerektirdiği için XAI açıklaması tarayıcı içinde üretilmez; \"Cihazda\" tespit modundayken açıklama isteği sırasıyla **Yerel sunucu**ya (`api/`), ulaşılamazsa **Bulut** (`ebboo/tracera` Space `/explain_api`) ucuna yönlendirilir.",
    "- **Telefon hızları ayrıca ölçülür**: Bu belgede herhangi bir telefon modeli için ölçülmemiş FPS/gecikme iddiası verilmez; kullanıcılar **Sonuçlar → Bu cihazda hız testi** üzerinden kendi cihazlarında `4 val görüntüsü × 5 tekrar` medyanını yerel olarak ölçebilir.",
    "",
  ];

  const markdown = lines.join("\n");
  const forbidden = ["mikron", "micron", "telefonda", "gerçek zamanlı"];
  const lower = markdown.toLowerCase();
  for (const term of forbidden) {
    if (lower.includes(term)) {
      throw new Error(`Yasak ifade tespit edildi (${term})!`);
    }
  }

  fs.mkdirSync(path.dirname(OUT_PATH), { recursive: true });
  fs.writeFileSync(OUT_PATH, markdown, "utf8");
  console.log(`[make_readme_web] Yazıldı: ${OUT_PATH} (${lines.length} satır, precache=${precacheMbText}, firstDetect=~${firstDetectMb} MB)`);
}

main();
