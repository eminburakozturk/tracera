# Tracera — Açıklanabilir ve Edge Uyumlu PCB Hata Tespiti

## Amaç
Baskılı devre kartlarındaki (PCB) üretim hatalarını YOLO ile tespit eden, açıklanabilir (Grad-CAM) ve
edge cihazlar için optimize edilmiş (ONNX / OpenVINO / TensorRT) bir model. Hedef kitle: elektronik
üretimi, otomotiv yan sanayi, edge AI firmaları.

## Ortam
- Windows laptop, RTX 4060 Laptop GPU (8 GB VRAM), 32 GB RAM
- Conda ortamı: `tracera` (Python 3.11), PyTorch CUDA, `ultralytics`, `mlflow`, `onnx`, `onnxruntime-gpu`, `openvino`

## Veri
- Ana: PKU-Market-PCB — 6 sınıf: missing_hole, mouse_bite, open_circuit, short, spur, spurious_copper
- İkincil (opsiyonel, bonus): PCB-AoI — https://www.kaggle.com/datasets/kubeedgeianvs/pcb-aoi (lehim pastası denetimi, 1.211 train + ~60 test). Önce format ve sınıflar incelenir; uygunsa 13–14 Kas "sentetikten gerçeğe" fine-tune deneyi.
- Bölme: %80 train / %10 val / %10 test, sabit seed (42). Test seti hiçbir ayarda kullanılmaz.
- Sızıntı kontrolü: dosya adındaki kart numarasına (ör. `01_...`) göre gruplu bölme de denenir; rastgele bölme ile karşılaştırılır (bkz. KARARLAR.md).
- Ham veri `data/raw/`, YOLO formatı `data/yolo/`. İkisi de git'e girmez.

## Klasör yapısı
```
tracera/
  PLAN.md  AGENTS.md  CLAUDE.md  KARARLAR.md  README.md
  requirements.txt  Dockerfile  .gitignore
  configs/          # data.yaml, deney ayarları
  scripts/          # prepare_data.py, train.py, evaluate.py, export.py, benchmark.py, gradcam.py
  app/              # Gradio demo
  results/          # metrik tabloları (csv), seçili grafikler (png) -> git'e girer
  data/  runs/  weights/   # git'e girmez
```

## Adımlar ve takvim
| Gün | Görev | Bitti kriteri |
|---|---|---|
| 11 Kas | Ortam + `prepare_data.py` + ilk eğitim (yolo11n/s, 640) | `runs/` altında ilk mAP değeri |
| 12 Kas | Tiling / imgsz 1024 / augmentation denemeleri, MLflow | Deney tablosu `results/experiments.csv` |
| 13 Kas | Test seti değerlendirmesi, confusion matrix, hata analizi, Grad-CAM | `results/` altında grafikler + 5 maddelik analiz notu |
| 14 Kas | Export: PyTorch FP32, ONNX, OpenVINO (CPU), TensorRT FP16/INT8 | `results/edge_benchmark.csv` (format, MB, FPS, mAP50-95) |
| 15 Kas | Gradio demo + Hugging Face Spaces | Çalışan demo linki |
| 16 Kas | Dockerfile, README (TR + kısa EN özet) | `docker run` ile demo açılıyor |
| 17 Kas | Son kontrol, GitHub | Repo public, README'de sonuç tablosu |

## Raporlanacak metrikler
mAP50, mAP50-95, sınıf bazlı precision/recall, model boyutu (MB), FPS (batch 1, 640), donanım adı.

## Kurallar
- Tüm sayılar gerçek çıktılardan gelir. Tahmini ya da uydurma metrik yazılmaz.
- TensorRT Windows'ta yarım günden fazla sorun çıkarırsa atlanır, `KARARLAR.md`'ye not düşülür.
- Her mantıklı adım ayrı commit. Push'u kullanıcı yapar.
- Lisansı veri setlerinin lisansına göre kullanıcı belirler.
