# Veri

Ham veri git'e girmez. İndirme:

1. https://www.kaggle.com/datasets/akhatova/pcb-defects adresinden zip'i indir (Kaggle hesabı gerekir).
2. Zip'i `data/raw/` içine aç. Beklenen yapı: `data/raw/PCB_DATASET/images/` ve `data/raw/PCB_DATASET/Annotations/` (VOC XML).
3. YOLO formatına çeviri `scripts/prepare_data.py` ile yapılır, çıktı `data/yolo/` altına yazılır.

Kaynak: PKU-Market-PCB (Peking Üniversitesi Open Lab on Human Robot Interaction), 693 görüntü, 6 sınıf:
missing_hole, mouse_bite, open_circuit, short, spur, spurious_copper.

## Opsiyonel: PCB-AoI
https://www.kaggle.com/datasets/kubeedgeianvs/pcb-aoi
Zip'i `data/raw/pcb-aoi/` içine aç. Format ve sınıflar ilk gün incelenecek (bkz. PLAN.md).
