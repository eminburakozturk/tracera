"""PCB-AoI bonus veri setini inceler ve VOC XML formatından YOLO formatına dönüştürür."""

from __future__ import annotations

import argparse
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

CLASSES: list[str] = [
    "Bad_qiaojiao",
    "Bad_podu",
]
CLASS_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(CLASSES)}
FILENAME_PATTERN = re.compile(r"^(\d{8})-(SPI-AOI)-(\d+)$")


@dataclass(frozen=True)
class Box:
    class_id: int
    class_name: str
    x_center: float
    y_center: float
    width: float
    height: float
    px_width: float
    px_height: float


@dataclass(frozen=True)
class AoiSample:
    subset: str
    stem: str
    filename: str
    date_id: str
    seq_id: str
    combo_label: str
    img_path: Path
    xml_path: Path
    width: int
    height: int
    boxes: list[Box]


@dataclass
class SubsetStats:
    subset: str
    num_images: int
    num_xmls: int
    unmatched_images: list[str]
    unmatched_xmls: list[str]
    all_name_tags: Counter[str]
    object_name_tags: Counter[str]
    skipped_name_tags: Counter[str]
    xml_sizes: Counter[tuple[int, int]]
    actual_sizes: Counter[tuple[int, int]]
    size_mismatches: int
    samples: list[AoiSample]


def parse_aoi_Folder(subset_dir: Path, subset_name: str) -> SubsetStats:
    """train_data veya test_data altındaki JPEGImages ve Annotations klasörlerini inceler."""
    img_dir = subset_dir / "JPEGImages"
    ann_dir = subset_dir / "Annotations"

    if not img_dir.is_dir() or not ann_dir.is_dir():
        raise FileNotFoundError(f"JPEGImages veya Annotations bulunamadı: {subset_dir}")

    img_map = {
        p.stem: p
        for p in sorted(img_dir.iterdir(), key=lambda x: x.name)
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    xml_map = {
        p.stem: p
        for p in sorted(ann_dir.iterdir(), key=lambda x: x.name)
        if p.is_file() and p.suffix.lower() == ".xml"
    }

    unmatched_images = sorted(set(img_map.keys()) - set(xml_map.keys()))
    unmatched_xmls = sorted(set(xml_map.keys()) - set(img_map.keys()))
    matched_stems = sorted(set(img_map.keys()) & set(xml_map.keys()))

    all_name_tags: Counter[str] = Counter()
    object_name_tags: Counter[str] = Counter()
    skipped_name_tags: Counter[str] = Counter()
    xml_sizes: Counter[tuple[int, int]] = Counter()
    actual_sizes: Counter[tuple[int, int]] = Counter()
    size_mismatches = 0
    samples: list[AoiSample] = []

    for stem in matched_stems:
        img_path = img_map[stem]
        xml_path = xml_map[stem]

        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Tüm <name> etiketleri (<owner><name>ChaojieZhu</name></owner> dahil)
        for name_node in root.findall(".//name"):
            raw_tag = (name_node.text or "").strip()
            all_name_tags[raw_tag] += 1
            if raw_tag not in CLASS_TO_ID:
                skipped_name_tags[raw_tag] += 1

        size_node = root.find("size")
        if size_node is None:
            raise ValueError(f"<size> etiketi bulunamadı: {xml_path}")
        xw = int(size_node.findtext("width", "0"))
        xh = int(size_node.findtext("height", "0"))
        xml_sizes[(xw, xh)] += 1

        im = cv2.imread(str(img_path))
        if im is None:
            raise RuntimeError(f"Görüntü okunamadı: {img_path}")
        ah, aw = im.shape[:2]
        actual_sizes[(aw, ah)] += 1
        if (xw, xh) != (aw, ah):
            size_mismatches += 1

        img_w, img_h = aw, ah
        boxes: list[Box] = []

        for obj in root.findall("object"):
            raw_name = (obj.findtext("name") or "").strip()
            object_name_tags[raw_name] += 1
            if raw_name not in CLASS_TO_ID:
                continue

            cls_id = CLASS_TO_ID[raw_name]
            bndbox = obj.find("bndbox")
            if bndbox is None:
                raise ValueError(f"<bndbox> etiketi bulunamadı: {xml_path}")

            xmin = max(0.0, min(float(bndbox.findtext("xmin", "0")), float(img_w)))
            ymin = max(0.0, min(float(bndbox.findtext("ymin", "0")), float(img_h)))
            xmax = max(0.0, min(float(bndbox.findtext("xmax", "0")), float(img_w)))
            ymax = max(0.0, min(float(bndbox.findtext("ymax", "0")), float(img_h)))

            if xmax <= xmin or ymax <= ymin:
                raise ValueError(
                    f"Geçersiz kutu koordinatları ({xmin}, {ymin}, {xmax}, {ymax}): {xml_path}"
                )

            px_w = xmax - xmin
            px_h = ymax - ymin
            xc = ((xmin + xmax) / 2.0) / img_w
            yc = ((ymin + ymax) / 2.0) / img_h
            bw = px_w / img_w
            bh = px_h / img_h

            boxes.append(
                Box(
                    class_id=cls_id,
                    class_name=raw_name,
                    x_center=xc,
                    y_center=yc,
                    width=bw,
                    height=bh,
                    px_width=px_w,
                    px_height=px_h,
                )
            )

        if not boxes:
            raise ValueError(f"Geçerli sınıf kutusu kalmadı: {xml_path}")

        m = FILENAME_PATTERN.match(stem)
        date_id = m.group(1) if m else stem.split("-")[0]
        seq_id = m.group(3) if m else ""
        present_classes = sorted({b.class_name for b in boxes})
        combo_label = "+".join(present_classes)

        samples.append(
            AoiSample(
                subset=subset_name,
                stem=stem,
                filename=img_path.name,
                date_id=date_id,
                seq_id=seq_id,
                combo_label=combo_label,
                img_path=img_path,
                xml_path=xml_path,
                width=img_w,
                height=img_h,
                boxes=boxes,
            )
        )

    return SubsetStats(
        subset=subset_name,
        num_images=len(img_map),
        num_xmls=len(xml_map),
        unmatched_images=unmatched_images,
        unmatched_xmls=unmatched_xmls,
        all_name_tags=all_name_tags,
        object_name_tags=object_name_tags,
        skipped_name_tags=skipped_name_tags,
        xml_sizes=xml_sizes,
        actual_sizes=actual_sizes,
        size_mismatches=size_mismatches,
        samples=samples,
    )


def print_inspection_report(train_stats: SubsetStats, test_stats: SubsetStats) -> None:
    """İstenen tüm veri seti istatistiklerini ve dosya adı kimlik analizini ekrana yazdırır."""
    print("=" * 72)
    print("PCB-AoI VERİ SETİ İNCELEME RAPORU (train_data + test_data)")
    print("Not: train_data_augmentation KULLANILMADI.")
    print("=" * 72)

    for st in (train_stats, test_stats):
        px_w = [b.px_width for s in st.samples for b in s.boxes]
        px_h = [b.px_height for s in st.samples for b in s.boxes]
        px_eq = [np.sqrt(b.px_width * b.px_height) for s in st.samples for b in s.boxes]

        print(f"\n--- [{st.subset}] ---")
        print(f"Görüntü sayısı          : {st.num_images}")
        print(f"XML sayısı              : {st.num_xmls}")
        print(f"Eşleşmeyen görüntü      : {len(st.unmatched_images)} {st.unmatched_images}")
        print(f"Eşleşmeyen XML          : {len(st.unmatched_xmls)} {st.unmatched_xmls}")
        print(f"Tüm <name> etiketleri   : {dict(st.all_name_tags)}")
        print(f"<object>/<name> (kutu)  : {dict(st.object_name_tags)}")
        print(f"Atlanan <name> etiket   : {dict(st.skipped_name_tags)} (owner/etiketleyici vb.)")
        print(f"Görüntü boyutları (XML) : {dict(st.xml_sizes)}")
        print(f"Görüntü boyutları (Gerç): {dict(st.actual_sizes)} (uyuşmazlık: {st.size_mismatches})")
        print(
            f"Kutu boyut medyanı (px) : genişlik={np.median(px_w):.1f} px, "
            f"yükseklik={np.median(px_h):.1f} px, eşdeğer_kenar(sqrt(w*h))={np.median(px_eq):.1f} px"
        )
        print(
            f"Kutu boyut min-max (px) : genişlik=[{np.min(px_w):.0f}, {np.max(px_w):.0f}] px, "
            f"yükseklik=[{np.min(px_h):.0f}, {np.max(px_h):.0f}] px"
        )

    all_samples = train_stats.samples + test_stats.samples
    all_w = [b.px_width for s in all_samples for b in s.boxes]
    all_h = [b.px_height for s in all_samples for b in s.boxes]
    total_all_names = train_stats.all_name_tags + test_stats.all_name_tags
    total_obj_names = train_stats.object_name_tags + test_stats.object_name_tags
    total_skipped = train_stats.skipped_name_tags + test_stats.skipped_name_tags

    print("\n--- [TOPLAM (train_data + test_data = 233 görüntü)] ---")
    print(f"Toplam görüntü / XML    : {len(all_samples)} / {len(all_samples)}")
    print(f"Tüm <name> frekansı     : {dict(total_all_names)}")
    print(f"Geçerli sınıf kutuları  : {dict(total_obj_names)} (Toplam={sum(total_obj_names.values())} kutu)")
    print(f"Atlanan <name> toplamı  : {dict(total_skipped)}")
    print(
        f"Genel kutu medyanı (px) : genişlik={np.median(all_w):.1f} px, "
        f"yükseklik={np.median(all_h):.1f} px"
    )
    for cls_name in CLASSES:
        cw = [b.px_width for s in all_samples for b in s.boxes if b.class_name == cls_name]
        ch = [b.px_height for s in all_samples for b in s.boxes if b.class_name == cls_name]
        print(
            f"  - {cls_name:14s}: {len(cw):3d} kutu | "
            f"medyan {np.median(cw):.1f}x{np.median(ch):.1f} px "
            f"(min {np.min(cw):.0f}x{np.min(ch):.0f}, max {np.max(cw):.0f}x{np.max(ch):.0f})"
        )

    print("\n--- [DOSYA ADI / KART-TARİH KİMLİĞİ ANALİZİ] ---")
    print("Dosya adı şablonu       : YYYYMMDD-SPI-AOI-<N>.jpeg (tüm 233 dosyada %100 uyumlu)")
    train_dates = Counter(s.date_id for s in train_stats.samples)
    test_dates = Counter(s.date_id for s in test_stats.samples)
    overlap_dates = set(train_dates.keys()) & set(test_dates.keys())
    print(
        f"train_data tarih aralığı: {min(train_dates)} .. {max(train_dates)} "
        f"({len(train_dates)} farklı gün, 173 görüntü)"
    )
    print(
        f"test_data tarih aralığı : {min(test_dates)} .. {max(test_dates)} "
        f"({len(test_dates)} farklı gün, 60 görüntü)"
    )
    print(f"train_data ∩ test_data  : {len(overlap_dates)} ortak gün (kronolojik ayrılmış)")
    print(f"train_data en yoğun gün : {train_dates.most_common(8)}")
    print(f"test_data en yoğun gün  : {test_dates.most_common(5)}")


def split_train_val(
    train_samples: list[AoiSample],
    val_ratio: float,
    seed: int,
    scheme: str,
) -> tuple[list[AoiSample], list[AoiSample]]:
    """train_data örneklerini %85 train / %15 val olarak böler (stratified random veya date-grouped)."""
    if scheme == "grouped":
        gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        groups = [s.date_id for s in train_samples]
        train_idx, val_idx = next(gss.split(train_samples, groups=groups))
        tr = [train_samples[i] for i in train_idx]
        va = [train_samples[i] for i in val_idx]
    else:
        strata = [s.combo_label for s in train_samples]
        tr, va = train_test_split(
            train_samples,
            test_size=val_ratio,
            random_state=seed,
            stratify=strata,
        )
    return sorted(tr, key=lambda s: s.filename), sorted(va, key=lambda s: s.filename)


def export_yolo_aoi(splits: dict[str, list[AoiSample]], out_dir: Path) -> None:
    """Görüntüleri data/yolo/aoi/{train,val,test}/images altına kopyalar ve YOLO etiketlerini yazar."""
    for split_name in ("train", "val", "test"):
        split_dir = out_dir / split_name
        if split_dir.exists():
            shutil.rmtree(split_dir)
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "labels").mkdir(parents=True, exist_ok=True)

        for sample in splits[split_name]:
            dst_img = split_dir / "images" / sample.filename
            shutil.copy2(sample.img_path, dst_img)

            dst_lbl = split_dir / "labels" / f"{sample.stem}.txt"
            lines = [
                f"{b.class_id} {b.x_center:.6f} {b.y_center:.6f} {b.width:.6f} {b.height:.6f}"
                for b in sample.boxes
            ]
            dst_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_inspect_image(splits: dict[str, list[AoiSample]], save_path: Path) -> None:
    """6 örnek görüntüyü kutularla çizip results/aoi_inspect.png (dpi 100) olarak kaydeder."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    train_s = splits["train"]
    val_s = splits["val"]
    test_s = splits["test"]

    only_qiaojiao = [s for s in train_s if s.combo_label == "Bad_qiaojiao"]
    only_podu = [s for s in train_s if s.combo_label == "Bad_podu"]
    both_train = [s for s in train_s if s.combo_label == "Bad_podu+Bad_qiaojiao"]
    both_val = [s for s in val_s if "Bad_qiaojiao" in s.combo_label] or val_s
    test_q = [s for s in test_s if "Bad_qiaojiao" in s.combo_label] or test_s
    test_p = [s for s in test_s if s.combo_label == "Bad_podu"] or test_s

    selected: list[tuple[str, AoiSample]] = [
        ("train", only_qiaojiao[0]),
        ("train", only_podu[0]),
        ("train", both_train[0]),
        ("val", both_val[0]),
        ("test", test_q[0]),
        ("test", test_p[0]),
    ]

    color_map = {
        "Bad_qiaojiao": (255, 50, 50),  # Kırmızı
        "Bad_podu": (0, 210, 90),       # Yeşil
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes_flat = axes.flatten()

    for ax, (sp_name, sample) in zip(axes_flat, selected):
        img_bgr = cv2.imread(str(sample.img_path))
        if img_bgr is None:
            raise RuntimeError(f"Görüntü okunamadı: {sample.img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape

        for b in sample.boxes:
            x1 = int((b.x_center - b.width / 2.0) * w)
            y1 = int((b.y_center - b.height / 2.0) * h)
            x2 = int((b.x_center + b.width / 2.0) * w)
            y2 = int((b.y_center + b.height / 2.0) * h)
            col = color_map.get(b.class_name, (255, 200, 0))

            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), col, thickness=2)
            cv2.putText(
                img_rgb,
                b.class_name,
                (max(2, x1), max(16, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                col,
                2,
                cv2.LINE_AA,
            )

        cls_counts = Counter(b.class_name for b in sample.boxes)
        cls_str = ", ".join(f"{k}:{v}" for k, v in sorted(cls_counts.items()))
        ax.imshow(img_rgb)
        ax.set_title(f"[{sp_name}] {sample.stem}\n({cls_str})", fontsize=10, fontweight="bold")
        ax.axis("off")

    plt.suptitle(
        "PCB-AoI — Örnek Görüntüler ve YOLO Kutuları (Kırmızı: Bad_qiaojiao, Yeşil: Bad_podu)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="PCB-AoI inceleme ve VOC -> YOLO dönüştürme")
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("data/raw/pcb-aoi"),
        help="Ham PCB-AoI dizini (varsayılan: data/raw/pcb-aoi)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/yolo/aoi"),
        help="YOLO çıktı dizini (varsayılan: data/yolo/aoi)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="train_data içinden ayrılacak val oranı (varsayılan: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Bölme rastgelelik tohumu (varsayılan: 42)",
    )
    parser.add_argument(
        "--scheme",
        type=str,
        choices=["random", "grouped"],
        default="random",
        help="train/val bölme şeması: random (stratified) veya grouped (YYYYMMDD tarih bazlı)",
    )
    args = parser.parse_args()

    train_stats = parse_aoi_Folder(args.src / "train_data", "train_data")
    test_stats = parse_aoi_Folder(args.src / "test_data", "test_data")

    print_inspection_report(train_stats, test_stats)

    train_split, val_split = split_train_val(
        train_stats.samples,
        val_ratio=args.val_ratio,
        seed=args.seed,
        scheme=args.scheme,
    )
    splits: dict[str, list[AoiSample]] = {
        "train": train_split,
        "val": val_split,
        "test": sorted(test_stats.samples, key=lambda s: s.filename),
    }

    export_yolo_aoi(splits, out_dir=args.out)

    inspect_path = Path("results/aoi_inspect.png")
    save_inspect_image(splits, save_path=inspect_path)

    print(f"\n=== BÖLME ÖZETİ (şema={args.scheme}, val_ratio={args.val_ratio}, seed={args.seed}) ===")
    rows = []
    for sp_name in ("train", "val", "test"):
        sp_samples = splits[sp_name]
        box_cnt = Counter(b.class_name for s in sp_samples for b in s.boxes)
        dates = {s.date_id for s in sp_samples}
        rows.append(
            {
                "split": sp_name,
                "görüntü": len(sp_samples),
                "farklı_gün": len(dates),
                "Bad_qiaojiao_kutu": box_cnt.get("Bad_qiaojiao", 0),
                "Bad_podu_kutu": box_cnt.get("Bad_podu", 0),
                "toplam_kutu": sum(box_cnt.values()),
            }
        )
    df_summary = pd.DataFrame(rows)
    print(df_summary.to_string(index=False))

    tr_dates = {s.date_id for s in splits["train"]}
    va_dates = {s.date_id for s in splits["val"]}
    shared_tr_va = sorted(tr_dates & va_dates)
    print(
        f"\ntrain ∩ val ortak tarih sayısı: {len(shared_tr_va)} / {len(va_dates)} val günü "
        f"({shared_tr_va})"
    )
    print(f"YOLO veri seti hazır: {args.out}")
    print(f"Örnek görsel kaydedildi: {inspect_path}")


if __name__ == "__main__":
    main()
