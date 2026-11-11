"""PCB-AoI bonus veri setini inceler, birebir kopyaları temizler ve VOC XML formatından YOLO formatına dönüştürür."""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
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
TEMPORAL_VAL_START_DATE = "20170103"


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
    file_md5: str
    pixel_md5: str
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


def parse_aoi_folder(subset_dir: Path, subset_name: str) -> SubsetStats:
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

        raw_bytes = img_path.read_bytes()
        file_md5 = hashlib.md5(raw_bytes).hexdigest()

        im = cv2.imread(str(img_path))
        if im is None:
            raise RuntimeError(f"Görüntü okunamadı: {img_path}")
        pixel_md5 = hashlib.md5(im.tobytes()).hexdigest()

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
                file_md5=file_md5,
                pixel_md5=pixel_md5,
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


def _class_signature(sample: AoiSample) -> tuple[tuple[str, int], ...]:
    """Kopya grubu içinde sınıf dağılımı ve kutu sayısının aynı olup olmadığını kontrol eder."""
    return tuple(sorted(Counter(b.class_name for b in sample.boxes).items()))


def _boxes_signature(sample: AoiSample) -> tuple[tuple[int, float, float, float, float], ...]:
    """Kopya grubu içinde kutu koordinatlarının birebir aynı olup olmadığını kontrol eder."""
    return tuple(
        sorted(
            (
                b.class_id,
                round(b.x_center, 6),
                round(b.y_center, 6),
                round(b.width, 6),
                round(b.height, 6),
            )
            for b in sample.boxes
        )
    )


def deduplicate_samples(
    train_samples: list[AoiSample],
    test_samples: list[AoiSample],
) -> tuple[
    list[AoiSample],
    list[AoiSample],
    list[tuple[AoiSample, AoiSample]],
    list[list[AoiSample]],
]:
    """233 görüntünün tamamında birebir piksel kopyalarını (MAE=0 / pixel MD5) bulur ve temizler.

    Kopya grubunda kutu sayısı en fazla olanı tutar; eşitse dosya adına göre ilkini tutar.
    Etiketleri farklı olan kopya gruplarını ayrıca raporlamak üzere döndürür.
    train_data ile test_data arasında kopya çıkarsa silmez, RuntimeError ile durdurur.
    """
    all_samples = train_samples + test_samples
    by_pixel_hash: dict[str, list[AoiSample]] = defaultdict(list)
    for s in all_samples:
        by_pixel_hash[s.pixel_md5].append(s)

    cross_subset_dups: list[list[str]] = []
    dropped_pairs: list[tuple[AoiSample, AoiSample]] = []
    label_diff_groups: list[list[AoiSample]] = []
    dropped_filenames: set[tuple[str, str]] = set()

    for p_hash, group in sorted(by_pixel_hash.items(), key=lambda kv: min(x.filename for x in kv[1])):
        if len(group) <= 1:
            continue
        # Kural: kutu sayısı en fazla olanı tut, eşitse dosya adına göre ilkini tut
        group_sorted = sorted(group, key=lambda s: (-len(s.boxes), s.filename))
        subsets_in_group = {s.subset for s in group_sorted}
        if len(subsets_in_group) > 1:
            cross_subset_dups.append([f"{s.subset}/{s.filename}" for s in group_sorted])
            continue

        cls_sigs = {_class_signature(s) for s in group_sorted}
        if len(cls_sigs) > 1:
            label_diff_groups.append(group_sorted)

        kept = group_sorted[0]
        for dup in group_sorted[1:]:
            dropped_pairs.append((kept, dup))
            dropped_filenames.add((dup.subset, dup.filename))

    if cross_subset_dups:
        raise RuntimeError(
            f"KRİTİK: train_data ve test_data arasında birebir kopya bulundu! "
            f"Silme işlemi durduruldu: {cross_subset_dups}"
        )

    clean_train = [s for s in train_samples if (s.subset, s.filename) not in dropped_filenames]
    clean_test = [s for s in test_samples if (s.subset, s.filename) not in dropped_filenames]
    return clean_train, clean_test, dropped_pairs, label_diff_groups


def print_inspection_report(
    train_stats: SubsetStats,
    test_stats: SubsetStats,
    dropped_pairs: list[tuple[AoiSample, AoiSample]],
    label_diff_groups: list[list[AoiSample]],
) -> None:
    """Veri seti istatistiklerini ve birebir kopya temizliği raporunu ekrana yazdırır."""
    print("=" * 72)
    print("PCB-AoI VERİ SETİ İNCELEME VE KOPYA TEMİZLİĞİ RAPORU")
    print("Not: train_data_augmentation KULLANILMADI.")
    print("=" * 72)

    for st in (train_stats, test_stats):
        px_w = [b.px_width for s in st.samples for b in s.boxes]
        px_h = [b.px_height for s in st.samples for b in s.boxes]
        px_eq = [np.sqrt(b.px_width * b.px_height) for s in st.samples for b in s.boxes]

        print(f"\n--- [{st.subset} (ham)] ---")
        print(f"Görüntü / XML sayısı    : {st.num_images} / {st.num_xmls} (eşleşmeyen: 0)")
        print(f"Tüm <name> etiketleri   : {dict(st.all_name_tags)}")
        print(f"<object>/<name> (kutu)  : {dict(st.object_name_tags)}")
        print(f"Atlanan <name> etiket   : {dict(st.skipped_name_tags)}")
        print(f"Görüntü boyutları       : {dict(st.actual_sizes)} (XML uyuşmazlığı: {st.size_mismatches})")
        print(
            f"Kutu boyut medyanı (px) : genişlik={np.median(px_w):.1f} px, "
            f"yükseklik={np.median(px_h):.1f} px, sqrt(w*h)={np.median(px_eq):.1f} px"
        )

    print("\n--- [BİREBİR PİKSEL KOPYA (MAE=0 / MD5) TEMİZLİĞİ (233 görüntü içinde)] ---")
    print("Kural: Kopya grubunda kutu sayısı en fazla olan tutulur; eşitse dosya adına göre ilki tutulur.")
    print("train_data ∩ test_data arası kopya : 0")
    print(f"Atılan birebir kopya sayısı        : {len(dropped_pairs)}")
    for idx, (kept, dup) in enumerate(dropped_pairs, start=1):
        same_file_md5 = kept.file_md5 == dup.file_md5
        kept_cls = dict(Counter(b.class_name for b in kept.boxes))
        dup_cls = dict(Counter(b.class_name for b in dup.boxes))
        print(
            f"  {idx}. [{dup.subset}] ATILDI: {dup.filename} ({len(dup.boxes)} kutu: {dup_cls}) "
            f"<-- TUTULAN: {kept.filename} ({len(kept.boxes)} kutu: {kept_cls}) "
            f"[pixel_md5={dup.pixel_md5[:10]}..., dosya_md5_aynı={same_file_md5}]"
        )

    if label_diff_groups:
        print(f"\n[UYARI] Etiketleri farklı olan birebir kopya grubu sayısı: {len(label_diff_groups)}")
        for g_idx, grp in enumerate(label_diff_groups, start=1):
            kept_s = grp[0]
            details = [
                f"{s.filename} (kutu={len(s.boxes)}, sınıflar={dict(Counter(b.class_name for b in s.boxes))})"
                for s in grp
            ]
            print(
                f"  ! Grup {g_idx} (TUTULAN={kept_s.filename}): " + " vs ".join(details)
            )


def split_train_val(
    train_samples: list[AoiSample],
    val_ratio: float,
    seed: int,
    scheme: str,
    val_start_date: str = TEMPORAL_VAL_START_DATE,
) -> tuple[list[AoiSample], list[AoiSample]]:
    """train_data örneklerini seçilen şemaya göre train ve val olarak böler."""
    if scheme == "temporal":
        tr = [s for s in train_samples if s.date_id < val_start_date]
        va = [s for s in train_samples if s.date_id >= val_start_date]
    elif scheme == "grouped":
        gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        groups = [s.date_id for s in train_samples]
        train_idx, val_idx = next(gss.split(train_samples, groups=groups))
        tr = [train_samples[i] for i in train_idx]
        va = [train_samples[i] for i in val_idx]
    elif scheme == "random":
        strata = [s.combo_label for s in train_samples]
        tr, va = train_test_split(
            train_samples,
            test_size=val_ratio,
            random_state=seed,
            stratify=strata,
        )
    else:
        raise ValueError(f"Bilinmeyen bölme şeması: {scheme}")

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

    only_qiaojiao = [s for s in train_s if s.combo_label == "Bad_qiaojiao"] or train_s
    only_podu = [s for s in train_s if s.combo_label == "Bad_podu"] or train_s
    both_train = [s for s in train_s if s.combo_label == "Bad_podu+Bad_qiaojiao"] or train_s
    val_q = [s for s in val_s if "Bad_qiaojiao" in s.combo_label] or val_s
    test_q = [s for s in test_s if "Bad_qiaojiao" in s.combo_label] or test_s
    test_p = [s for s in test_s if s.combo_label == "Bad_podu"] or test_s

    selected: list[tuple[str, AoiSample]] = [
        ("train", only_qiaojiao[0]),
        ("train", only_podu[0]),
        ("train", both_train[0]),
        ("val", val_q[0]),
        ("test", test_q[0]),
        ("test", test_p[0]),
    ]

    color_map = {
        "Bad_qiaojiao": (255, 50, 50),  # Kırmızı (0)
        "Bad_podu": (0, 210, 90),       # Yeşil (1)
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
        "PCB-AoI — Örnek Görüntüler ve YOLO Kutuları (0 Kırmızı: Bad_qiaojiao, 1 Yeşil: Bad_podu)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="PCB-AoI inceleme, tekilleştirme ve VOC -> YOLO dönüştürme")
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
        "--scheme",
        type=str,
        choices=["temporal", "grouped", "random"],
        default="temporal",
        help="train/val bölme şeması: temporal (varsayılan, >=20170103 val), grouped veya random",
    )
    parser.add_argument(
        "--val-start-date",
        type=str,
        default=TEMPORAL_VAL_START_DATE,
        help="temporal şemada val başlangıç tarihi YYYYMMDD (varsayılan: 20170103)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="random/grouped şemalarında train_data içinden ayrılacak val oranı (varsayılan: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random/grouped şemalarında rastgelelik tohumu (varsayılan: 42)",
    )
    args = parser.parse_args()

    train_stats = parse_aoi_folder(args.src / "train_data", "train_data")
    test_stats = parse_aoi_folder(args.src / "test_data", "test_data")

    clean_train, clean_test, dropped_pairs, label_diff_groups = deduplicate_samples(
        train_stats.samples,
        test_stats.samples,
    )

    print_inspection_report(train_stats, test_stats, dropped_pairs, label_diff_groups)

    train_split, val_split = split_train_val(
        clean_train,
        val_ratio=args.val_ratio,
        seed=args.seed,
        scheme=args.scheme,
        val_start_date=args.val_start_date,
    )
    splits: dict[str, list[AoiSample]] = {
        "train": train_split,
        "val": val_split,
        "test": sorted(clean_test, key=lambda s: s.filename),
    }

    tr_dates = {s.date_id for s in splits["train"]}
    va_dates = {s.date_id for s in splits["val"]}
    te_dates = {s.date_id for s in splits["test"]}

    if args.scheme in {"temporal", "grouped"}:
        assert tr_dates.isdisjoint(va_dates), f"train ve val tarih kesişimi boş değil: {tr_dates & va_dates}"
    assert tr_dates.isdisjoint(te_dates), f"train ve test tarih kesişimi boş değil: {tr_dates & te_dates}"
    assert va_dates.isdisjoint(te_dates), f"val ve test tarih kesişimi boş değil: {va_dates & te_dates}"

    export_yolo_aoi(splits, out_dir=args.out)

    inspect_path = Path("results/aoi_inspect.png")
    save_inspect_image(splits, save_path=inspect_path)

    print(f"\n=== SPLIT × (GÖRÜNTÜ, GÜN, KUTU/SINIF) TABLOSU [şema={args.scheme}] ===")
    rows = []
    for sp_name in ("train", "val", "test"):
        sp_samples = splits[sp_name]
        box_cnt = Counter(b.class_name for s in sp_samples for b in s.boxes)
        dates = sorted({s.date_id for s in sp_samples})
        rows.append(
            {
                "split": sp_name,
                "görüntü": len(sp_samples),
                "gün": len(dates),
                "tarih_aralığı": f"{dates[0]}..{dates[-1]}" if dates else "-",
                "Bad_qiaojiao (0)": box_cnt.get("Bad_qiaojiao", 0),
                "Bad_podu (1)": box_cnt.get("Bad_podu", 0),
                "toplam_kutu": sum(box_cnt.values()),
            }
        )
    total_imgs = sum(r["görüntü"] for r in rows)
    total_days = sum(r["gün"] for r in rows)
    total_q = sum(r["Bad_qiaojiao (0)"] for r in rows)
    total_p = sum(r["Bad_podu (1)"] for r in rows)
    rows.append(
        {
            "split": "TOPLAM",
            "görüntü": total_imgs,
            "gün": total_days,
            "tarih_aralığı": "20161019..20180601",
            "Bad_qiaojiao (0)": total_q,
            "Bad_podu (1)": total_p,
            "toplam_kutu": total_q + total_p,
        }
    )
    df_summary = pd.DataFrame(rows)
    print(df_summary.to_string(index=False))

    print(
        f"\nAssert doğrulandı: train ∩ val = {len(tr_dates & va_dates)}, "
        f"train ∩ test = {len(tr_dates & te_dates)}, val ∩ test = {len(va_dates & te_dates)}"
    )
    print(f"Atılan birebir kopya sayısı: {len(dropped_pairs)} (233 -> {total_imgs} görüntü)")
    print(f"YOLO veri seti hazır: {args.out}")
    print(f"Örnek görsel güncellendi: {inspect_path}")


if __name__ == "__main__":
    main()
