"""PKU-Market-PCB veri setini VOC XML formatından YOLO formatına dönüştürür ve rastgele (stratified) böler."""

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

CLASSES: list[str] = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]
CLASS_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(CLASSES)}


@dataclass(frozen=True)
class Box:
    class_id: int
    class_name: str
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass(frozen=True)
class Sample:
    stem: str
    filename: str
    board_id: str
    folder_class: str
    img_path: Path
    xml_path: Path
    width: int
    height: int
    boxes: list[Box]


def parse_voc_xml(xml_path: Path) -> tuple[int, int, list[Box]]:
    """VOC XML dosyasını okur ve (genişlik, yükseklik, YOLO kutuları) döndürür."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_node = root.find("size")
    if size_node is None:
        raise ValueError(f"<size> etiketi bulunamadı: {xml_path}")
    img_w = int(size_node.findtext("width", "0"))
    img_h = int(size_node.findtext("height", "0"))
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Geçersiz görüntü boyutu ({img_w}x{img_h}): {xml_path}")

    boxes: list[Box] = []
    for obj in root.findall("object"):
        raw_name = obj.findtext("name", "").strip()
        if raw_name not in CLASS_TO_ID:
            raise ValueError(
                f"Bilinmeyen sınıf etiketi '{raw_name}' ({xml_path}). "
                f"Beklenen sınıflar: {CLASSES}"
            )
        cls_id = CLASS_TO_ID[raw_name]

        bndbox = obj.find("bndbox")
        if bndbox is None:
            raise ValueError(f"<bndbox> etiketi bulunamadı: {xml_path}")

        xmin = float(bndbox.findtext("xmin", "0"))
        ymin = float(bndbox.findtext("ymin", "0"))
        xmax = float(bndbox.findtext("xmax", "0"))
        ymax = float(bndbox.findtext("ymax", "0"))

        xmin = max(0.0, min(xmin, float(img_w)))
        xmax = max(0.0, min(xmax, float(img_w)))
        ymin = max(0.0, min(ymin, float(img_h)))
        ymax = max(0.0, min(ymax, float(img_h)))

        if xmax <= xmin or ymax <= ymin:
            raise ValueError(
                f"Geçersiz kutu koordinatları ({xmin}, {ymin}, {xmax}, {ymax}): {xml_path}"
            )

        x_center = ((xmin + xmax) / 2.0) / img_w
        y_center = ((ymin + ymax) / 2.0) / img_h
        bw = (xmax - xmin) / img_w
        bh = (ymax - ymin) / img_h

        for val_name, val in [("x_center", x_center), ("y_center", y_center), ("w", bw), ("h", bh)]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"Koordinat 0-1 aralığı dışında ({val_name}={val}): {xml_path}")

        boxes.append(
            Box(
                class_id=cls_id,
                class_name=raw_name,
                x_center=x_center,
                y_center=y_center,
                width=bw,
                height=bh,
            )
        )

    if not boxes:
        raise ValueError(f"Hiç kutu bulunamadı: {xml_path}")

    return img_w, img_h, boxes


def collect_samples(src_dir: Path) -> list[Sample]:
    """data/raw/PCB_DATASET/images ve Annotations altındaki örnekleri toplar."""
    images_root = src_dir / "images"
    ann_root = src_dir / "Annotations"

    if not images_root.is_dir() or not ann_root.is_dir():
        raise FileNotFoundError(
            f"'images' veya 'Annotations' klasörü bulunamadı: {src_dir}"
        )

    img_subdirs = sorted([d for d in images_root.iterdir() if d.is_dir()], key=lambda p: p.name.lower())
    samples: list[Sample] = []

    for img_subdir in img_subdirs:
        folder_class = img_subdir.name.lower()
        if folder_class not in CLASS_TO_ID:
            raise ValueError(
                f"Bilinmeyen klasör sınıfı '{img_subdir.name}'. Beklenen: {CLASSES}"
            )

        ann_subdir = ann_root / img_subdir.name
        if not ann_subdir.is_dir():
            raise FileNotFoundError(f"Eşleşen Annotations alt klasörü yok: {ann_subdir}")

        img_files = sorted(
            [p for p in img_subdir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
            key=lambda p: p.name,
        )

        for img_path in img_files:
            xml_path = ann_subdir / f"{img_path.stem}.xml"
            if not xml_path.is_file():
                raise FileNotFoundError(f"Görüntü için XML bulunamadı: {xml_path}")

            img_w, img_h, boxes = parse_voc_xml(xml_path)
            board_id = img_path.name[:2]

            samples.append(
                Sample(
                    stem=img_path.stem,
                    filename=img_path.name,
                    board_id=board_id,
                    folder_class=folder_class,
                    img_path=img_path,
                    xml_path=xml_path,
                    width=img_w,
                    height=img_h,
                    boxes=boxes,
                )
            )

    samples.sort(key=lambda s: (s.folder_class, s.filename))
    return samples


def split_samples(samples: list[Sample], seed: int) -> dict[str, list[Sample]]:
    """%80 train, %10 val, %10 test olacak şekilde sınıfa göre stratified böler."""
    labels = [s.folder_class for s in samples]
    train_list, temp_list = train_test_split(
        samples,
        test_size=0.20,
        random_state=seed,
        stratify=labels,
    )
    temp_labels = [s.folder_class for s in temp_list]
    val_list, test_list = train_test_split(
        temp_list,
        test_size=0.50,
        random_state=seed,
        stratify=temp_labels,
    )

    return {
        "train": sorted(train_list, key=lambda s: s.filename),
        "val": sorted(val_list, key=lambda s: s.filename),
        "test": sorted(test_list, key=lambda s: s.filename),
    }


def export_yolo_dataset(
    splits: dict[str, list[Sample]],
    out_dir: Path,
    meta_path: Path,
) -> pd.DataFrame:
    """Görüntüleri kopyalar, YOLO etiket dosyalarını ve meta.csv'yi yazar."""
    for split_name in ("train", "val", "test"):
        split_dir = out_dir / split_name
        if split_dir.exists():
            shutil.rmtree(split_dir)
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "labels").mkdir(parents=True, exist_ok=True)

    meta_rows: list[dict[str, object]] = []

    for split_name in ("train", "val", "test"):
        img_out_dir = out_dir / split_name / "images"
        lbl_out_dir = out_dir / split_name / "labels"

        for sample in splits[split_name]:
            dst_img = img_out_dir / sample.filename
            shutil.copy2(sample.img_path, dst_img)

            dst_lbl = lbl_out_dir / f"{sample.stem}.txt"
            lines = [
                f"{b.class_id} {b.x_center:.6f} {b.y_center:.6f} {b.width:.6f} {b.height:.6f}"
                for b in sample.boxes
            ]
            dst_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")

            meta_rows.append(
                {
                    "dosya": sample.filename,
                    "kart_no": sample.board_id,
                    "sınıf": sample.folder_class,
                    "kutu_sayısı": len(sample.boxes),
                    "genişlik": sample.width,
                    "yükseklik": sample.height,
                    "split": split_name,
                }
            )

    df_meta = pd.DataFrame(meta_rows)
    df_meta = df_meta.sort_values(by=["split", "sınıf", "dosya"]).reset_index(drop=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    df_meta.to_csv(meta_path, index=False, encoding="utf-8")
    return df_meta


def save_label_check_image(out_dir: Path, save_path: Path) -> None:
    """Her sınıftan 1 örnek görüntüyü YOLO etiketleriyle çizip kaydeder."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    train_img_dir = out_dir / "train" / "images"
    train_lbl_dir = out_dir / "train" / "labels"

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes_flat = axes.flatten()

    for idx, cls_name in enumerate(CLASSES):
        ax = axes_flat[idx]
        matching_imgs = sorted(train_img_dir.glob(f"*_{cls_name}_*.jpg"))
        if not matching_imgs:
            raise FileNotFoundError(f"'{cls_name}' sınıfı için örnek görüntü bulunamadı.")

        img_path = matching_imgs[0]
        lbl_path = train_lbl_dir / f"{img_path.stem}.txt"

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise RuntimeError(f"Görüntü okunamadı: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape

        lines = [ln.strip() for ln in lbl_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for line in lines:
            parts = line.split()
            cls_id = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:5])

            x1 = int((xc - bw / 2.0) * w)
            y1 = int((yc - bh / 2.0) * h)
            x2 = int((xc + bw / 2.0) * w)
            y2 = int((yc + bh / 2.0) * h)

            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 40, 40), thickness=5)
            cv2.putText(
                img_rgb,
                CLASSES[cls_id],
                (max(0, x1), max(35, y1 - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                (255, 40, 40),
                3,
                cv2.LINE_AA,
            )

        ax.imshow(img_rgb)
        ax.set_title(f"{img_path.name} ({len(lines)} kutu)", fontsize=12, fontweight="bold")
        ax.axis("off")

    plt.suptitle("PKU-Market-PCB — YOLO Etiket Doğrulama (Her Sınıftan 1 Örnek)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="PKU-Market-PCB VOC XML -> YOLO veri hazırlama")
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("data/raw/PCB_DATASET"),
        help="Ham veri kök dizini (varsayılan: data/raw/PCB_DATASET)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/yolo/random"),
        help="YOLO çıktı dizini (varsayılan: data/yolo/random)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Rastgele bölme tohumu (varsayılan: 42)",
    )
    args = parser.parse_args()

    meta_path = (
        args.out.parent / "meta.csv"
        if args.out.name == "random"
        else args.out / "meta.csv"
    )
    check_img_path = Path("results/label_check.png")

    samples = collect_samples(args.src)
    splits = split_samples(samples, seed=args.seed)
    df_meta = export_yolo_dataset(splits, out_dir=args.out, meta_path=meta_path)
    save_label_check_image(out_dir=args.out, save_path=check_img_path)

    split_order = ["train", "val", "test"]
    df_meta["split"] = pd.Categorical(df_meta["split"], categories=split_order, ordered=True)

    print("\n=== SPLIT x SINIF GÖRÜNTÜ SAYISI TABLOSU ===")
    tbl_class = pd.crosstab(df_meta["split"], df_meta["sınıf"], margins=True, margins_name="Toplam")
    print(tbl_class.to_string())

    print("\n=== SPLIT x KART_NO GÖRÜNTÜ SAYISI TABLOSU ===")
    tbl_board = pd.crosstab(df_meta["split"], df_meta["kart_no"], margins=True, margins_name="Toplam")
    print(tbl_board.to_string())

    print(f"\nToplam görüntü: {len(df_meta)} | Toplam kutu: {df_meta['kutu_sayısı'].sum()}")
    print(f"meta.csv kaydedildi: {meta_path}")
    print(f"Etiket kontrol görseli kaydedildi: {check_img_path}")


if __name__ == "__main__":
    main()
