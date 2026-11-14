"""Val kartlarından (yalnız 05 ve 11) her kusur sınıfı için 3 adet 4× bağlamlı 320×320 rehber kırpımı ve guide_index.json üretir."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CLASSES: list[str] = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]

VAL_IMAGES_DIR = REPO_ROOT / "data" / "yolo" / "grouped" / "val" / "images"
VAL_LABELS_DIR = REPO_ROOT / "data" / "yolo" / "grouped" / "val" / "labels"
OUT_DIR = REPO_ROOT / "web" / "public" / "guide"
ALLOWED_VAL_CARDS = {"05", "11"}
FORBIDDEN_TEST_CARDS = {"06", "10"}
CONTEXT_FACTOR = 4.0
CROP_PX = 320
CROPS_PER_CLASS = 3


def _collect_val_gt_candidates() -> dict[str, list[dict[str, Any]]]:
    assert VAL_IMAGES_DIR.is_dir(), f"Val görüntü dizini yok: {VAL_IMAGES_DIR}"
    assert VAL_LABELS_DIR.is_dir(), f"Val etiket dizini yok: {VAL_LABELS_DIR}"

    by_class: dict[str, list[dict[str, Any]]] = {c: [] for c in CLASSES}
    for img_path in sorted(VAL_IMAGES_DIR.glob("*.*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        kart_no = img_path.stem.split("_")[0]
        assert kart_no not in FORBIDDEN_TEST_CARDS, f"Test kartı val dizininde olamaz: {img_path.name}"
        if kart_no not in ALLOWED_VAL_CARDS:
            continue

        lbl_path = VAL_LABELS_DIR / f"{img_path.stem}.txt"
        if not lbl_path.is_file():
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        h_img, w_img = img_bgr.shape[:2]

        for line in lbl_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            xc, yc, bw, bh = [float(v) for v in parts[1:5]]
            cx = xc * w_img
            cy = yc * h_img
            bw_px = bw * w_img
            bh_px = bh * h_img
            x1 = cx - 0.5 * bw_px
            y1 = cy - 0.5 * bh_px
            x2 = cx + 0.5 * bw_px
            y2 = cy + 0.5 * bh_px

            crop_side = min(float(min(w_img, h_img)), max(80.0, CONTEXT_FACTOR * max(bw_px, bh_px)))
            half = 0.5 * crop_side
            # Kırpım penceresinin görüntünün kenarından taşmadan kare kalması için merkeze uzaklık skoru
            margin_ok = (
                cx >= half and (w_img - cx) >= half and cy >= half and (h_img - cy) >= half
            )
            by_class[CLASSES[cls_id]].append(
                {
                    "class": CLASSES[cls_id],
                    "cls_id": cls_id,
                    "img_path": img_path,
                    "source_image": img_path.name,
                    "kart_no": kart_no,
                    "w_img": w_img,
                    "h_img": h_img,
                    "gt_box": [x1, y1, x2, y2],
                    "cx": cx,
                    "cy": cy,
                    "bw_px": bw_px,
                    "bh_px": bh_px,
                    "crop_side": crop_side,
                    "margin_ok": margin_ok,
                }
            )
    return by_class


def _pick_diverse_candidates(cands: list[dict[str, Any]], k: int = CROPS_PER_CLASS) -> list[dict[str, Any]]:
    # Önce kenardan taşmayan ve orta boyuttaki net örnekleri sırala; farklı görüntüler ve 05/11 dengesi seç
    ordered = sorted(
        cands,
        key=lambda c: (
            0 if c["margin_ok"] else 1,
            -min(c["bw_px"], c["bh_px"]),
            c["source_image"],
        ),
    )

    picked: list[dict[str, Any]] = []
    used_images: set[str] = set()

    # İlk olarak hem kart 05 hem kart 11'den birer örnek al (varsa)
    for card in ("05", "11"):
        for cand in ordered:
            if cand["kart_no"] == card and cand["source_image"] not in used_images:
                picked.append(cand)
                used_images.add(cand["source_image"])
                break

    # Kalanları farklı görüntülerden tamamla
    for cand in ordered:
        if len(picked) >= k:
            break
        if cand["source_image"] not in used_images:
            picked.append(cand)
            used_images.add(cand["source_image"])

    # Hâlâ eksikse aynı görüntüden farklı kutulara izin ver
    for cand in ordered:
        if len(picked) >= k:
            break
        if cand not in picked:
            picked.append(cand)

    assert len(picked) == k, f"Yeterli GT adayı bulunamadı: {len(picked)} != {k}"
    return picked


def generate_guide_crops(out_dir: Path = OUT_DIR) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Eski kırpımları temizle
    for old in out_dir.glob("*.jpg"):
        old.unlink()

    by_class = _collect_val_gt_candidates()
    items_out: list[dict[str, Any]] = []
    img_cache: dict[Path, Any] = {}

    for cls_name in CLASSES:
        picked = _pick_diverse_candidates(by_class[cls_name], k=CROPS_PER_CLASS)
        for idx, cand in enumerate(picked, start=1):
            img_path: Path = cand["img_path"]
            assert cand["kart_no"] in ALLOWED_VAL_CARDS
            if img_path not in img_cache:
                img_cache[img_path] = cv2.imread(str(img_path))
            img_bgr = img_cache[img_path]
            h_img, w_img = cand["h_img"], cand["w_img"]

            side = float(cand["crop_side"])
            half = 0.5 * side
            cx = max(half, min(float(cand["cx"]), float(w_img) - half))
            cy = max(half, min(float(cand["cy"]), float(h_img) - half))

            cx1 = max(0, int(round(cx - half)))
            cy1 = max(0, int(round(cy - half)))
            side_i = max(1, min(int(round(side)), w_img - cx1, h_img - cy1))
            cx2 = cx1 + side_i
            cy2 = cy1 + side_i

            crop = img_bgr[cy1:cy2, cx1:cx2]
            interp = cv2.INTER_AREA if side_i > CROP_PX else cv2.INTER_LINEAR
            resized = cv2.resize(crop, (CROP_PX, CROP_PX), interpolation=interp)

            fname = f"{cls_name}_{idx}.jpg"
            dst_path = out_dir / fname
            ok = cv2.imwrite(str(dst_path), resized, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            assert ok and dst_path.is_file(), f"Kırpım kaydedilemedi: {dst_path}"

            gx1, gy1, gx2, gy2 = cand["gt_box"]
            scale_x = CROP_PX / float(cx2 - cx1)
            scale_y = CROP_PX / float(cy2 - cy1)
            bx1 = max(0.0, min(float(CROP_PX), (gx1 - cx1) * scale_x))
            by1 = max(0.0, min(float(CROP_PX), (gy1 - cy1) * scale_y))
            bx2 = max(0.0, min(float(CROP_PX), (gx2 - cx1) * scale_x))
            by2 = max(0.0, min(float(CROP_PX), (gy2 - cy1) * scale_y))

            items_out.append(
                {
                    "id": f"{cls_name}_{idx}",
                    "class": cls_name,
                    "cls_id": cand["cls_id"],
                    "variant": idx,
                    "filename": fname,
                    "path": f"/guide/{fname}",
                    "split": "val",
                    "kart_no": cand["kart_no"],
                    "source_image": cand["source_image"],
                    "crop_size": CROP_PX,
                    "context_factor": CONTEXT_FACTOR,
                    "gt_box_orig": [round(gx1, 1), round(gy1, 1), round(gx2, 1), round(gy2, 1)],
                    "crop_box_orig": [cx1, cy1, cx2, cy2],
                    "box_in_crop": [round(bx1, 1), round(by1, 1), round(bx2, 1), round(by2, 1)],
                    "size_bytes": dst_path.stat().st_size,
                }
            )

    payload: dict[str, Any] = {
        "split": "val",
        "allowed_cards": sorted(ALLOWED_VAL_CARDS),
        "crop_size": CROP_PX,
        "context_factor": CONTEXT_FACTOR,
        "crops_per_class": CROPS_PER_CLASS,
        "total_crops": len(items_out),
        "dataset_note": (
            "Bu veri setinde kusurlar, kusursuz kart fotoğraflarına sentetik olarak eklenmiştir "
            "(PKU-Market-PCB). Gerçek üretim hattı görüntülerinde performans farklı olabilir."
        ),
        "items": items_out,
    }

    index_path = out_dir / "guide_index.json"
    index_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    total_img_bytes = sum(it["size_bytes"] for it in items_out)
    print(
        f"[OK] {len(items_out)} kırpım + {index_path.name} yazıldı "
        f"(görseller toplam={total_img_bytes} bayt, index={index_path.stat().st_size} bayt)"
    )
    return payload


if __name__ == "__main__":
    generate_guide_crops()
