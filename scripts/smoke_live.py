"""Tracera canlı Hugging Face Space (/detect_api ve /explain_api) duman testi.

Kullanım:
    python scripts/smoke_live.py
    python scripts/smoke_live.py --space ebboo/tracera --wait-s 90 --out results/smoke_live.md
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any

from gradio_client import Client, handle_file

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPACE = "ebboo/tracera"
DEFAULT_IMAGE = REPO_ROOT / "app" / "examples" / "val_05_missing_hole_01.jpg"
DEFAULT_OUT = REPO_ROOT / "results" / "smoke_live.md"
DEFAULT_CONF = 0.395


def _call_with_wakeup_retry(
    fn,
    *,
    max_wait_s: float = 90.0,
    retry_interval_s: float = 6.0,
    label: str = "call",
) -> tuple[Any, float, int]:
    """Space uykudaysa veya uyanıyorsa max_wait_s boyunca yeniden dener; (sonuc, sure_s, deneme) dondurur."""
    t_start = time.perf_counter()
    attempts = 0
    last_err: Exception | None = None

    while True:
        attempts += 1
        t_call = time.perf_counter()
        try:
            res = fn()
            call_s = time.perf_counter() - t_call
            return res, call_s, attempts
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            elapsed_total = time.perf_counter() - t_start
            remaining = max_wait_s - elapsed_total
            if remaining <= 0.5:
                raise RuntimeError(
                    f"{label} {attempts} denemede ({elapsed_total:.1f} s) basarisiz oldu: {last_err}"
                ) from exc
            sleep_s = min(retry_interval_s, remaining)
            print(
                f"[smoke_live] {label} deneme {attempts} hata verdi ({type(exc).__name__}: {exc}); "
                f"Space uyaniyor olabilir, {sleep_s:.1f} s sonra tekrar denenecek..."
            )
            time.sleep(sleep_s)


def _write_report(
    out_path: Path,
    *,
    timestamp_str: str,
    space_id: str,
    img_path: Path,
    conf: float,
    overall_ok: bool,
    connect_s: float | None,
    connect_attempts: int,
    detect_s: float | None,
    detect_attempts: int,
    detect_res: dict[str, Any] | None,
    explain_s: float | None,
    explain_attempts: int,
    explain_res: dict[str, Any] | None,
    selected_box_str: str | None,
    selected_cls_id: int | None,
    errors: list[str],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    status_badge = "BASARILI (PASS)" if overall_ok else "BASARISIZ / EKSIK (FAIL)"

    lines: list[str] = [
        "# Tracera — Canlı Hugging Face Space Duman Testi (`smoke_live.md`)",
        "",
        f"- **Tarih / Saat:** `{timestamp_str}`",
        f"- **Hedef Space:** `https://huggingface.co/spaces/{space_id}` (`{space_id}`)",
        f"- **Test Görüntüsü:** `{img_path.relative_to(REPO_ROOT).as_posix()}`",
        f"- **Güven Eşiği (`conf`):** `{conf}`",
        f"- **Genel Sonuç:** **{status_badge}**",
        "",
        "## 1. İstek Süreleri ve Deneme Sayıları",
        "",
        "| Aşama | Süre (s) | Deneme | Durum |",
        "|---|---:|---:|---|",
        f"| Space bağlantısı (`Client`) | {f'{connect_s:.2f}' if connect_s is not None else '—'} | {connect_attempts} | {'OK' if connect_s is not None else 'HATA'} |",
        f"| `/detect_api` | {f'{detect_s:.2f}' if detect_s is not None else '—'} | {detect_attempts} | {'OK' if isinstance(detect_res, dict) else 'HATA'} |",
        f"| `/explain_api` | {f'{explain_s:.2f}' if explain_s is not None else '—'} | {explain_attempts} | {'OK' if isinstance(explain_res, dict) and not errors else 'HATA / EKSIK'} |",
        "",
    ]

    if isinstance(detect_res, dict):
        dets = detect_res.get("detections") or []
        timing = detect_res.get("timing") or {}
        lines.extend([
            "## 2. `/detect_api` Çıktı Alanları",
            "",
            f"- **Boyut (`width × height`):** `{detect_res.get('width')} × {detect_res.get('height')}`",
            f"- **Backend:** `{detect_res.get('backend')}`",
            f"- **Tespit Sayısı:** `{len(dets)}`",
            f"- **Sunucu Zamanlaması (`timing`):** `{json.dumps(timing, ensure_ascii=False)}`",
            "",
        ])
        if dets:
            lines.extend([
                "| id | cls | cls_id | conf | box [x1, y1, x2, y2] |",
                "|---:|---|---:|---:|---|",
            ])
            for d in dets:
                box_fmt = ", ".join(f"{float(v):.1f}" for v in d.get("box", []))
                lines.append(
                    f"| {d.get('id')} | `{d.get('cls')}` | {d.get('cls_id')} | "
                    f"{float(d.get('conf', 0.0)):.4f} | `[{box_fmt}]` |"
                )
            lines.append("")

    if isinstance(explain_res, dict):
        exp_keys = sorted(explain_res.keys())
        e_timing = explain_res.get("timing") or {}
        lines.extend([
            "## 3. `/explain_api` Çıktı Alanları (CONTRACT v1.3)",
            "",
            f"- **Açıklanan Kutu (`box`, `cls_id`):** `{selected_box_str}` (`cls_id={selected_cls_id}`)",
            f"- **Dönen Anahtarlar:** `{', '.join(exp_keys)}`",
            f"- **`method` / `layer`:** `{explain_res.get('method')}` / `{explain_res.get('layer')}`",
            f"- **`overlay_png` uzunluğu:** `{len(str(explain_res.get('overlay_png', '')))} karakter`",
            f"- **`zoom_png` uzunluğu:** `{len(str(explain_res.get('zoom_png', '')))} karakter`",
            f"- **`energy_in_box`:** `{explain_res.get('energy_in_box', 'EKSIK')}`",
            f"- **`peak_in_box`:** `{explain_res.get('peak_in_box', 'EKSIK')}`",
            f"- **`peak_xy`:** `{explain_res.get('peak_xy', 'EKSIK')}`",
            f"- **Sunucu Zamanlaması (`timing`):** `{json.dumps(e_timing, ensure_ascii=False)}`",
            "",
        ])

    if errors:
        lines.extend([
            "## 4. Doğrulama Hataları / Uyarılar",
            "",
        ])
        for err in errors:
            lines.append(f"- {err}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def run_smoke(space_id: str, img_path: Path, conf: float, wait_s: float, out_path: Path) -> int:
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not img_path.is_file():
        raise FileNotFoundError(f"Test goruntusu bulunamadi: {img_path}")

    connect_s: float | None = None
    connect_attempts = 0
    detect_s: float | None = None
    detect_attempts = 0
    detect_res: dict[str, Any] | None = None
    explain_s: float | None = None
    explain_attempts = 0
    explain_res: dict[str, Any] | None = None
    selected_box_str: str | None = None
    selected_cls_id: int | None = None
    errors: list[str] = []

    client: Client | None = None
    try:
        client, connect_s, connect_attempts = _call_with_wakeup_retry(
            lambda: Client(space_id),
            max_wait_s=wait_s,
            label=f"Client({space_id})",
        )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Space baglantisi kurulamadi: {exc}")

    if client is not None:
        try:
            detect_res, detect_s, detect_attempts = _call_with_wakeup_retry(
                lambda: client.predict(
                    image=handle_file(str(img_path)),
                    conf=conf,
                    api_name="/detect_api",
                ),
                max_wait_s=wait_s,
                label="/detect_api",
            )
            if not isinstance(detect_res, dict):
                errors.append(f"/detect_api dict beklenirken {type(detect_res).__name__} dondu.")
            else:
                dets = detect_res.get("detections")
                if not isinstance(dets, list) or len(dets) == 0:
                    errors.append("/detect_api tespit listesi bos dondu.")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"/detect_api hatasi: {exc}")

    if client is not None and isinstance(detect_res, dict) and detect_res.get("detections"):
        det0 = detect_res["detections"][0]
        box_vals = det0.get("box")
        selected_cls_id = int(det0["cls_id"])
        selected_box_str = ",".join(str(float(v)) for v in box_vals)
        try:
            explain_res, explain_s, explain_attempts = _call_with_wakeup_retry(
                lambda: client.predict(
                    image=handle_file(str(img_path)),
                    box=selected_box_str,
                    cls_id=selected_cls_id,
                    api_name="/explain_api",
                ),
                max_wait_s=wait_s,
                label="/explain_api",
            )
            if not isinstance(explain_res, dict):
                errors.append(f"/explain_api dict beklenirken {type(explain_res).__name__} dondu.")
            else:
                # CONTRACT v1.3 dogrulamalari
                if "energy_in_box" not in explain_res:
                    errors.append("/explain_api yanitinda `energy_in_box` alani yok (canli Space henuz CONTRACT v1.3 ile guncellenmemis olabilir).")
                else:
                    ev = explain_res["energy_in_box"]
                    if not isinstance(ev, (int, float)) or isinstance(ev, bool) or not (0.0 <= float(ev) <= 1.0):
                        errors.append(f"energy_in_box [0,1] araliginda sayi degil: {ev!r}")

                if "peak_in_box" not in explain_res:
                    errors.append("/explain_api yanitinda `peak_in_box` alani yok.")
                elif not isinstance(explain_res["peak_in_box"], bool):
                    errors.append(f"peak_in_box bool degil: {explain_res['peak_in_box']!r}")

                if "peak_xy" not in explain_res:
                    errors.append("/explain_api yanitinda `peak_xy` alani yok.")
                else:
                    pxy = explain_res["peak_xy"]
                    if (
                        not isinstance(pxy, (list, tuple))
                        or len(pxy) != 2
                        or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in pxy)
                    ):
                        errors.append(f"peak_xy 2 sayidan olusmuyor: {pxy!r}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"/explain_api hatasi: {exc}")

    overall_ok = len(errors) == 0 and isinstance(detect_res, dict) and isinstance(explain_res, dict)
    _write_report(
        out_path,
        timestamp_str=timestamp_str,
        space_id=space_id,
        img_path=img_path,
        conf=conf,
        overall_ok=overall_ok,
        connect_s=connect_s,
        connect_attempts=connect_attempts,
        detect_s=detect_s,
        detect_attempts=detect_attempts,
        detect_res=detect_res,
        explain_s=explain_s,
        explain_attempts=explain_attempts,
        explain_res=explain_res,
        selected_box_str=selected_box_str,
        selected_cls_id=selected_cls_id,
        errors=errors,
    )
    print(f"[smoke_live] Rapor yazildi: {out_path}")
    print(out_path.read_text(encoding="utf-8"))
    return 0 if overall_ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Canli Hugging Face Space API duman testi")
    parser.add_argument("--space", default=DEFAULT_SPACE, help="Hugging Face Space ID (orn. ebboo/tracera)")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="Test edilecek val goruntusu")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Guven esigi")
    parser.add_argument("--wait-s", type=float, default=90.0, help="Space uyandirma/yeniden deneme suresi (sn)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Markdown rapor cikti yolu")
    args = parser.parse_args()
    return run_smoke(args.space, args.image, args.conf, args.wait_s, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
