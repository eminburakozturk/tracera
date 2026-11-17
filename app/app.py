"""Tracera - PCB Hata Tespiti ve Açıklanabilir Yapay Zeka (XAI) Web Demosu.

Gradio tabanlı etkileşimli arayüz:
- Tespit: OpenVINO INT8 @1024 CPU çıkarımı (fallback: PyTorch last.pt)
- XAI: LayerCAM @ Katman 4 ısı haritası ve yakınlaştırılmış bölge (PyTorch CPU)
- Doğrulama örnekleri: SADECE val kartları (05, 11); test kartları (06, 10) kullanılmaz.
- Metrikler: results/*.csv dosyalarından dinamik olarak yüklenir.
"""

from __future__ import annotations

import base64
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    import spaces
except ImportError:
    spaces = None

if spaces is not None:
    @spaces.GPU
    def _zerogpu_dummy():
        """ZeroGPU ortamının donanımı tanıması için boş yer tutucu fonksiyon."""
        return None
else:
    # Yerel ortamda GPU kullanımını sınırla
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import torch
from gradio.data_classes import FileData

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_IOU = 0.50

# HF Space paketi (BASE_DIR == space/) veya repo düzeni için tracera_core kökü
ROOT_CANDIDATE = BASE_DIR if (BASE_DIR / "tracera_core").is_dir() else BASE_DIR.parent
if str(ROOT_CANDIDATE) not in sys.path:
    sys.path.insert(0, str(ROOT_CANDIDATE))

EXAMPLES_DIR = BASE_DIR / "examples"
RESULTS_DIR = (BASE_DIR / "results") if (BASE_DIR / "results").is_dir() else (ROOT_CANDIDATE / "results")

from tracera_core import (  # noqa: E402
    CLASS_COLORS,
    CLASS_DISPLAY_NAMES_TR,
    CLASS_NAMES,
    DEFAULT_CONF,
    DEFAULT_IOU_NMS,
    IMGSZ,
    LOW_CONF_REVIEW_NOTE_TR,
    LOW_CONF_REVIEW_UPPER_BOUND,
    MODELS,
    OPENVINO_MODEL_PATH,
    PYTORCH_MODEL_PATH,
    WEIGHTS_DIR,
    ExplainResult,
    ModelManager,
    MultiLayerYoloCAM,
    box_orig_to_lb,
    detect_openvino,
    explain_layercam_with_metrics,
    letterbox_image,
    make_overlay_and_zoom,
)

# -----------------------------------------------------------------------------
# Dinamik Metrik Yükleyicileri
# -----------------------------------------------------------------------------
def _tr_num(value: float, digits: int) -> str:
    """Arayüz metni için Türkçe ondalık (virgül). API yanıtları bunu kullanmaz."""
    return f"{value:.{digits}f}".replace(".", ",")


def _tr_pct(fraction: float, digits: int = 1) -> str:
    return f"%{_tr_num(fraction * 100.0, digits)}"


def load_dynamic_metrics_markdown() -> str:
    """results/*.csv dosyalarından güncel ve doğrulanmış metrikleri dinamik çeker."""
    op_path = RESULTS_DIR / "operating_point.csv"
    bench_path = RESULTS_DIR / "edge_benchmark.csv"

    md_lines = [
        "### 📊 Raporlanan Doğrulanmış Model Metrikleri (Dinamik Kaynak: `results/*.csv`)",
        "",
    ]

    if op_path.is_file():
        try:
            df_op = pd.read_csv(op_path)
            md_lines.append("#### Çalışma Noktası Performansı (Sabit Eşik = 0.395, IoU = 0.50)")
            md_lines.append("| Veri Kümesi Split | Güven Eşiği | Precision | Recall | F1 Skoru | TP | FP (Toplam) | Arka Plan FP (IoU<0.1) | Konum Hatası FP | FN |")
            md_lines.append("|---|---|---|---|---|---|---|---|---|---|")
            for _, row in df_op.iterrows():
                split_name = str(row.get("split", ""))
                conf = float(row.get("conf_threshold", 0.395))
                p = float(row.get("P", 0.0))
                r = float(row.get("R", 0.0))
                f1 = float(row.get("F1", 0.0))
                tp = int(row.get("TP", 0))
                fp = int(row.get("FP", 0))
                bg_fp = int(row.get("background_FP", 0))
                loc_fp = int(row.get("konum_hatası_FP", 0))
                fn = int(row.get("FN", 0))
                md_lines.append(
                    f"| **{split_name}** | {_tr_num(conf, 3)} | {_tr_pct(p)} | {_tr_pct(r)} | **{_tr_pct(f1)}** | "
                    f"{tp} | {fp} | {bg_fp} | {loc_fp} | {fn} |"
                )
            md_lines.append("")
        except Exception as e:
            md_lines.append(f"*Operating point tablosu yüklenemedi: {e}*")

    if bench_path.is_file():
        try:
            df_b = pd.read_csv(bench_path)
            df_b_1024 = df_b[df_b["imgsz"] == 1024]
            md_lines.append("#### Donanım ve Hız Başarımı (Edge Benchmark @ 1024x1024)")
            md_lines.append("| Format / Çalışma Zamanı | Çıkarım Gecikmesi (Ort) | Uçtan Uca Gecikme (Ort) | FPS (İşlem Dahil) | Donanım |")
            md_lines.append("|---|---|---|---|---|")
            for _, row in df_b_1024.iterrows():
                fmt = str(row.get("format", ""))
                inf_ms = float(row.get("gecikme_inf_ort_ms", 0.0))
                tot_ms = float(row.get("gecikme_toplam_ort_ms", 0.0))
                fps = float(row.get("fps_on_son_islem_dahil", 0.0))
                hw = str(row.get("donanım", ""))
                md_lines.append(f"| {fmt} | {inf_ms:.2f} ms | {tot_ms:.2f} ms | **{fps:.1f}** | {hw} |")
            md_lines.append("")
        except Exception as e:
            md_lines.append(f"*Edge benchmark tablosu yüklenemedi: {e}*")

    md_lines.extend([
        "> [!WARNING] **Kart Tasarımı Limitasyonu ve Güvenilirlik Notu:**",
        "> Model, 10 kartlık PKU-Market-PCB setinin 6 kartında eğitildi (2 kart val, 2 kart test). Yeni ve görülmemiş bir kart tasarımında ",
        "> sabit arka plan özellikleri (ör. üretim şablon artifactları) yanlış alarmlara yol açabilir. ",
        "> Test setinde şablon kaynaklı tekrarlayan sabit noktalar hariç tutulduğunda F1 skoru **%91.13** (Precision %93.69, Recall %88.70) seviyesindedir.",
        "",
        "> [!NOTE] **XAI Açıklanabilirlik Metodolojisi:**",
        "> Açıklamalar için **LayerCAM (Katman 4)** kullanılmaktadır. Katman 4 (backbone stride 8), "
        "daha derin boyun katmanlarına göre çok daha yüksek uzamsal çözünürlük sunar ve "
        "silme testinde GT kutu alanı ölçeğinde en yüksek hassasiyeti doğrulanmıştır.",
    ])

    return "\n".join(md_lines)


# ModelManager / MODELS: Space modunda yerel, repo modunda tracera_core üzerinden sağlanır.


# -----------------------------------------------------------------------------
# Çıkarım ve Çizim Fonksiyonları
# -----------------------------------------------------------------------------
def run_pcb_detection(
    input_img: np.ndarray | None,
    conf_threshold: float,
) -> tuple[np.ndarray | None, pd.DataFrame, list[str], dict, str]:
    """Girdi görüntüsü üzerinde nesne tespiti yapar, kutuları çizer ve tablo döner."""
    if input_img is None:
        return (
            None,
            pd.DataFrame(columns=["#", "Hata Sınıfı", "Güven", "x1", "y1", "x2", "y2", "Not"]),
            [],
            {},
            "Lütfen bir PCB görüntüsü yükleyin veya örneklerden seçin.",
        )

    detector, engine_name = MODELS.get_detector()

    # Giriş görüntüsü RGB formatında gelir (Gradio varsayılanı)
    img_bgr = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
    orig_h, orig_w = img_bgr.shape[:2]

    t0 = time.perf_counter()
    results = detector.predict(
        source=img_bgr,
        imgsz=IMGSZ,
        conf=float(conf_threshold),
        iou=DEFAULT_IOU,
        device="cpu",
        verbose=False,
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    boxes_obj = results[0].boxes
    detections_list: list[dict] = []
    dropdown_choices: list[str] = []
    annotated_rgb = input_img.copy()

    if boxes_obj is not None and len(boxes_obj) > 0:
        xyxy_arr = boxes_obj.xyxy.cpu().numpy()
        conf_arr = boxes_obj.conf.cpu().numpy()
        cls_arr = boxes_obj.cls.cpu().numpy().astype(int)

        for i, (xyxy, conf_val, cls_id) in enumerate(zip(xyxy_arr, conf_arr, cls_arr)):
            x1, y1, x2, y2 = xyxy.tolist()
            cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
            cls_tr = CLASS_DISPLAY_NAMES_TR.get(cls_name, cls_name)
            color = CLASS_COLORS.get(cls_id, (0, 255, 0))
            conf_f = float(conf_val)
            note_tr = LOW_CONF_REVIEW_NOTE_TR if conf_f < LOW_CONF_REVIEW_UPPER_BOUND else ""

            det_item = {
                "idx": i,
                "class_id": int(cls_id),
                "class_name": cls_name,
                "class_name_tr": cls_tr,
                "conf": conf_f,
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "note": note_tr,
            }
            detections_list.append(det_item)

            choice_label = f"#{i} — {cls_tr} (Güven: {_tr_num(float(conf_val), 3)}) @ [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]"
            dropdown_choices.append(choice_label)

            # Görsel üzerine kutu ve etiket çizimi
            cv2.rectangle(
                annotated_rgb,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                thickness=4,
            )
            label_text = f"{cls_name} {conf_val:.2f}"
            t_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(
                annotated_rgb,
                (int(x1), max(0, int(y1) - t_size[1] - 8)),
                (int(x1) + t_size[0] + 6, max(0, int(y1))),
                color,
                -1,
            )
            cv2.putText(
                annotated_rgb,
                label_text,
                (int(x1) + 3, max(t_size[1] + 2, int(y1) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    # DataFrame oluştur
    if detections_list:
        df_display = pd.DataFrame([
            {
                "#": d["idx"],
                "Hata Sınıfı": d["class_name_tr"],
                "Güven": round(d["conf"], 4),
                "x1": int(round(d["box"][0])),
                "y1": int(round(d["box"][1])),
                "x2": int(round(d["box"][2])),
                "y2": int(round(d["box"][3])),
                "Not": d["note"],
            }
            for d in detections_list
        ])
    else:
        df_display = pd.DataFrame(columns=["#", "Hata Sınıfı", "Güven", "x1", "y1", "x2", "y2", "Not"])

    status_str = (
        f"✅ **Tespit Tamamlandı:** {len(detections_list)} kusur tespit edildi | "
        f"**Model:** {engine_name} | **Gecikme:** {latency_ms:.1f} ms (CPU) | "
        f"**Boyut:** {orig_w}x{orig_h}"
    )

    state_data = {
        "img_rgb": input_img,
        "detections": detections_list,
    }

    return annotated_rgb, df_display, dropdown_choices, state_data, status_str


def run_explainability(
    state_data: dict,
    selected_choice: str | None,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """Seçilen tespit için LayerCAM @ Katman 4 ısı haritası ve yakınlaştırılmış görüntüyü üretir."""
    if not state_data or "img_rgb" not in state_data or not state_data.get("detections"):
        return None, None, "⚠️ Açıklanacak aktif tespit bulunamadı. Önce bir görüntü üzerinde tespit çalıştırın."

    if not selected_choice:
        return None, None, "⚠️ Lütfen açılır listeden açıklamak istediğiniz tespiti seçin."

    try:
        idx = int(selected_choice.split("—")[0].replace("#", "").strip())
    except Exception:
        return None, None, f"⚠️ Seçim çözümlenemedi: {selected_choice}"

    detections = state_data["detections"]
    if idx < 0 or idx >= len(detections):
        return None, None, f"⚠️ Geçersiz tespit indeksi: {idx}"

    det = detections[idx]
    img_rgb = state_data["img_rgb"]
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    box_orig = tuple(det["box"])
    label_str = f"{det['class_name']} {det['conf']:.2f}"
    color_rgb = CLASS_COLORS.get(det["class_id"], (0, 255, 100))

    overlay_rgb, zoom_crop_rgb, cam_timing, cam_metrics = explain_layercam_with_metrics(
        img_bgr=img_bgr,
        box_orig=box_orig,
        cls_id=int(det["class_id"]),
        imgsz=IMGSZ,
        label_str=label_str,
        box_color_rgb=color_rgb,
        manager=MODELS,
    )

    cls_tr = det.get("class_name_tr") or CLASS_DISPLAY_NAMES_TR.get(det["class_name"], det["class_name"])
    energy_pct = float(cam_metrics["energy_in_box"]) * 100.0
    peak_loc = "kutunun içinde" if bool(cam_metrics["peak_in_box"]) else "kutunun dışında"

    info_str = (
        f"🎯 **Açıklama Üretildi:** #{idx} {cls_tr} (Güven: {_tr_num(float(det['conf']), 3)}) | "
        f"**Yöntem:** LayerCAM @ Katman 4 (Backbone Stride 8) | "
        f"**Süre:** {cam_timing.total_ms:.1f} ms (PyTorch CPU)\n\n"
        f"Isı haritası enerjisinin kutu içindeki payı: %{_tr_num(energy_pct, 1)}; en sıcak nokta {peak_loc}."
    )

    return overlay_rgb, zoom_crop_rgb, info_str


# -----------------------------------------------------------------------------
# Gradio UI Arayüz Tasarımı
# -----------------------------------------------------------------------------
def build_app() -> gr.Blocks:
    val_examples = [
        [str(EXAMPLES_DIR / "val_05_missing_hole_01.jpg"), DEFAULT_CONF],
        [str(EXAMPLES_DIR / "val_05_short_01.jpg"), DEFAULT_CONF],
        [str(EXAMPLES_DIR / "val_11_mouse_bite_01.jpg"), DEFAULT_CONF],
        [str(EXAMPLES_DIR / "val_11_spur_01.jpg"), DEFAULT_CONF],
    ]

    with gr.Blocks(title="Tracera - PCB Hata Tespiti & XAI") as demo:
        gr.Markdown(
            "# 🔍 Tracera: PCB Hata Tespiti ve Açıklanabilir Yapay Zeka (XAI)\n"
            "Baskı devre kartlarında (PCB) mikron düzeyindeki 6 sınıf kusuru tespit eder ve "
            "**LayerCAM (Katman 4)** ile modelin karar gerekçesini şeffaf biçimde görselleştirir."
        )

        state_holder = gr.State(value={})

        with gr.Row():
            with gr.Column(scale=5):
                input_image = gr.Image(label="Girdi PCB Görüntüsü", type="numpy")
                conf_slider = gr.Slider(
                    minimum=0.05,
                    maximum=1.0,
                    value=DEFAULT_CONF,
                    step=0.005,
                    label="Güven Eşiği (Confidence Threshold)",
                )
                gr.Markdown("*Not: Eşik val setinde F1-maks ile seçildi (0.395); raporlanan metrikler bu eşikle elde edilmiştir.*")
                detect_btn = gr.Button("🔎 Kusurları Tespit Et", variant="primary")

                gr.Examples(
                    examples=val_examples,
                    inputs=[input_image, conf_slider],
                    label="Doğrulama (Val) Kümesi Örnekleri (Kart 05 ve Kart 11)",
                )

            with gr.Column(scale=6):
                status_box = gr.Markdown("Görüntü yükleyin ve **Kusurları Tespit Et** butonuna basın.")
                annotated_output = gr.Image(label="Tespit Edilen Kusurlar (Bounding Boxes)", type="numpy")
                table_output = gr.Dataframe(
                    headers=["#", "Hata Sınıfı", "Güven", "x1", "y1", "x2", "y2", "Not"],
                    label="Tespit Listesi",
                    interactive=False,
                )

        gr.Markdown("---")
        gr.Markdown(
            "## 💡 Model Kararını Açıkla (XAI — LayerCAM)\n"
            "Aşağıdaki listeden incelemek istediğiniz tespiti seçerek modelin PCB üzerindeki hangi mikro dokulara odaklandığını görün."
        )

        with gr.Row():
            with gr.Column(scale=4):
                choice_dropdown = gr.Dropdown(
                    label="Açıklanacak Tespiti Seçin",
                    choices=[],
                    interactive=True,
                )
                explain_btn = gr.Button("🔥 Isı Haritası Üret (LayerCAM @ Katman 4)", variant="secondary")
                cam_status_box = gr.Markdown("Tespit seçildikten sonra butona basarak ısı haritası üretebilirsiniz.")
                zoom_output = gr.Image(label="Kusur Bölgesi (4x Yakınlaştırma)", type="numpy")

            with gr.Column(scale=7):
                cam_overlay_output = gr.Image(label="Tam PCB Isı Haritası (LayerCAM Overlay)", type="numpy")

        gr.Markdown("---")
        gr.Markdown(load_dynamic_metrics_markdown())

        # Olay bağlamaları
        def on_detect_click(img, conf):
            ann_img, df_res, choices, state, st_str = run_pcb_detection(img, conf)
            initial_choice = choices[0] if choices else None
            return ann_img, df_res, gr.update(choices=choices, value=initial_choice), state, st_str

        detect_btn.click(
            fn=on_detect_click,
            inputs=[input_image, conf_slider],
            outputs=[annotated_output, table_output, choice_dropdown, state_holder, status_box],
        )

        explain_btn.click(
            fn=run_explainability,
            inputs=[state_holder, choice_dropdown],
            outputs=[cam_overlay_output, zoom_output, cam_status_box],
        )

        gr.api(detect_api, api_name="detect_api")
        gr.api(explain_api, api_name="explain_api")

    return demo


def _decode_api_image_bgr(image: FileData | dict[str, Any] | str) -> np.ndarray:
    """Gradio API ucuna gelen FileData/yol/base64 görüntüyü BGR ndarray'e çözer."""
    path_str: str | None = None
    if isinstance(image, FileData):
        path_str = image.path
    elif isinstance(image, dict):
        path_str = image.get("path") or image.get("url")
    elif isinstance(image, str):
        path_str = image

    if not path_str:
        raise gr.Error("Geçerli bir görüntü dosyası gönderilmedi.")

    if path_str.startswith("data:image") and "," in path_str:
        b64_payload = path_str.split(",", 1)[1]
        raw = base64.b64decode(b64_payload)
        img_bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    else:
        img_bgr = cv2.imread(path_str)

    if img_bgr is None or img_bgr.size == 0:
        raise gr.Error("Görüntü okunamadı; bozuk veya desteklenmeyen biçim.")
    return img_bgr


def _parse_api_box(box: str, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    parts = [p.strip() for p in box.split(",")]
    if len(parts) != 4:
        raise gr.Error("box alanı 'x1,y1,x2,y2' biçiminde 4 sayı olmalıdır.")
    try:
        vals = [float(p) for p in parts]
    except ValueError as exc:
        raise gr.Error("box koordinatları geçerli sayı olmalıdır.") from exc
    if any(math.isnan(v) or math.isinf(v) for v in vals):
        raise gr.Error("box koordinatları sonlu olmalıdır.")
    x1, y1, x2, y2 = vals
    x1 = max(0.0, min(x1, float(img_w)))
    y1 = max(0.0, min(y1, float(img_h)))
    x2 = max(0.0, min(x2, float(img_w)))
    y2 = max(0.0, min(y2, float(img_h)))
    if x2 <= x1 or y2 <= y1:
        raise gr.Error("box koordinatları geçersiz (x2 > x1 ve y2 > y1 olmalı).")
    return (x1, y1, x2, y2)


def _encode_rgb_png_base64(rgb_img: np.ndarray) -> str:
    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr_img)
    if not ok:
        raise gr.Error("PNG kodlaması başarısız oldu.")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def detect_api(image: FileData, conf: float = DEFAULT_CONF) -> dict[str, Any]:
    """CONTRACT §2 DetectResult JSON döndüren arayüzsüz Gradio API ucu."""
    conf_val = float(conf)
    if math.isnan(conf_val) or math.isinf(conf_val) or not (0.0 < conf_val <= 1.0):
        raise gr.Error("conf parametresi (0, 1] aralığında olmalıdır.")
    img_bgr = _decode_api_image_bgr(image)
    res = detect_openvino(
        img_bgr,
        conf=conf_val,
        iou_nms=DEFAULT_IOU_NMS,
        imgsz=IMGSZ,
        manager=MODELS,
    )
    return res.model_dump()


def explain_api(image: FileData, box: str, cls_id: int) -> dict[str, Any]:
    """CONTRACT §2 ve §7 ExplainResult JSON döndüren arayüzsüz Gradio API ucu (LayerCAM @ katman 4)."""
    cls_int = int(cls_id)
    if cls_int not in CLASS_NAMES:
        raise gr.Error(f"Geçersiz cls_id={cls_int}; beklenen aralık: 0..{len(CLASS_NAMES) - 1}.")
    img_bgr = _decode_api_image_bgr(image)
    h, w = img_bgr.shape[:2]
    box_orig = _parse_api_box(box, img_w=w, img_h=h)
    overlay_rgb, zoom_rgb, timing, cam_metrics = explain_layercam_with_metrics(
        img_bgr=img_bgr,
        box_orig=box_orig,
        cls_id=cls_int,
        imgsz=IMGSZ,
        manager=MODELS,
    )
    return ExplainResult(
        overlay_png=_encode_rgb_png_base64(overlay_rgb),
        zoom_png=_encode_rgb_png_base64(zoom_rgb),
        method="layercam",
        layer=4,
        timing=timing,
        energy_in_box=float(cam_metrics["energy_in_box"]),
        peak_in_box=bool(cam_metrics["peak_in_box"]),
        peak_xy=list(cam_metrics["peak_xy"]),
    ).model_dump()


if __name__ == "__main__":
    app = build_app()
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    app.launch(server_name=server_name, server_port=server_port, share=False)
