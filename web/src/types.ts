export interface ModelMeta {
  version: string;
  model: string;
  imgsz: number;
  conf: number;
  iou_nms: number;
  letterbox_pad: number;
  classes: string[];
  colors: string[];
  onnx: string;
}

// CONTRACT §2 — TypeScript ve Pydantic ile birebir aynı alan adları
export interface Detection {
  id: number;
  cls: string;
  cls_id: number;
  conf: number;
  box: [number, number, number, number]; // [x1, y1, x2, y2] orijinal piksel
}

export interface Timing {
  pre_ms: number;
  model_ms: number;
  post_ms: number;
  total_ms: number;
}

export interface DetectResult {
  width: number;
  height: number;
  detections: Detection[];
  timing: Timing;
  backend: "ort-webgpu" | "ort-wasm" | "openvino-int8" | "pytorch" | string;
}

export interface ExplainResult {
  overlay_png: string; // ham base64 PNG (data URL değil)
  zoom_png: string; // ham base64 PNG (data URL değil)
  method: "layercam";
  layer: 4;
  timing: Timing;
}

export interface HealthResponse {
  status: string;
  model: string;
  version: string;
  device: string;
  backends: string[];
}

export interface OperatingPointRow {
  split: string;
  conf_threshold: number;
  iou_threshold: number;
  TP: number;
  FP: number;
  background_FP: number;
  konum_hatası_FP: number;
  duplicate_FP: number;
  cross_class_FP: number;
  FN: number;
  background_FN: number;
  P: number;
  R: number;
  F1: number;
  P_macro: number;
  R_macro: number;
  F1_macro: number;
  [key: string]: string | number;
}

export interface EdgeBenchmarkRow {
  model: string;
  imgsz: number;
  format: string;
  boyut_mb: number;
  gecikme_inf_ort_ms: number;
  gecikme_inf_medyan_ms: number;
  gecikme_inf_p95_ms: number;
  gecikme_toplam_ort_ms: number;
  gecikme_toplam_medyan_ms: number;
  gecikme_toplam_p95_ms: number;
  fps_sadece_inference: number;
  fps_on_son_islem_dahil: number;
  mAP50: number;
  "mAP50-95": number;
  donanım: string;
  güç: string;
  cpu_threads: number;
  ultralytics_ver: string;
  torch_ver: string;
  onnxruntime_ver: string;
  openvino_ver: string;
}

export interface CvFoldSummary {
  n_folds: number;
  mAP50_mean: number;
  mAP50_std: number;
  mAP50_95_mean: number;
  mAP50_95_std: number;
  imgsz?: number;
  weight?: string;
}

export interface XaiRow {
  model: string;
  imgsz: number;
  katman: number;
  method: string;
  n_tespit: number;
  kendi_kutu_tum_goruntu: number;
  pointing_rastgele_model: number;
  şans_kendi_kutu_tum_goruntu: number;
  herhangi_kutu_tum_goruntu: number;
  şans_herhangi_kutu_tum_goruntu: number;
  randomizasyon_spearman_medyan_tum_goruntu: number;
  pencere_alan_orani: number;
  randomizasyon_spearman_medyan_pencere: number;
  enerji_kutuda_pencere: number;
  taban_kutu_pencere: number;
  goreli_dusus_harita_ort_1x: number | null;
  goreli_dusus_harita_med_1x: number | null;
  goreli_dusus_kontrol_ort_1x: number | null;
  goreli_dusus_kontrol_med_1x: number | null;
  goreli_kazanma_orani_1x: number | null;
  goreli_dusus_harita_ort_2x: number | null;
  goreli_dusus_harita_med_2x: number | null;
  goreli_dusus_kontrol_ort_2x: number | null;
  goreli_dusus_kontrol_med_2x: number | null;
  goreli_kazanma_orani_2x: number | null;
}

export interface AoiSeedSummary {
  n_seeds: number;
  mAP50_mean: number;
  mAP50_std: number;
  mAP50_95_mean: number;
  mAP50_95_std: number;
  Bad_qiaojiao_mAP50_mean: number;
  Bad_podu_mAP50_mean: number;
}

export interface WebSpikeRow {
  backend: string;
  model: string;
  boyut_mb: number;
  medyan_ms: number | null;
  eşleşme_oranı: number | null;
  ort_iou: number | null;
  ort_dconf: number | null;
}

export interface MetricsPayload {
  operating_point: OperatingPointRow[];
  edge_benchmark: EdgeBenchmarkRow[];
  cv: {
    leakage_rows: Array<Record<string, string | number>>;
    leakage_summary: {
      random: CvFoldSummary;
      grouped: CvFoldSummary;
    };
    resolution_rows: Array<Record<string, string | number>>;
    resolution_summary: Record<string, CvFoldSummary>;
  };
  xai: XaiRow[];
  aoi: {
    rows: Array<Record<string, string | number>>;
    test_summary: {
      coco: AoiSeedSummary;
      pku: AoiSeedSummary;
    };
  };
  final_val?: Array<Record<string, string | number>>;
  final_test?: Array<Record<string, string | number>>;
  web_spike?: WebSpikeRow[];
}
