import type { LangCode } from "../store/useAppStore";

export const CLASS_DICTIONARY: Record<
  string,
  {
    tr: string;
    en: string;
    shortTr: string;
    descriptionTr: string;
    descriptionEn: string;
    color: string;
  }
> = {
  missing_hole: {
    tr: "Eksik delik",
    en: "Missing hole",
    shortTr: "Eksik delik",
    descriptionTr:
      "Kartta bulunması gereken bir deliğin (bileşen ya da geçiş deliği) olmaması.",
    descriptionEn:
      "Absence of a hole (component or via hole) that should be present on the board.",
    color: "#38bdf8",
  },
  mouse_bite: {
    tr: "Fare ısırığı (kenar çentiği)",
    en: "Mouse bite",
    shortTr: "Fare ısırığı",
    descriptionTr:
      "İletken yolun kenarında küçük, düzensiz çentikler; yol yerel olarak incelir.",
    descriptionEn:
      "Small irregular notches along the edge of a conductive trace, locally narrowing the path.",
    color: "#f472b6",
  },
  open_circuit: {
    tr: "Açık devre",
    en: "Open circuit",
    shortTr: "Açık devre",
    descriptionTr:
      "İletken yolun kopması; iki nokta arasındaki elektriksel bağlantı kesilir.",
    descriptionEn:
      "Break in a conductive trace, interrupting the electrical connection between two points.",
    color: "#facc15",
  },
  short: {
    tr: "Kısa devre",
    en: "Short",
    shortTr: "Kısa devre",
    descriptionTr:
      "Ayrı kalması gereken iki iletken arasında istenmeyen bakır bağlantı.",
    descriptionEn:
      "Unintended copper bridge between two conductive paths that should remain separate.",
    color: "#f87171",
  },
  spur: {
    tr: "Çıkıntı",
    en: "Spur",
    shortTr: "Çıkıntı",
    descriptionTr:
      "İletken yolun kenarından dışarı taşan ince, sivri bakır parçası.",
    descriptionEn:
      "Thin, pointed copper projection extending outward from the edge of a conductive trace.",
    color: "#a78bfa",
  },
  spurious_copper: {
    tr: "İstenmeyen bakır",
    en: "Spurious copper",
    shortTr: "İstenmeyen bakır",
    descriptionTr:
      "Bakır olmaması gereken bir bölgede kalmış bakır parçası.",
    descriptionEn:
      "Residual copper left in an area where no copper should be present.",
    color: "#34d399",
  },
};

export function getClassDisplayName(
  cls: string,
  lang: LangCode = "tr",
  short = false,
): string {
  const entry = CLASS_DICTIONARY[cls];
  if (!entry) return cls;
  if (lang === "en") return entry.en;
  return short ? entry.shortTr : entry.tr;
}

export function getClassDescription(cls: string, lang: LangCode = "tr"): string {
  const entry = CLASS_DICTIONARY[cls];
  if (!entry) return "";
  return lang === "en" ? entry.descriptionEn : entry.descriptionTr;
}

export function getClassColor(cls: string): string {
  return CLASS_DICTIONARY[cls]?.color ?? "#38bdf8";
}

export const GUIDE_DATASET_NOTE_TR =
  "Bu veri setinde kusurlar, kusursuz kart fotoğraflarına sentetik olarak eklenmiştir (PKU-Market-PCB). Gerçek üretim hattı görüntülerinde performans farklı olabilir.";

export const GUIDE_DATASET_NOTE_EN =
  "In this dataset, defects were synthetically added onto defect-free board photographs (PKU-Market-PCB). Performance on real production line images may differ.";

export const VERDICT_RULE_NOTE_TR =
  "Karar kuralı: eşik üstünde ≥1 kusur = KALDI; bu bir yazılım kararıdır, standart değildir.";

export const VERDICT_RULE_NOTE_EN =
  "Decision rule: ≥1 defect above threshold = FAIL; this is a software rule, not an industry standard.";

export const PDF_MANDATORY_FOOTER_TR =
  "Bu rapor Tracera yazılımının otomatik çıktısıdır; herhangi bir standarda uygunluk belgesi değildir. Karar kuralı: güven eşiği (varsayılan 0,395) üzerinde en az bir kusur → KALDI.";

export const LOW_CONF_REVIEW_UPPER_BOUND = 0.6;
export const LOW_CONF_REVIEW_DEFAULT_MIN = 0.395;

export const LOW_CONF_REVIEW_LABEL_TR = "Düşük güven: gözden geçirin";
export const LOW_CONF_REVIEW_LABEL_EN = "Low confidence: review";

export const LOW_CONF_REVIEW_TOOLTIP_TR =
  "Test setinde bu güven aralığındaki tespitlerin çoğu yanlış pozitifti; bu etiket değerlendirmeden sonra eklendi.";
export const LOW_CONF_REVIEW_TOOLTIP_EN =
  "Most detections in this confidence range on the test set were false positives; this label was added post-evaluation.";

export function isLowConfidenceReview(
  conf: number,
  minThreshold: number = LOW_CONF_REVIEW_DEFAULT_MIN,
): boolean {
  const effectiveMin = Math.max(minThreshold, LOW_CONF_REVIEW_DEFAULT_MIN);
  return conf >= effectiveMin && conf < LOW_CONF_REVIEW_UPPER_BOUND;
}
