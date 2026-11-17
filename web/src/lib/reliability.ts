import type { LangCode } from "../store/useAppStore";
import type { ReliabilityBin, ReliabilityData } from "../types";
import { fmtNum } from "./format";

export interface SplitReliabilityEntry {
  split: "val" | "test";
  label: "Val" | "Test";
  binRange: string;
  precision: number;
  precisionPct: string;
  n: number;
  lowSample: boolean;
  shortText: string;
}

export interface ReliabilityComparisonSummary {
  binRange: string;
  valPart: SplitReliabilityEntry | null;
  testPart: SplitReliabilityEntry | null;
  singleSplit: "val" | "test" | null;
  headlineText: string;
}

export function findReliabilityBin(
  data: ReliabilityData | null | undefined,
  conf: number,
): ReliabilityBin | null {
  if (!data || !Array.isArray(data.bins) || data.bins.length === 0) {
    return null;
  }
  for (let i = 0; i < data.bins.length; i++) {
    const b = data.bins[i];
    const isLast = i === data.bins.length - 1;
    if (conf >= b.min && (conf < b.max || (isLast && conf <= b.max))) {
      return b;
    }
  }
  return null;
}

export function formatSplitReliability(
  split: "val" | "test",
  bin: ReliabilityBin | null | undefined,
  lang: LangCode = "tr",
): SplitReliabilityEntry | null {
  if (
    !bin ||
    bin.n <= 0 ||
    bin.precision === null ||
    bin.precision === undefined ||
    Number.isNaN(bin.precision)
  ) {
    return null;
  }
  const label = split === "val" ? "Val" : "Test";
  const decimals = (v: number) => (String(v).split(".")[1] ?? "").length;
  const binRange = `${fmtNum(bin.min, decimals(bin.min), lang)}–${fmtNum(bin.max, decimals(bin.max), lang)}`;
  const precisionPct = fmtNum(bin.precision * 100, 1, lang);
  const lowSample = bin.n < 20;
  const shortText =
    lang === "en"
      ? `${label}: ${precisionPct}% (n=${bin.n})`
      : `${label}: %${precisionPct} (n=${bin.n})`;

  return {
    split,
    label,
    binRange,
    precision: bin.precision,
    precisionPct,
    n: bin.n,
    lowSample,
    shortText,
  };
}

export function buildReliabilitySummary(
  conf: number,
  valData: ReliabilityData | null | undefined,
  testData: ReliabilityData | null | undefined,
  lang: LangCode = "tr",
): ReliabilityComparisonSummary | null {
  const valBin = findReliabilityBin(valData, conf);
  const testBin = findReliabilityBin(testData, conf);

  const valPart = formatSplitReliability("val", valBin, lang);
  const testPart = formatSplitReliability("test", testBin, lang);

  if (!valPart && !testPart) {
    return null;
  }

  if (valPart && testPart) {
    const headlineText = `${valPart.shortText} · ${testPart.shortText}`;
    return {
      binRange: valPart.binRange,
      valPart,
      testPart,
      singleSplit: null,
      headlineText,
    };
  }

  const single = valPart ?? testPart!;
  const splitNameTr = single.split === "val" ? "Val setinde" : "Test setinde";
  const splitNameEn =
    single.split === "val" ? "In the Val split" : "In the Test split";

  const headlineText =
    lang === "en"
      ? `${splitNameEn}, ${single.precisionPct}% of detections in this confidence bin (${single.binRange}) were correct (n=${single.n}).`
      : `${splitNameTr} bu güven aralığında (${single.binRange}) doğru tespit oranı: %${single.precisionPct} (n=${single.n}).`;

  return {
    binRange: single.binRange,
    valPart,
    testPart,
    singleSplit: single.split,
    headlineText,
  };
}
