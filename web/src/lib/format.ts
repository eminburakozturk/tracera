import i18n from "../i18n";

/**
 * Arayüzde gösterilen sayılar için tek biçimlendirme yardımcısı.
 * TR: ondalık ayırıcı virgül, yüzde işareti başta (%83,1).
 * EN: ondalık ayırıcı nokta, yüzde işareti sonda (83.1%).
 * İndirilen dosyalar (CSV/JSON) ve kod içi hesaplar bu yardımcıyı KULLANMAZ.
 */
function resolveLocale(lang?: string): "tr-TR" | "en-US" {
  const l = (lang ?? i18n.language ?? "tr").toLowerCase();
  return l.startsWith("en") ? "en-US" : "tr-TR";
}

const cache = new Map<string, Intl.NumberFormat>();

function formatter(locale: string, digits: number): Intl.NumberFormat {
  const key = `${locale}|${digits}`;
  let f = cache.get(key);
  if (!f) {
    f = new Intl.NumberFormat(locale, {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
      useGrouping: false,
    });
    cache.set(key, f);
  }
  return f;
}

export function fmtNum(value: number, digits: number, lang?: string): string {
  if (!Number.isFinite(value)) return "—";
  return formatter(resolveLocale(lang), digits).format(value);
}

/** Oran (0–1) → yüzde metni. */
export function fmtPct(fraction: number, digits = 1, lang?: string): string {
  if (!Number.isFinite(fraction)) return "—";
  const n = fmtNum(fraction * 100, digits, lang);
  return resolveLocale(lang) === "tr-TR" ? `%${n}` : `${n}%`;
}

/** Süre (ms) → "36,7 ms". */
export function fmtMs(ms: number, digits = 1, lang?: string): string {
  if (!Number.isFinite(ms)) return "—";
  return `${fmtNum(ms, digits, lang)} ms`;
}

/** Ortalama ± std → "0,858 ± 0,152". */
export function fmtMeanStd(
  mean: number,
  std: number,
  digits = 3,
  lang?: string,
): string {
  return `${fmtNum(mean, digits, lang)} ± ${fmtNum(std, digits, lang)}`;
}
