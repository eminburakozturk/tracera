import * as React from "react";
import { Check, Copy, Gauge } from "lucide-react";
import { useTranslation } from "react-i18next";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { loadAnalysis } from "../lib/analysis";
import {
  type DeviceBenchmarkReport,
  runDeviceBenchmark,
} from "../lib/deviceBenchmark";
import { getClassDisplayName } from "../lib/v11Rules";
import { useAppStore } from "../store/useAppStore";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./ui/card";
import { fmtMs, fmtNum as sharedFmtNum, fmtPct as sharedFmtPct } from "../lib/format";

function fmtPct(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return sharedFmtPct(v, 1);
}

function fmtNum(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return sharedFmtNum(v, digits);
}

const OP_COLUMNS = [
  { tr: "Küme", en: "Split", trTitle: "Veri kümesi", enTitle: "Data split" },
  { tr: "DP", en: "TP", trTitle: "Doğru pozitif", enTitle: "True positives" },
  { tr: "YP", en: "FP", trTitle: "Yanlış pozitif (toplam)", enTitle: "False positives (total)" },
  { tr: "Arka plan YP", en: "Bg FP", trTitle: "Hiçbir kusurla örtüşmüyor (IoU < 0,1)", enTitle: "No overlap with any defect (IoU < 0.1)" },
  { tr: "Konum hatası", en: "Loc FP", trTitle: "Kutu kaymış (0,1 ≤ IoU ≤ 0,5)", enTitle: "Box misplaced (0.1 ≤ IoU ≤ 0.5)" },
  { tr: "Tekrar", en: "Dup FP", trTitle: "Aynı kusura ikinci kutu (IoU > 0,5)", enTitle: "Second box on same defect (IoU > 0.5)" },
  { tr: "Sınıf karışması", en: "Cross FP", trTitle: "Doğru yer, yanlış sınıf (IoU > 0,5)", enTitle: "Right place, wrong class (IoU > 0.5)" },
  { tr: "YN", en: "FN", trTitle: "Kaçırılan kusur", enTitle: "Missed defects" },
  { tr: "P", en: "P", trTitle: "Kesinlik", enTitle: "Precision" },
  { tr: "R", en: "R", trTitle: "Duyarlılık", enTitle: "Recall" },
  { tr: "F1", en: "F1", trTitle: "F1 skoru", enTitle: "F1 score" },
] as const;

function splitLabel(split: string, lang: string): string {
  const en = lang === "en";
  switch (split) {
    case "val":
      return en ? "Validation" : "Doğrulama";
    case "test":
      return en ? "Test" : "Test";
    case "test_sablon_noktalari_haric":
      return en ? "Test (2 template points excluded)" : "Test (2 şablon noktası hariç)";
    default:
      return split;
  }
}

export function ResultsView() {
  const { t } = useTranslation();
  const lang = useAppStore((s) => s.lang);
  const metrics = useAppStore((s) => s.metrics);
  const metricsSource = useAppStore((s) => s.metricsSource);
  const meta = useAppStore((s) => s.meta);
  const analysis = useAppStore((s) => s.analysis);
  const setAnalysis = useAppStore((s) => s.setAnalysis);

  const [benchRunning, setBenchRunning] = React.useState(false);
  const [benchProgress, setBenchProgress] = React.useState<{
    done: number;
    total: number;
  }>({ done: 0, total: 21 });
  const [benchReport, setBenchReport] =
    React.useState<DeviceBenchmarkReport | null>(null);
  const [benchError, setBenchError] = React.useState<string | null>(null);
  const [benchCopied, setBenchCopied] = React.useState(false);
  const abortedRef = React.useRef(false);

  React.useEffect(() => {
    abortedRef.current = false;
    return () => {
      abortedRef.current = true;
    };
  }, []);

  const handleRunDeviceBenchmark = React.useCallback(async () => {
    if (benchRunning) return;
    setBenchRunning(true);
    setBenchError(null);
    setBenchCopied(false);
    setBenchProgress({ done: 0, total: 21 });
    try {
      const report = await runDeviceBenchmark({
        meta,
        repeats: 5,
        onProgress: (done, total) => setBenchProgress({ done, total }),
        shouldAbort: () => abortedRef.current,
      });
      if (!abortedRef.current) {
        setBenchReport(report);
      }
    } catch (err) {
      if (!abortedRef.current) {
        setBenchError(err instanceof Error ? err.message : String(err));
      }
    } finally {
      if (!abortedRef.current) {
        setBenchRunning(false);
      }
    }
  }, [benchRunning, meta]);

  const handleCopyBenchJson = React.useCallback(async () => {
    if (!benchReport) return;
    try {
      await navigator.clipboard.writeText(
        JSON.stringify(benchReport, null, 2),
      );
      setBenchCopied(true);
      setTimeout(() => setBenchCopied(false), 2000);
    } catch (err) {
      setBenchError(err instanceof Error ? err.message : String(err));
    }
  }, [benchReport]);

  React.useEffect(() => {
    if (analysis) return;
    let cancelled = false;
    void loadAnalysis().then((loaded) => {
      if (!cancelled && loaded) {
        setAnalysis(loaded);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [analysis, setAnalysis]);

  if (!metrics) {
    return (
      <Card>
        <CardContent className="py-8 text-center text-sm text-[var(--text-muted)]">
          Loading metrics...
        </CardContent>
      </Card>
    );
  }

  const cv1024 = metrics.cv.resolution_summary["1024_last"];
  const cv640 = metrics.cv.resolution_summary["640_last"];
  const leakRand = metrics.cv.leakage_summary.random;
  const leakGroup = metrics.cv.leakage_summary.grouped;
  const testRow = metrics.final_test?.[0];
  const testMap50 =
    typeof testRow?.mAP50 === "number" ? testRow.mAP50 : null;
  const testMap5095 =
    typeof testRow?.["mAP50-95"] === "number" ? testRow["mAP50-95"] : null;

  const opTest = metrics.operating_point.find((r) => r.split === "test");
  const opSens = metrics.operating_point.find(
    (r) => r.split === "test_sablon_noktalari_haric",
  );
  const firstOp = metrics.operating_point[0];
  const xai1024 = metrics.xai.filter((r) => r.imgsz === 1024);

  const cvChartData = [
    {
      name: lang === "en" ? "Random (640)" : "Rastgele (640)",
      mAP50: Number((leakRand?.mAP50_mean ?? 0).toFixed(3)),
      mAP50_95: Number((leakRand?.mAP50_95_mean ?? 0).toFixed(3)),
    },
    {
      name: lang === "en" ? "Grouped (640)" : "Kart Bazlı (640)",
      mAP50: Number((cv640?.mAP50_mean ?? leakGroup?.mAP50_mean ?? 0).toFixed(3)),
      mAP50_95: Number(
        (cv640?.mAP50_95_mean ?? leakGroup?.mAP50_95_mean ?? 0).toFixed(3),
      ),
    },
    {
      name: lang === "en" ? "Grouped (1024)" : "Kart Bazlı (1024)",
      mAP50: Number((cv1024?.mAP50_mean ?? 0).toFixed(3)),
      mAP50_95: Number((cv1024?.mAP50_95_mean ?? 0).toFixed(3)),
    },
  ];

  const platformSpeedData = [
    ...metrics.edge_benchmark
      .filter((r) => r.imgsz === 1024)
      .map((r) => ({
        platform: `${r.format} (Desktop)`,
        ms: Number(r.gecikme_toplam_ort_ms.toFixed(1)),
      })),
    ...(metrics.web_spike ?? [])
      .filter((w) => w.medyan_ms !== null)
      .map((w) => ({
        platform: `Web ${w.backend}`,
        ms: Number((w.medyan_ms ?? 0).toFixed(1)),
      })),
    {
      platform: "Android SM-S948B (WebGPU)",
      ms: 287.1,
    },
    {
      platform: "Android SM-S948B (WASM)",
      ms: 458.6,
    },
  ];

  return (
    <div className="space-y-5" data-testid="results-view">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h2 className="text-xl font-semibold tracking-tight text-[var(--text-primary)]">
            {t("results.title")}
          </h2>
          <p className="text-xs text-[var(--text-muted)]">
            {t("results.subtitle")}
          </p>
        </div>
        {metricsSource && (
          <Badge variant="secondary" className="font-mono">
            {t("results.sourceLabel")}: {metricsSource}
          </Badge>
        )}
      </div>

      {/* 1. 5-Katlı CV Sızıntı & Çözünürlük (640 vs 1024) */}
      <Card>
        <CardHeader>
          <CardTitle>{t("results.cvTitle")}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-4">
            <div className="elevated-card rounded-lg p-3.5">
              <p className="text-xs text-[var(--text-muted)]">
                {t("results.cvSummaryGrouped1024")}
              </p>
              <p className="mt-1 font-mono text-xl font-semibold text-[var(--accent-ink)] tabular-nums">
                {fmtNum(cv1024?.mAP50_mean)} ± {fmtNum(cv1024?.mAP50_std)}
              </p>
              <p className="mt-0.5 font-mono text-xs text-[var(--text-secondary)] tabular-nums">
                mAP50-95: {fmtNum(cv1024?.mAP50_95_mean)} ±{" "}
                {fmtNum(cv1024?.mAP50_95_std)}
              </p>
            </div>
            <div className="elevated-card rounded-lg p-3.5">
              <p className="text-xs text-[var(--text-muted)]">
                {t("results.cvSummaryGrouped640")}
              </p>
              <p className="mt-1 font-mono text-xl font-semibold text-[var(--text-primary)] tabular-nums">
                {fmtNum(cv640?.mAP50_mean)} ± {fmtNum(cv640?.mAP50_std)}
              </p>
              <p className="mt-0.5 font-mono text-xs text-[var(--text-secondary)] tabular-nums">
                mAP50-95: {fmtNum(cv640?.mAP50_95_mean)} ±{" "}
                {fmtNum(cv640?.mAP50_95_std)}
              </p>
            </div>
            <div className="elevated-card rounded-lg p-3.5">
              <p className="text-xs text-[var(--text-muted)]">
                {t("results.cvLeakageRandom")}
              </p>
              <p className="mt-1 font-mono text-xl font-semibold text-[var(--text-primary)] tabular-nums">
                {fmtNum(leakRand.mAP50_mean)} ± {fmtNum(leakRand.mAP50_std)}
              </p>
              <p className="mt-0.5 font-mono text-xs text-[var(--text-secondary)] tabular-nums">
                mAP50-95: {fmtNum(leakRand.mAP50_95_mean)} ±{" "}
                {fmtNum(leakRand.mAP50_95_std)}
              </p>
            </div>
            <div className="elevated-card rounded-lg p-3.5">
              <p className="text-xs text-[var(--text-muted)]">
                {t("results.cvLeakageGrouped")}
              </p>
              <p className="mt-1 font-mono text-xl font-semibold text-[var(--text-primary)] tabular-nums">
                {fmtNum(leakGroup.mAP50_mean)} ± {fmtNum(leakGroup.mAP50_std)}
              </p>
              <p className="mt-0.5 font-mono text-xs text-[var(--text-secondary)] tabular-nums">
                mAP50-95: {fmtNum(leakGroup.mAP50_95_mean)} ±{" "}
                {fmtNum(leakGroup.mAP50_95_std)}
              </p>
            </div>
          </div>

          {/* Recharts Sızıntı & Çözünürlük Grafiği */}
          <div
            className="h-56 w-full rounded-xl border border-[var(--border-hairline)] bg-[var(--bg-elevated)] p-3"
            data-testid="results-cv-chart"
          >
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={cvChartData}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="mAP50" fill="#38bdf8" radius={[4, 4, 0, 0]} />
                <Bar dataKey="mAP50_95" fill="#a78bfa" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* 2. Kilitli Test ve Çalışma Noktası */}
      <Card>
        <CardHeader>
          <CardTitle>
            {t("results.testTitle", {
              conf: firstOp ? firstOp.conf_threshold : 0.395,
              iou: firstOp ? firstOp.iou_threshold : 0.5,
            })}
          </CardTitle>
          {testMap50 !== null && (
            <CardDescription className="font-mono text-xs text-[var(--text-primary)] tabular-nums">
              {t("results.lockedTestMap")}: mAP50 = {fmtNum(testMap50, 4)} ·
              mAP50-95 = {fmtNum(testMap5095, 4)}
            </CardDescription>
          )}
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-left font-mono text-xs tabular-nums">
              <thead>
                <tr className="border-b border-[var(--border-hairline)] text-[var(--text-muted)]">
                  {OP_COLUMNS.map((c, i) => (
                    <th
                      key={c.en}
                      title={lang === "en" ? c.enTitle : c.trTitle}
                      className={i === OP_COLUMNS.length - 1 ? "py-2" : "py-2 pr-3"}
                    >
                      {lang === "en" ? c.en : c.tr}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {metrics.operating_point.map((row) => {
                  const isSens = row.split === "test_sablon_noktalari_haric";
                  return (
                    <tr
                      key={row.split}
                      className="border-b border-[var(--border-hairline)]/60 text-[var(--text-primary)]"
                    >
                      <td className="py-2 pr-3 font-semibold">
                        {splitLabel(row.split, lang)}
                        {isSens && (
                          <Badge variant="warning" className="ml-2">
                            post-hoc
                          </Badge>
                        )}
                      </td>
                      <td className="py-2 pr-3">{row.TP}</td>
                      <td className="py-2 pr-3">{row.FP}</td>
                      <td className="py-2 pr-3">{row.background_FP}</td>
                      <td className="py-2 pr-3">{row.konum_hatası_FP}</td>
                      <td className="py-2 pr-3">{row.duplicate_FP}</td>
                      <td className="py-2 pr-3">{row.cross_class_FP}</td>
                      <td className="py-2 pr-3">{row.FN}</td>
                      <td className="py-2 pr-3">{fmtNum(row.P, 4)}</td>
                      <td className="py-2 pr-3">{fmtNum(row.R, 4)}</td>
                      <td className="py-2 font-semibold text-[var(--accent-ink)]">
                        {fmtNum(row.F1, 4)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {opTest && opSens && (
            <p className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-xs text-[var(--text-primary)]">
              {t("results.sensitivityNote", {
                fpBase: opTest.FP,
                fpSens: opSens.FP,
              })}
            </p>
          )}
        </CardContent>
      </Card>

      {/* 3. Platform Hızları (edge_benchmark, tarayıcı, masaüstü, Android) */}
      <Card>
        <CardHeader>
          <CardTitle>{t("results.edgeTitle")}</CardTitle>
          <CardDescription>{t("results.edgeHardwareNote")}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div
            className="h-56 w-full rounded-xl border border-[var(--border-hairline)] bg-[var(--bg-elevated)] p-3"
            data-testid="results-speed-chart"
          >
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={platformSpeedData}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                <XAxis dataKey="platform" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar
                  dataKey="ms"
                  name={lang === "en" ? "Latency (ms)" : "Gecikme (ms)"}
                  fill="#34d399"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-left font-mono text-xs tabular-nums">
              <thead>
                <tr className="border-b border-[var(--border-hairline)] text-[var(--text-muted)]">
                  <th className="py-2 pr-3">Model / Format</th>
                  <th className="py-2 pr-3">ImgSz</th>
                  <th className="py-2 pr-3">MB</th>
                  <th className="py-2 pr-3">Inf (ms)</th>
                  <th className="py-2 pr-3">Total (ms)</th>
                  <th className="py-2 pr-3">p95 (ms)</th>
                  <th className="py-2 pr-3">FPS</th>
                  <th className="py-2 pr-3">mAP50</th>
                  <th className="py-2">Hardware</th>
                </tr>
              </thead>
              <tbody>
                {metrics.edge_benchmark.map((row, i) => (
                  <tr
                    key={`${row.model}-${row.format}-${i}`}
                    className="border-b border-[var(--border-hairline)]/60 text-[var(--text-primary)]"
                  >
                    <td className="py-2 pr-3">
                      {row.model} ({row.format})
                    </td>
                    <td className="py-2 pr-3">{row.imgsz}</td>
                    <td className="py-2 pr-3">{fmtNum(row.boyut_mb, 1)}</td>
                    <td className="py-2 pr-3">
                      {fmtNum(row.gecikme_inf_ort_ms, 1)}
                    </td>
                    <td className="py-2 pr-3">
                      {fmtNum(row.gecikme_toplam_ort_ms, 1)}
                    </td>
                    <td className="py-2 pr-3">
                      {fmtNum(row.gecikme_toplam_p95_ms, 1)}
                    </td>
                    <td className="py-2 pr-3">
                      {fmtNum(row.fps_on_son_islem_dahil, 2)}
                    </td>
                    <td className="py-2 pr-3 font-semibold text-[var(--accent-ink)]">
                      {fmtNum(row.mAP50, 4)}
                    </td>
                    <td className="py-2 font-sans text-[11px] text-[var(--text-secondary)]">
                      {row.donanım}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {metrics.web_spike && metrics.web_spike.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>{t("results.webSpikeTitle")}</CardTitle>
          </CardHeader>
          <CardContent className="overflow-x-auto">
            <table className="w-full border-collapse text-left font-mono text-xs tabular-nums">
              <thead>
                <tr className="border-b border-[var(--border-hairline)] text-[var(--text-muted)]">
                  <th className="py-2 pr-3">Backend</th>
                  <th className="py-2 pr-3">Model</th>
                  <th className="py-2 pr-3">MB</th>
                  <th className="py-2 pr-3">Medyan (ms)</th>
                  <th className="py-2 pr-3">Eşleşme</th>
                  <th className="py-2 pr-3">Ort. IoU</th>
                  <th className="py-2">Ort. |Δconf|</th>
                </tr>
              </thead>
              <tbody>
                {metrics.web_spike.map((row, idx) => (
                  <tr
                    key={`${row.backend}-${row.model}-${idx}`}
                    className="border-b border-[var(--border-hairline)]/60 text-[var(--text-primary)]"
                  >
                    <td className="py-2 pr-3">{row.backend}</td>
                    <td className="py-2 pr-3">{row.model}</td>
                    <td className="py-2 pr-3">{fmtNum(row.boyut_mb, 1)}</td>
                    <td className="py-2 pr-3">
                      {row.medyan_ms !== null ? fmtNum(row.medyan_ms, 1) : "—"}
                    </td>
                    <td className="py-2 pr-3">{fmtPct(row.eşleşme_oranı)}</td>
                    <td className="py-2 pr-3">{fmtNum(row.ort_iou, 4)}</td>
                    <td className="py-2">{fmtNum(row.ort_dconf, 4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>
              {t("results.xaiTitle", { n: xai1024[0]?.n_tespit ?? 150 })}
            </CardTitle>
          </CardHeader>
          <CardContent className="overflow-x-auto">
            <table className="w-full border-collapse text-left font-mono text-xs tabular-nums">
              <thead>
                <tr className="border-b border-[var(--border-hairline)] text-[var(--text-muted)]">
                  <th className="py-2 pr-3">Yöntem</th>
                  <th className="py-2 pr-3">1× Düşüş</th>
                  <th className="py-2 pr-3">1× Kazanma</th>
                  <th className="py-2 pr-3">2× Düşüş</th>
                  <th className="py-2 pr-3">2× Kazanma</th>
                  <th className="py-2">Enerji</th>
                </tr>
              </thead>
              <tbody>
                {xai1024.map((row, idx) => (
                  <tr
                    key={`${row.model}-L${row.katman}-${row.method}-${idx}`}
                    className="border-b border-[var(--border-hairline)]/60 text-[var(--text-primary)]"
                  >
                    <td className="py-2 pr-3 font-semibold">
                      {row.method} (L{row.katman})
                    </td>
                    <td className="py-2 pr-3">
                      {fmtNum(row.goreli_dusus_harita_ort_1x)}
                    </td>
                    <td className="py-2 pr-3">
                      {fmtPct(row.goreli_kazanma_orani_1x)}
                    </td>
                    <td className="py-2 pr-3">
                      {fmtNum(row.goreli_dusus_harita_ort_2x)}
                    </td>
                    <td className="py-2 pr-3">
                      {fmtPct(row.goreli_kazanma_orani_2x)}
                    </td>
                    <td className="py-2">
                      {fmtPct(row.enerji_kutuda_pencere)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>{t("results.aoiTitle")}</CardTitle>
            <CardDescription>{t("results.aoiNote")}</CardDescription>
          </CardHeader>
          <CardContent className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <div className="elevated-card rounded-lg p-3.5">
              <p className="text-xs text-[var(--text-muted)]">COCO Başlangıç</p>
              <p className="mt-1 font-mono text-lg font-semibold text-[var(--text-primary)] tabular-nums">
                mAP50: {fmtNum(metrics.aoi.test_summary.coco.mAP50_mean)} ±{" "}
                {fmtNum(metrics.aoi.test_summary.coco.mAP50_std)}
              </p>
              <p className="font-mono text-xs text-[var(--text-secondary)] tabular-nums">
                mAP50-95:{" "}
                {fmtNum(metrics.aoi.test_summary.coco.mAP50_95_mean)} ±{" "}
                {fmtNum(metrics.aoi.test_summary.coco.mAP50_95_std)}
              </p>
            </div>
            <div className="elevated-card rounded-lg p-3.5">
              <p className="text-xs text-[var(--text-muted)]">PKU Başlangıç</p>
              <p className="mt-1 font-mono text-lg font-semibold text-[var(--text-primary)] tabular-nums">
                mAP50: {fmtNum(metrics.aoi.test_summary.pku.mAP50_mean)} ±{" "}
                {fmtNum(metrics.aoi.test_summary.pku.mAP50_std)}
              </p>
              <p className="font-mono text-xs text-[var(--text-secondary)] tabular-nums">
                mAP50-95:{" "}
                {fmtNum(metrics.aoi.test_summary.pku.mAP50_95_mean)} ±{" "}
                {fmtNum(metrics.aoi.test_summary.pku.mAP50_95_std)}
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* 7. Hata Analizi Galerisi */}
      {analysis && (
        <Card data-testid="error-analysis-section">
          <CardHeader>
            <CardTitle>{t("results.errorAnalysisTitle")}</CardTitle>
            <CardDescription>
              {t("results.errorAnalysisDesc", {
                testFp: analysis.split_totals.test.total,
                valFp: analysis.split_totals.val.total,
              })}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-2">
              <h3 className="text-xs font-semibold uppercase tracking-wide text-[var(--text-muted)]">
                {t("results.fpMatrixTitle")}
              </h3>
              <div className="overflow-x-auto">
                <table
                  className="w-full text-left font-mono text-xs tabular-nums"
                  data-testid="fp-matrix-table"
                >
                  <thead>
                    <tr className="border-b border-[var(--border-hairline)] text-[var(--text-muted)]">
                      <th className="py-2 pr-3">{lang === "en" ? "Board" : "Kart"}</th>
                      <th className="py-2 pr-3">{lang === "en" ? "Split" : "Küme"}</th>
                      {analysis.classes.map((cls) => (
                        <th key={cls} className="py-2 pr-3" title={cls}>
                          {getClassDisplayName(cls, lang, true)}
                        </th>
                      ))}
                      <th className="py-2 font-semibold">Toplam</th>
                    </tr>
                  </thead>
                  <tbody>
                    {analysis.board_class_matrix
                      .filter((row) => row.split === "test")
                      .map((row) => (
                        <tr
                          key={`${row.split}-${row.kart_no}`}
                          className="border-b border-[var(--border-hairline)]/60 text-[var(--text-primary)]"
                        >
                          <td className="py-2 pr-3 font-semibold">
                            {row.kart_no}
                          </td>
                          <td className="py-2 pr-3 text-[var(--text-secondary)]">
                            {splitLabel(row.split, lang)}
                          </td>
                          {analysis.classes.map((cls) => (
                            <td key={cls} className="py-2 pr-3">
                              {row.counts[cls] ?? 0}
                            </td>
                          ))}
                          <td className="py-2 font-semibold">{row.total}</td>
                        </tr>
                      ))}
                    <tr
                      className="border-b-2 border-[var(--border-hairline)] bg-[var(--bg-elevated)] font-semibold text-[var(--text-primary)]"
                      data-testid="fp-total-test"
                    >
                      <td className="py-2 pr-3" colSpan={2}>
                        {t("results.fpSplitTestTotal")}
                      </td>
                      {analysis.classes.map((cls) => (
                        <td key={cls} className="py-2 pr-3">
                          {analysis.split_totals.test.counts[cls] ?? 0}
                        </td>
                      ))}
                      <td className="py-2">
                        {analysis.split_totals.test.total}
                      </td>
                    </tr>
                    {analysis.board_class_matrix
                      .filter((row) => row.split === "val")
                      .map((row) => (
                        <tr
                          key={`${row.split}-${row.kart_no}`}
                          className="border-b border-[var(--border-hairline)]/60 text-[var(--text-primary)]"
                        >
                          <td className="py-2 pr-3 font-semibold">
                            {row.kart_no}
                          </td>
                          <td className="py-2 pr-3 text-[var(--text-secondary)]">
                            {splitLabel(row.split, lang)}
                          </td>
                          {analysis.classes.map((cls) => (
                            <td key={cls} className="py-2 pr-3">
                              {row.counts[cls] ?? 0}
                            </td>
                          ))}
                          <td className="py-2 font-semibold">{row.total}</td>
                        </tr>
                      ))}
                    <tr
                      className="border-t border-[var(--border-hairline)] bg-[var(--bg-elevated)] font-semibold text-[var(--text-primary)]"
                      data-testid="fp-total-val"
                    >
                      <td className="py-2 pr-3" colSpan={2}>
                        {t("results.fpSplitValTotal")}
                      </td>
                      {analysis.classes.map((cls) => (
                        <td key={cls} className="py-2 pr-3">
                          {analysis.split_totals.val.counts[cls] ?? 0}
                        </td>
                      ))}
                      <td className="py-2">
                        {analysis.split_totals.val.total}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <div className="space-y-3">
              <h3 className="text-xs font-semibold uppercase tracking-wide text-[var(--text-muted)]">
                {t("results.templateSpotsTitle")}
              </h3>
              <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                {analysis.template_spots.map((spot) => (
                  <div
                    key={spot.id}
                    data-testid={`template-spot-${spot.cls}`}
                    className="elevated-card space-y-3 rounded-lg p-3.5"
                  >
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div>
                        <p className="font-mono text-sm font-semibold text-[var(--text-primary)]">
                          Kart {spot.kart_no} · {spot.cls}
                        </p>
                        <p className="font-mono text-xs text-[var(--text-muted)]">
                          ({spot.cx}, {spot.cy}) · n={spot.fp_count} · conf{" "}
                          {fmtNum(spot.conf_min, 4)}–{fmtNum(spot.conf_max, 4)}
                        </p>
                      </div>
                      <Badge
                        variant={spot.template_alarm ? "danger" : "outline"}
                      >
                        {spot.template_alarm
                          ? t("results.templateAlarmYes", {
                              conf: fmtNum(spot.template_conf, 4),
                            })
                          : t("results.templateAlarmNo")}
                      </Badge>
                    </div>

                    <div className="grid grid-cols-2 gap-2.5">
                      <figure className="space-y-1">
                        <img
                          src={spot.fp_crop}
                          alt={`${spot.cls} FP`}
                          className="aspect-square w-full rounded border border-[var(--border-hairline)] object-cover"
                        />
                        <figcaption className="text-[11px] text-[var(--text-muted)]">
                          {t("results.fpCropLabel")}
                        </figcaption>
                      </figure>
                      <figure className="space-y-1">
                        <img
                          src={spot.template_crop}
                          alt={`${spot.cls} Template`}
                          className="aspect-square w-full rounded border border-[var(--border-hairline)] object-cover"
                        />
                        <figcaption className="text-[11px] text-[var(--text-muted)]">
                          {t("results.tplCropLabel")}
                        </figcaption>
                      </figure>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="space-y-2">
              <h3 className="text-xs font-semibold uppercase tracking-wide text-[var(--text-muted)]">
                {t("results.quantParityTitle")}
              </h3>
              <p className="text-xs text-[var(--text-secondary)]">
                {t("results.quantParityNote")}
              </p>
              <div className="overflow-x-auto">
                <table
                  className="w-full text-left font-mono text-xs tabular-nums"
                  data-testid="quant-parity-table"
                >
                  <thead>
                    <tr className="border-b border-[var(--border-hairline)] text-[var(--text-muted)]">
                      <th className="py-2 pr-3">{lang === "en" ? "Threshold" : "Eşik"}</th>
                      <th className="py-2 pr-3">{lang === "en" ? "Matched" : "Eşleşen"}</th>
                      <th className="py-2 pr-3">{lang === "en" ? "FP32 only" : "Yalnız FP32"}</th>
                      <th className="py-2 pr-3">{lang === "en" ? "INT8 only" : "Yalnız INT8"}</th>
                      <th className="py-2 pr-3">{lang === "en" ? "Mean IoU" : "Ort. IoU"}</th>
                      <th className="py-2">{lang === "en" ? "Mean |Δconf|" : "Ort. |Δgüven|"}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {analysis.quant_parity.map((q) => (
                      <tr
                        key={q.conf_esik}
                        className="border-b border-[var(--border-hairline)]/60 text-[var(--text-primary)]"
                      >
                        <td className="py-2 pr-3 font-semibold">
                          {fmtNum(Number(q.conf_esik), 3)}
                        </td>
                        <td className="py-2 pr-3">{q.eslesen}</td>
                        <td className="py-2 pr-3">{q.yalniz_fp32}</td>
                        <td className="py-2 pr-3">{q.yalniz_int8}</td>
                        <td className="py-2 pr-3">{fmtNum(q.ort_iou, 4)}</td>
                        <td className="py-2">{fmtNum(q.ort_dconf, 4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* 8. Bu Cihazda Hız Testi */}
      <Card data-testid="device-bench-section">
        <CardHeader>
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <CardTitle>{t("results.deviceBenchTitle")}</CardTitle>
              <CardDescription>{t("results.deviceBenchDesc")}</CardDescription>
            </div>
            <div className="flex items-center gap-2">
              <Button
                type="button"
                size="sm"
                onClick={() => void handleRunDeviceBenchmark()}
                disabled={benchRunning}
                data-testid="device-bench-btn"
              >
                <Gauge className="h-4 w-4" />
                <span>
                  {benchRunning
                    ? t("results.deviceBenchRunning", benchProgress)
                    : t("results.deviceBenchRunBtn")}
                </span>
              </Button>
              {benchReport && (
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  onClick={() => void handleCopyBenchJson()}
                  data-testid="device-bench-copy-btn"
                >
                  {benchCopied ? (
                    <>
                      <Check className="h-4 w-4 text-emerald-500" />
                      <span>{t("results.deviceBenchCopied")}</span>
                    </>
                  ) : (
                    <>
                      <Copy className="h-4 w-4" />
                      <span>{t("results.deviceBenchCopyBtn")}</span>
                    </>
                  )}
                </Button>
              )}
            </div>
          </div>
        </CardHeader>

        {(benchError || benchReport) && (
          <CardContent className="space-y-3">
            {benchError && (
              <div
                role="alert"
                className="rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-xs text-red-700 dark:text-red-300"
              >
                {benchError}
              </div>
            )}

            {benchReport && (
              <div className="overflow-x-auto">
                <table
                  className="w-full text-left font-mono text-xs tabular-nums"
                  data-testid="device-bench-table"
                >
                  <tbody>
                    <tr className="border-b border-[var(--border-hairline)]/60">
                      <th className="py-2 pr-4 font-medium text-[var(--text-muted)]">
                        {t("results.deviceBenchBackend")}
                      </th>
                      <td className="py-2 font-semibold text-[var(--text-primary)]">
                        {benchReport.backend}
                      </td>
                    </tr>
                    <tr className="border-b border-[var(--border-hairline)]/60">
                      <th className="py-2 pr-4 font-medium text-[var(--text-muted)]">
                        {t("results.deviceBenchMedianModel")}
                      </th>
                      <td className="py-2 font-semibold text-[var(--text-primary)]">
                        {fmtMs(benchReport.median_model_ms, 2)}
                      </td>
                    </tr>
                    <tr className="border-b border-[var(--border-hairline)]/60">
                      <th className="py-2 pr-4 font-medium text-[var(--text-muted)]">
                        {t("results.deviceBenchMedianTotal")}
                      </th>
                      <td className="py-2 font-semibold text-[var(--text-primary)]">
                        {fmtMs(benchReport.median_total_ms, 2)}
                      </td>
                    </tr>
                    <tr className="border-b border-[var(--border-hairline)]/60">
                      <th className="py-2 pr-4 font-medium text-[var(--text-muted)]">
                        {t("results.deviceBenchRuns")}
                      </th>
                      <td className="py-2 text-[var(--text-primary)]">
                        {benchReport.timed_runs} ({benchReport.images} ×{" "}
                        {benchReport.repeats}, +{benchReport.warmup_runs} ısınma)
                      </td>
                    </tr>
                    <tr className="border-b border-[var(--border-hairline)]/60">
                      <th className="py-2 pr-4 font-medium text-[var(--text-muted)]">
                        {t("results.deviceBenchGpu")}
                      </th>
                      <td className="py-2 text-[var(--text-primary)]">
                        {benchReport.gpu_adapter.available
                          ? [
                              benchReport.gpu_adapter.vendor,
                              benchReport.gpu_adapter.architecture,
                              benchReport.gpu_adapter.device,
                              benchReport.gpu_adapter.description,
                            ]
                              .filter(Boolean)
                              .join(" · ") || "WebGPU Adapter"
                          : t("results.deviceBenchNoGpu")}
                      </td>
                    </tr>
                    <tr>
                      <th className="py-2 pr-4 font-medium text-[var(--text-muted)]">
                        {t("results.deviceBenchUa")}
                      </th>
                      <td className="py-2 break-all text-[var(--text-secondary)]">
                        {benchReport.user_agent}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        )}
      </Card>
    </div>
  );
}
