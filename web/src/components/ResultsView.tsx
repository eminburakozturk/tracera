import * as React from "react";
import { useTranslation } from "react-i18next";
import { loadAnalysis } from "../lib/analysis";
import { useAppStore } from "../store/useAppStore";
import { Badge } from "./ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./ui/card";

function fmtPct(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

function fmtNum(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return v.toFixed(digits);
}

export function ResultsView() {
  const { t } = useTranslation();
  const metrics = useAppStore((s) => s.metrics);
  const metricsSource = useAppStore((s) => s.metricsSource);
  const analysis = useAppStore((s) => s.analysis);
  const setAnalysis = useAppStore((s) => s.setAnalysis);

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

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h2 className="text-xl font-semibold tracking-tight text-[var(--text-primary)]">
            {t("results.title")}
          </h2>
          <p className="text-xs text-[var(--text-muted)]">{t("results.subtitle")}</p>
        </div>
        {metricsSource && (
          <Badge variant="secondary" className="font-mono">
            {t("results.sourceLabel")}: {metricsSource}
          </Badge>
        )}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>{t("results.cvTitle")}</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-4">
          <div className="elevated-card rounded-lg p-3.5">
            <p className="text-xs text-[var(--text-muted)]">
              {t("results.cvSummaryGrouped1024")}
            </p>
            <p className="mt-1 font-mono text-xl font-semibold text-[var(--accent)] tabular-nums">
              {fmtNum(cv1024?.mAP50_mean)} ± {fmtNum(cv1024?.mAP50_std)}
            </p>
            <p className="mt-0.5 font-mono text-xs text-[var(--text-secondary)] tabular-nums">
              mAP50-95: {fmtNum(cv1024?.mAP50_95_mean)} ± {fmtNum(cv1024?.mAP50_95_std)}
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
              mAP50-95: {fmtNum(cv640?.mAP50_95_mean)} ± {fmtNum(cv640?.mAP50_95_std)}
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
              mAP50-95: {fmtNum(leakRand.mAP50_95_mean)} ± {fmtNum(leakRand.mAP50_95_std)}
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
              mAP50-95: {fmtNum(leakGroup.mAP50_95_mean)} ± {fmtNum(leakGroup.mAP50_95_std)}
            </p>
          </div>
        </CardContent>
      </Card>

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
              {t("results.lockedTestMap")}: mAP50 = {fmtNum(testMap50, 4)} · mAP50-95 ={" "}
              {fmtNum(testMap5095, 4)}
            </CardDescription>
          )}
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-left font-mono text-xs tabular-nums">
              <thead>
                <tr className="border-b border-[var(--border-hairline)] text-[var(--text-muted)]">
                  <th className="py-2 pr-3">Split</th>
                  <th className="py-2 pr-3">TP</th>
                  <th className="py-2 pr-3">FP</th>
                  <th className="py-2 pr-3">Bg FP</th>
                  <th className="py-2 pr-3">Konum FP</th>
                  <th className="py-2 pr-3">Dup FP</th>
                  <th className="py-2 pr-3">Cross FP</th>
                  <th className="py-2 pr-3">FN</th>
                  <th className="py-2 pr-3">P</th>
                  <th className="py-2 pr-3">R</th>
                  <th className="py-2">F1</th>
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
                        {row.split}
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
                      <td className="py-2 font-semibold text-[var(--accent)]">
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

      <Card>
        <CardHeader>
          <CardTitle>{t("results.edgeTitle")}</CardTitle>
          <CardDescription>{t("results.edgeHardwareNote")}</CardDescription>
        </CardHeader>
        <CardContent className="overflow-x-auto">
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
                  <td className="py-2 pr-3">{fmtNum(row.gecikme_inf_ort_ms, 1)}</td>
                  <td className="py-2 pr-3">{fmtNum(row.gecikme_toplam_ort_ms, 1)}</td>
                  <td className="py-2 pr-3">{fmtNum(row.gecikme_toplam_p95_ms, 1)}</td>
                  <td className="py-2 pr-3">{fmtNum(row.fps_on_son_islem_dahil, 2)}</td>
                  <td className="py-2 pr-3 font-semibold text-[var(--accent)]">
                    {fmtNum(row.mAP50, 4)}
                  </td>
                  <td className="py-2 font-sans text-[11px] text-[var(--text-secondary)]">
                    {row.donanım}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
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
                    <td className="py-2 pr-3">{fmtNum(row.goreli_dusus_harita_ort_1x)}</td>
                    <td className="py-2 pr-3">{fmtPct(row.goreli_kazanma_orani_1x)}</td>
                    <td className="py-2 pr-3">{fmtNum(row.goreli_dusus_harita_ort_2x)}</td>
                    <td className="py-2 pr-3">{fmtPct(row.goreli_kazanma_orani_2x)}</td>
                    <td className="py-2">{fmtPct(row.enerji_kutuda_pencere)}</td>
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
                mAP50-95: {fmtNum(metrics.aoi.test_summary.coco.mAP50_95_mean)} ±{" "}
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
                mAP50-95: {fmtNum(metrics.aoi.test_summary.pku.mAP50_95_mean)} ±{" "}
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
              {t("results.errorAnalysisDesc", { totalFp: analysis.total_fp_rows })}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Kart × Sınıf FP tablosu */}
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
                      <th className="py-2 pr-3">Kart</th>
                      <th className="py-2 pr-3">Split</th>
                      {analysis.classes.map((cls) => (
                        <th key={cls} className="py-2 pr-3">
                          {cls}
                        </th>
                      ))}
                      <th className="py-2 font-semibold">Toplam</th>
                    </tr>
                  </thead>
                  <tbody>
                    {analysis.board_class_matrix.map((row) => (
                      <tr
                        key={`${row.split}-${row.kart_no}`}
                        className="border-b border-[var(--border-hairline)]/60 text-[var(--text-primary)]"
                      >
                        <td className="py-2 pr-3 font-semibold">{row.kart_no}</td>
                        <td className="py-2 pr-3 text-[var(--text-secondary)]">
                          {row.split}
                        </td>
                        {analysis.classes.map((cls) => (
                          <td key={cls} className="py-2 pr-3">
                            {row.counts[cls] ?? 0}
                          </td>
                        ))}
                        <td className="py-2 font-semibold">{row.total}</td>
                      </tr>
                    ))}
                  </tbody>
                  <tfoot>
                    <tr className="border-t border-[var(--border-hairline)] font-semibold text-[var(--text-primary)]">
                      <td className="py-2 pr-3" colSpan={2}>
                        Toplam
                      </td>
                      {analysis.classes.map((cls) => (
                        <td key={cls} className="py-2 pr-3">
                          {analysis.board_class_matrix.reduce(
                            (acc, r) => acc + (r.counts[cls] ?? 0),
                            0,
                          )}
                        </td>
                      ))}
                      <td className="py-2">{analysis.total_fp_rows}</td>
                    </tr>
                  </tfoot>
                </table>
              </div>
            </div>

            {/* Kart 06'daki 2 Sabit Şablon Noktası */}
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
                      <Badge variant={spot.template_alarm ? "danger" : "outline"}>
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

            {/* FP32 vs INT8 Niceleme Kıyası */}
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
                      <th className="py-2 pr-3">conf_esik</th>
                      <th className="py-2 pr-3">eslesen</th>
                      <th className="py-2 pr-3">yalniz_fp32</th>
                      <th className="py-2 pr-3">yalniz_int8</th>
                      <th className="py-2 pr-3">ort_iou</th>
                      <th className="py-2">ort_dconf</th>
                    </tr>
                  </thead>
                  <tbody>
                    {analysis.quant_parity.map((q) => (
                      <tr
                        key={q.conf_esik}
                        className="border-b border-[var(--border-hairline)]/60 text-[var(--text-primary)]"
                      >
                        <td className="py-2 pr-3 font-semibold">{q.conf_esik}</td>
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
    </div>
  );
}
