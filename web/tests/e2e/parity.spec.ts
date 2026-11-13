import fs from "node:fs";
import path from "node:path";
import { expect, test } from "@playwright/test";
import type { Detection, DetectResult } from "../../src/types";

interface TwoWayMatchStats {
  matched: number;
  onlyOrt: number;
  onlyPt: number;
  twoWayPct: number;
  meanIoU: number;
  meanDConf: number;
}

function boxIoU(a: Detection["box"], b: Detection["box"]): number {
  const ix1 = Math.max(a[0], b[0]);
  const iy1 = Math.max(a[1], b[1]);
  const ix2 = Math.min(a[2], b[2]);
  const iy2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
  if (inter <= 0) return 0;
  const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  const union = areaA + areaB - inter;
  return union > 0 ? inter / union : 0;
}

/**
 * İki yönlü açgözlü eşleşme:
 * Aynı sınıf, IoU >= 0.5, güvene göre sıralı (yüksek güvenli kutudan başlayarak 1-1 eşleme).
 */
function twoWayGreedyMatch(
  ptDets: Detection[],
  ortDets: Detection[],
  iouMin = 0.5,
): TwoWayMatchStats {
  const ortSorted = [...ortDets].sort((a, b) => b.conf - a.conf);
  const usedPt = new Set<number>();

  let matched = 0;
  let sumIoU = 0;
  let sumDConf = 0;

  for (const od of ortSorted) {
    let bestPtIdx = -1;
    let bestIoU = iouMin;
    for (let pi = 0; pi < ptDets.length; pi++) {
      if (usedPt.has(pi)) continue;
      const pd = ptDets[pi];
      if (pd.cls_id !== od.cls_id) continue;
      const iou = boxIoU(pd.box, od.box);
      if (iou >= bestIoU) {
        bestIoU = iou;
        bestPtIdx = pi;
      }
    }
    if (bestPtIdx >= 0) {
      usedPt.add(bestPtIdx);
      matched += 1;
      sumIoU += bestIoU;
      sumDConf += Math.abs(ptDets[bestPtIdx].conf - od.conf);
    }
  }

  const onlyOrt = ortDets.length - matched;
  const onlyPt = ptDets.length - matched;
  const denom = ptDets.length + ortDets.length;
  const twoWayPct = denom > 0 ? (2 * matched) / denom : 1;

  return {
    matched,
    onlyOrt,
    onlyPt,
    twoWayPct,
    meanIoU: matched > 0 ? sumIoU / matched : 0,
    meanDConf: matched > 0 ? sumDConf / matched : 0,
  };
}

test("Ek Adım 3 — OrtDetector two-way parity against PyTorch c025 refs (IoU>=0.5, conf=0.25 + counts at 0.395)", async ({
  browser,
}) => {
  test.setTimeout(360_000);

  const repoRoot = path.resolve(process.cwd(), "..");
  const refDir = path.join(repoRoot, "web_spike/ref");
  const outCsvPath = path.join(repoRoot, "results/web_parity.csv");

  const refFiles = fs
    .readdirSync(refDir)
    .filter((f) => f.endsWith("__pytorch_c025.json"))
    .sort();
  expect(refFiles).toHaveLength(10);

  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
  });
  const page = await context.newPage();

  await page.goto("/");

  await page.getByTestId("backend-select").click();
  await page.getByRole("option", { name: /Cihazda \(tarayıcı\)/i }).click();
  await page.waitForSelector('[data-testid^="bbox-"]', { timeout: 120_000 });

  const rows: string[] = [
    "image,backend,pt_count_025,ort_count_025,matched_025,only_ort_025,only_pt_025,two_way_match_pct,mean_iou,mean_abs_dconf,pt_count_0395,ort_count_0395,diff_0395,model_ms,total_ms",
  ];

  let totalPt025 = 0;
  let totalOrt025 = 0;
  let totalMatched = 0;
  let totalOnlyOrt = 0;
  let totalOnlyPt = 0;
  let weightedIoU = 0;
  let weightedDConf = 0;
  let totalPt0395 = 0;
  let totalOrt0395 = 0;
  let activeBackend = "";

  for (const refName of refFiles) {
    const stem = refName.replace(/__pytorch_c025\.json$/, "");
    const filename = `${stem}.jpg`;
    const refFull = JSON.parse(
      fs.readFileSync(path.join(refDir, refName), "utf-8"),
    ) as DetectResult;

    const pt025 = refFull.detections.filter((d) => d.conf >= 0.25);
    const pt0395 = refFull.detections.filter((d) => d.conf >= 0.395);

    const ortRes = (await page.evaluate(async (fname: string) => {
      const ortUrl = "/src/inference/ort/OrtDetector.ts";
      const metaUrl = "/src/lib/meta.ts";
      const [{ OrtDetector }, { loadModelMeta }] = await Promise.all([
        import(/* @vite-ignore */ ortUrl),
        import(/* @vite-ignore */ metaUrl),
      ]);
      const meta = await loadModelMeta();
      const resp = await fetch(`/spike-val/${fname}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status} for ${fname}`);
      const blob = await resp.blob();
      const file = new File([blob], fname, { type: "image/jpeg" });
      const detector = new OrtDetector(meta);
      return await detector.detect(file, 0.25);
    }, filename)) as DetectResult;

    activeBackend = ortRes.backend;
    const ort025 = ortRes.detections.filter((d: Detection) => d.conf >= 0.25);
    const ort0395 = ortRes.detections.filter((d: Detection) => d.conf >= 0.395);

    const m = twoWayGreedyMatch(pt025, ort025, 0.5);
    totalPt025 += pt025.length;
    totalOrt025 += ort025.length;
    totalMatched += m.matched;
    totalOnlyOrt += m.onlyOrt;
    totalOnlyPt += m.onlyPt;
    weightedIoU += m.meanIoU * m.matched;
    weightedDConf += m.meanDConf * m.matched;
    totalPt0395 += pt0395.length;
    totalOrt0395 += ort0395.length;

    const diff0395 = ort0395.length - pt0395.length;
    rows.push(
      [
        filename,
        ortRes.backend,
        pt025.length,
        ort025.length,
        m.matched,
        m.onlyOrt,
        m.onlyPt,
        (m.twoWayPct * 100).toFixed(2),
        m.meanIoU.toFixed(6),
        m.meanDConf.toFixed(6),
        pt0395.length,
        ort0395.length,
        diff0395,
        ortRes.timing.model_ms.toFixed(2),
        ortRes.timing.total_ms.toFixed(2),
      ].join(","),
    );
  }

  const overallIoU = totalMatched > 0 ? weightedIoU / totalMatched : 0;
  const overallDConf = totalMatched > 0 ? weightedDConf / totalMatched : 0;
  const twoWayMatchRate =
    totalPt025 + totalOrt025 > 0
      ? (2 * totalMatched) / (totalPt025 + totalOrt025)
      : 0;

  rows.push(
    [
      "SUMMARY",
      activeBackend,
      totalPt025,
      totalOrt025,
      totalMatched,
      totalOnlyOrt,
      totalOnlyPt,
      (twoWayMatchRate * 100).toFixed(2),
      overallIoU.toFixed(6),
      overallDConf.toFixed(6),
      totalPt0395,
      totalOrt0395,
      totalOrt0395 - totalPt0395,
      "-",
      "-",
    ].join(","),
  );

  fs.writeFileSync(outCsvPath, `${rows.join("\n")}\n`, "utf-8");

  await context.close();

  expect(totalMatched).toBeGreaterThan(0);
  expect(twoWayMatchRate).toBeGreaterThanOrEqual(0.95);
  expect(overallIoU).toBeGreaterThanOrEqual(0.95);
  expect(overallDConf).toBeLessThanOrEqual(0.01);
});
