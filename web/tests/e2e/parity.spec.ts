import fs from "node:fs";
import path from "node:path";
import { expect, test } from "@playwright/test";
import type { Detection, DetectResult } from "../../src/types";

interface MatchStats {
  matched: number;
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

function matchDetections(refDets: Detection[], ortDets: Detection[]): MatchStats {
  const pairs: Array<{ ri: number; oi: number; iou: number; dconf: number }> = [];
  for (let ri = 0; ri < refDets.length; ri++) {
    for (let oi = 0; oi < ortDets.length; oi++) {
      if (refDets[ri].cls_id !== ortDets[oi].cls_id) continue;
      const iou = boxIoU(refDets[ri].box, ortDets[oi].box);
      if (iou >= 0.3) {
        pairs.push({
          ri,
          oi,
          iou,
          dconf: Math.abs(refDets[ri].conf - ortDets[oi].conf),
        });
      }
    }
  }
  pairs.sort((a, b) => b.iou - a.iou);
  const usedR = new Set<number>();
  const usedO = new Set<number>();
  let sumIoU = 0;
  let sumDConf = 0;
  let matched = 0;

  for (const p of pairs) {
    if (usedR.has(p.ri) || usedO.has(p.oi)) continue;
    usedR.add(p.ri);
    usedO.add(p.oi);
    matched += 1;
    sumIoU += p.iou;
    sumDConf += p.dconf;
  }

  return {
    matched,
    meanIoU: matched > 0 ? sumIoU / matched : 0,
    meanDConf: matched > 0 ? sumDConf / matched : 0,
  };
}

test("Adım 3 — OrtDetector parity against PyTorch on 10 val images (conf>=0.25 + counts at 0.395)", async ({
  browser,
}) => {
  test.setTimeout(360_000);

  const repoRoot = path.resolve(process.cwd(), "..");
  const samplesPath = path.join(repoRoot, "web_spike/val_manifest.json");
  const refDir = path.join(repoRoot, "web_spike/ref");
  const outCsvPath = path.join(repoRoot, "results/web_parity.csv");

  const samples = JSON.parse(fs.readFileSync(samplesPath, "utf-8")) as string[];
  expect(samples).toHaveLength(10);

  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
  });
  const page = await context.newPage();
  page.on("console", (msg) => {
    if (msg.type() === "error" || msg.type() === "warning") {
      console.log(`[browser:${msg.type()}]`, msg.text());
    }
  });
  page.on("pageerror", (err) => {
    console.log(`[browser:pageerror]`, err.message);
  });

  await page.goto("/");

  // Verify UI selector includes "Cihazda (tarayıcı)" and runs cleanly in Worker
  await page.getByTestId("backend-select").click();
  await page.getByRole("option", { name: /Cihazda \(tarayıcı\)/i }).click();
  await page.waitForSelector('[data-testid^="bbox-"]', { timeout: 120_000 });
  const uiBackend = await page.getByTestId("backend-badge").textContent();
  expect(uiBackend).toContain("Cihazda (tarayıcı)");

  // Now run parity batch for all 10 spike val images inside the Web Worker via OrtDetector
  const rows: string[] = [
    "image,backend,pt_count_025,ort_count_025,matched_025,mean_iou,mean_abs_dconf,pt_count_0395,ort_count_0395,diff_0395,model_ms,total_ms",
  ];

  let totalMatched = 0;
  let totalPt025 = 0;
  let weightedIoU = 0;
  let weightedDConf = 0;
  let activeBackend = "";

  for (const filename of samples) {
    const stem = filename.replace(/\.[^.]+$/, "");
    const refJsonPath = path.join(refDir, `${stem}__pytorch.json`);
    const refFull = JSON.parse(fs.readFileSync(refJsonPath, "utf-8")) as DetectResult;

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

    const m = matchDetections(pt025, ort025);
    totalPt025 += pt025.length;
    totalMatched += m.matched;
    weightedIoU += m.meanIoU * m.matched;
    weightedDConf += m.meanDConf * m.matched;

    const diff0395 = ort0395.length - pt0395.length;
    rows.push(
      [
        filename,
        ortRes.backend,
        pt025.length,
        ort025.length,
        m.matched,
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
  const matchRate = totalPt025 > 0 ? totalMatched / totalPt025 : 0;

  rows.push(
    [
      "SUMMARY",
      activeBackend,
      totalPt025,
      "-",
      totalMatched,
      overallIoU.toFixed(6),
      overallDConf.toFixed(6),
      "-",
      "-",
      matchRate.toFixed(4),
      "-",
      "-",
    ].join(","),
  );

  fs.writeFileSync(outCsvPath, `${rows.join("\n")}\n`, "utf-8");

  await context.close();

  expect(totalMatched).toBeGreaterThan(0);
  expect(matchRate).toBeGreaterThanOrEqual(0.9);
  expect(overallIoU).toBeGreaterThanOrEqual(0.95);
  expect(overallDConf).toBeLessThanOrEqual(0.01);
});
