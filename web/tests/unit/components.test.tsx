import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { AboutView } from "../../src/components/AboutView";
import { DetectView } from "../../src/components/DetectView";
import { ExplainView } from "../../src/components/ExplainView";
import { ResultsView } from "../../src/components/ResultsView";
import { TopBar } from "../../src/components/TopBar";
import { METRICS_FALLBACK, VAL_EXAMPLES } from "../../src/fixtures";
import enJson from "../../src/i18n/en.json";
import trJson from "../../src/i18n/tr.json";
import { useAppStore } from "../../src/store/useAppStore";

const FORBIDDEN_PATTERNS = [
  /mikron/i,
  /micron/i,
  /mikro[\s-]?doku/i,
  /micro[\s-]?texture/i,
  /uzamsal çözünürlük/i,
  /spatial resolution/i,
];

describe("Tracera UI components & wording rules", () => {
  beforeEach(() => {
    useAppStore.setState({
      metrics: METRICS_FALLBACK,
      metricsSource: "static",
      activeTab: "results",
      lang: "tr",
      theme: "dark",
    });
  });

  it("never exposes forbidden wording in TR or EN i18n dictionaries", () => {
    const combined = JSON.stringify({ trJson, enJson });
    for (const pat of FORBIDDEN_PATTERNS) {
      expect(combined).not.toMatch(pat);
    }
  });

  it("renders ResultsView dynamically from metrics.json without oracle P/R", () => {
    const { container } = render(<ResultsView />);
    const text = container.textContent ?? "";

    expect(text).toContain("0.858 ± 0.152");
    expect(text).toContain("0.395");
    expect(text).toContain("test_sablon_noktalari_haric");
    expect(text).toContain("Ryzen 7 8845HS");
    // Oracle test precision/recall (0.9462 / 0.8577) must NEVER be displayed
    expect(text).not.toContain("0.9462");
    expect(text).not.toContain("0.8577");
    for (const pat of FORBIDDEN_PATTERNS) {
      expect(text).not.toMatch(pat);
    }
  });

  it("shows only val boards 05 and 11 in validation gallery (never test boards 06/10)", () => {
    render(
      <DetectView
        onSelectExample={async () => {}}
        onSelectFile={async () => {}}
        onRunDetect={async () => {}}
        onInspectDetection={() => {}}
      />,
    );
    for (const ex of VAL_EXAMPLES) {
      expect(["05", "11"]).toContain(ex.board);
    }
    expect(screen.getByTestId("example-val_05_missing_hole_01")).toBeInTheDocument();
    expect(screen.queryByText(/06_/)).not.toBeInTheDocument();
    expect(screen.queryByText(/10_/)).not.toBeInTheDocument();
  });

  it("renders ExplainView & AboutView cleanly without forbidden terms", () => {
    const { container: expContainer } = render(
      <ExplainView onRunExplain={async () => {}} />,
    );
    const { container: aboutContainer } = render(<AboutView />);
    const text = `${expContainer.textContent ?? ""}\n${aboutContainer.textContent ?? ""}`;
    expect(text).toContain("AGPL-3.0");
    for (const pat of FORBIDDEN_PATTERNS) {
      expect(text).not.toMatch(pat);
    }
  });

  it("renders backend parity info button next to backend selector", () => {
    render(<TopBar onRecheckHealth={() => {}} ortEnabled />);
    const infoBtn = screen.getByTestId("backend-parity-info");
    expect(infoBtn).toBeInTheDocument();
    expect(infoBtn).toHaveAttribute(
      "title",
      "Cihazda (FP32) ve Yerel sunucu (INT8) modelleri eşik sınırındaki birkaç kutuda farklı sonuç verebilir; testlerde bu farklar kart şablonundaki sabit noktalarda yoğunlaştı.",
    );
  });
});
