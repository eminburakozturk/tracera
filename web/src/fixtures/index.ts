import type { DetectResult, MetricsPayload } from "../types";
import metricsFallbackJson from "./metrics.fallback.json";
import ref05MissingHoleOv from "./ref/05_missing_hole_01__openvino-int8.json";
import ref05MissingHolePt from "./ref/05_missing_hole_01__pytorch.json";
import ref05MouseBiteOv from "./ref/05_mouse_bite_01__openvino-int8.json";
import ref05OpenCircuitOv from "./ref/05_open_circuit_01__openvino-int8.json";
import ref05ShortOv from "./ref/05_short_01__openvino-int8.json";
import ref05ShortPt from "./ref/05_short_01__pytorch.json";
import ref05SpurOv from "./ref/05_spur_01__openvino-int8.json";
import ref05SpuriousCopperOv from "./ref/05_spurious_copper_01__openvino-int8.json";
import ref11MissingHoleOv from "./ref/11_missing_hole_01__openvino-int8.json";
import ref11MouseBiteOv from "./ref/11_mouse_bite_01__openvino-int8.json";
import ref11MouseBitePt from "./ref/11_mouse_bite_01__pytorch.json";
import ref11OpenCircuitOv from "./ref/11_open_circuit_01__openvino-int8.json";
import ref11SpurOv from "./ref/11_spur_01__openvino-int8.json";
import ref11SpurPt from "./ref/11_spur_01__pytorch.json";
import ref11SpuriousCopperOv from "./ref/11_spurious_copper_01__openvino-int8.json";

export interface ValExampleItem {
  id: string;
  filename: string;
  url: string;
  board: "05" | "11";
  defectClass: string;
}

export const VAL_EXAMPLES: ValExampleItem[] = [
  {
    id: "val_05_missing_hole_01",
    filename: "val_05_missing_hole_01.jpg",
    url: "/examples/val_05_missing_hole_01.jpg",
    board: "05",
    defectClass: "missing_hole",
  },
  {
    id: "val_05_short_01",
    filename: "val_05_short_01.jpg",
    url: "/examples/val_05_short_01.jpg",
    board: "05",
    defectClass: "short",
  },
  {
    id: "val_11_mouse_bite_01",
    filename: "val_11_mouse_bite_01.jpg",
    url: "/examples/val_11_mouse_bite_01.jpg",
    board: "11",
    defectClass: "mouse_bite",
  },
  {
    id: "val_11_spur_01",
    filename: "val_11_spur_01.jpg",
    url: "/examples/val_11_spur_01.jpg",
    board: "11",
    defectClass: "spur",
  },
];

const FIXTURE_MAP: Record<string, DetectResult> = {
  "05_missing_hole_01": ref05MissingHoleOv as DetectResult,
  val_05_missing_hole_01: ref05MissingHoleOv as DetectResult,
  "05_short_01": ref05ShortOv as DetectResult,
  val_05_short_01: ref05ShortOv as DetectResult,
  "11_mouse_bite_01": ref11MouseBiteOv as DetectResult,
  val_11_mouse_bite_01: ref11MouseBiteOv as DetectResult,
  "11_spur_01": ref11SpurOv as DetectResult,
  val_11_spur_01: ref11SpurOv as DetectResult,
  "05_mouse_bite_01": ref05MouseBiteOv as DetectResult,
  "05_open_circuit_01": ref05OpenCircuitOv as DetectResult,
  "05_spur_01": ref05SpurOv as DetectResult,
  "05_spurious_copper_01": ref05SpuriousCopperOv as DetectResult,
  "11_missing_hole_01": ref11MissingHoleOv as DetectResult,
  "11_open_circuit_01": ref11OpenCircuitOv as DetectResult,
  "11_spurious_copper_01": ref11SpuriousCopperOv as DetectResult,
};

export const PYTORCH_REF_MAP: Record<string, DetectResult> = {
  "05_missing_hole_01": ref05MissingHolePt as DetectResult,
  "05_short_01": ref05ShortPt as DetectResult,
  "11_mouse_bite_01": ref11MouseBitePt as DetectResult,
  "11_spur_01": ref11SpurPt as DetectResult,
};

export function findDetectFixture(filename: string): DetectResult {
  const stem = filename.replace(/\.[^.]+$/, "").replace(/^val_/, "");
  return (
    FIXTURE_MAP[stem] ??
    FIXTURE_MAP[filename.replace(/\.[^.]+$/, "")] ??
    (ref05MissingHoleOv as DetectResult)
  );
}

export const METRICS_FALLBACK = metricsFallbackJson as unknown as MetricsPayload;
