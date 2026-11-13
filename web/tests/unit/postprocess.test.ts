import { describe, expect, it } from "vitest";
import { FALLBACK_META } from "../../src/lib/meta";
import {
  classWiseNms,
  computeIoU,
  computeLetterbox,
  postprocessYolo,
  roundHalfToEven,
  unscaleBox,
} from "../../src/inference/ort/postprocess";

describe("postprocess helpers", () => {
  it("computes letterbox geometry accurately for 3056x2464 -> 1024", () => {
    const lb = computeLetterbox(3056, 2464, 1024);
    expect(lb.ratio).toBeCloseTo(1024 / 3056, 6);
    expect(lb.unpadW).toBe(1024);
    expect(lb.unpadH).toBe(826);
    expect(lb.top).toBe(99);
    expect(lb.bottom).toBe(99);
    expect(lb.left).toBe(0);
    expect(lb.right).toBe(0);
  });

  it("applies round-half-to-even matching cv2 padding rules", () => {
    expect(roundHalfToEven(98.5)).toBe(98);
    expect(roundHalfToEven(99.5)).toBe(100);
    expect(roundHalfToEven(2.6)).toBe(3);
  });

  it("computes IoU accurately", () => {
    expect(
      computeIoU({ x1: 0, y1: 0, x2: 10, y2: 10 }, { x1: 0, y1: 0, x2: 10, y2: 10 }),
    ).toBeCloseTo(1, 6);
    expect(
      computeIoU({ x1: 0, y1: 0, x2: 10, y2: 10 }, { x1: 10, y1: 10, x2: 20, y2: 20 }),
    ).toBeCloseTo(0, 6);
    expect(
      computeIoU({ x1: 0, y1: 0, x2: 10, y2: 10 }, { x1: 5, y1: 0, x2: 15, y2: 10 }),
    ).toBeCloseTo(1 / 3, 5);
  });

  it("unscales padded 1024 boxes back to original pixel bounds", () => {
    const lb = computeLetterbox(2048, 1024, 1024);
    const box = unscaleBox([512, 256, 1024, 768], lb);
    expect(box[0]).toBeCloseTo(1024, 2);
    expect(box[1]).toBeCloseTo(0, 2);
    expect(box[2]).toBeCloseTo(2048, 2);
    expect(box[3]).toBeCloseTo(1024, 2);
  });

  it("suppresses overlapping same-class candidates while keeping other classes", () => {
    const kept = classWiseNms(
      [
        { x1: 100, y1: 100, x2: 200, y2: 200, conf: 0.9, clsId: 0 },
        { x1: 102, y1: 102, x2: 198, y2: 198, conf: 0.8, clsId: 0 },
        { x1: 102, y1: 102, x2: 198, y2: 198, conf: 0.75, clsId: 3 },
      ],
      0.7,
    );
    expect(kept).toHaveLength(2);
    expect(kept[0].clsId).toBe(0);
    expect(kept[0].conf).toBeCloseTo(0.9, 5);
    expect(kept[1].clsId).toBe(3);
  });

  it("decodes raw [1, 10, N] YOLO output tensor correctly", () => {
    const nBoxes = 4;
    const data = new Float32Array(10 * nBoxes);
    // Box 0: missing_hole (cls 0) conf 0.88 at center (512, 512) size 100x100
    data[0 * nBoxes + 0] = 512;
    data[1 * nBoxes + 0] = 512;
    data[2 * nBoxes + 0] = 100;
    data[3 * nBoxes + 0] = 100;
    data[4 * nBoxes + 0] = 0.88;

    // Box 1: overlapping lower-conf missing_hole
    data[0 * nBoxes + 1] = 514;
    data[1 * nBoxes + 1] = 512;
    data[2 * nBoxes + 1] = 100;
    data[3 * nBoxes + 1] = 100;
    data[4 * nBoxes + 1] = 0.65;

    // Box 2: below conf threshold 0.395
    data[0 * nBoxes + 2] = 100;
    data[1 * nBoxes + 2] = 100;
    data[2 * nBoxes + 2] = 40;
    data[3 * nBoxes + 2] = 40;
    data[5 * nBoxes + 2] = 0.3;

    const lb = computeLetterbox(1024, 1024, 1024);
    const dets = postprocessYolo(data, [1, 10, nBoxes], lb, FALLBACK_META, 0.395);
    expect(dets).toHaveLength(1);
    expect(dets[0].cls).toBe("missing_hole");
    expect(dets[0].conf).toBeCloseTo(0.88, 3);
    expect(dets[0].box[0]).toBeCloseTo(462, 1);
    expect(dets[0].box[2]).toBeCloseTo(562, 1);
  });
});
