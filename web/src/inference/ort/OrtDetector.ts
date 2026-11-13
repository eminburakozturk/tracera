import type { DetectResult } from "../../types";
import type { Detector } from "../types";

/**
 * Tarayıcı içi ONNX Runtime Web çıkarımı iskeleti.
 * TODO(Adım 3): Web Worker içinde WebGPU/WASM + cv2.INTER_LINEAR uyumlu bilineer ön işleme ile doldurulacak.
 */
export class OrtDetector implements Detector {
  async detect(_file: File, _conf: number): Promise<DetectResult> {
    void _file;
    void _conf;
    throw new Error(
      "OrtDetector (tarayıcı içi çıkarım) henüz etkinleştirilmedi — Adım 3 parite doğrulamasından sonra açılacaktır.",
    );
  }
}
