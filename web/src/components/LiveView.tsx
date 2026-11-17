import * as React from "react";
import {
  AlertTriangle,
  Camera,
  FileVideo,
  Info,
  Pause,
  Play,
  Square,
  Upload,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import {
  OrtDetector,
  type OrtEpPreference,
} from "../inference/ort/OrtDetector";
import { getDownloadStats } from "../lib/downloadInfo";
import { useAppStore } from "../store/useAppStore";
import type { DetectResult, Detection } from "../types";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./ui/card";
import { Slider } from "./ui/slider";
import { fmtNum, fmtMs } from "../lib/format";

type LiveSourceKind = "sample" | "file" | "camera";

const SAMPLE_VIDEO_URL = "/examples/val_pan.mp4";
const FPS_WINDOW_SIZE = 30;

function getOrtEpPrefFromUrl(): OrtEpPreference {
  if (typeof window === "undefined") return "auto";
  const param = new URLSearchParams(window.location.search).get("ep");
  if (param === "wasm" || param === "webgpu") return param;
  return "auto";
}

function drawDetectionsOverlay(
  canvas: HTMLCanvasElement,
  result: DetectResult,
  colors: string[],
  classes: string[],
): void {
  if (canvas.width !== result.width || canvas.height !== result.height) {
    canvas.width = result.width;
    canvas.height = result.height;
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const scale = Math.max(1, result.width / 640);
  const lineWidth = Math.max(2, Math.round(2 * scale));
  const fontSize = Math.round(13 * scale);
  ctx.font = `600 ${fontSize}px ui-monospace, SFMono-Regular, Menlo, monospace`;
  ctx.textBaseline = "top";

  for (const det of result.detections) {
    const [x1, y1, x2, y2] = det.box;
    const colorIdx = classes.indexOf(det.cls);
    const color =
      colors[colorIdx >= 0 ? colorIdx : det.cls_id % colors.length] ?? "#38bdf8";

    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.strokeRect(x1, y1, Math.max(1, x2 - x1), Math.max(1, y2 - y1));

    const label = `${det.cls} ${fmtNum(det.conf, 2)}`;
    const padX = Math.round(5 * scale);
    const padY = Math.round(2 * scale);
    const metrics = ctx.measureText(label);
    const boxW = Math.ceil(metrics.width + padX * 2);
    const boxH = Math.ceil(fontSize + padY * 2);
    const labelY = y1 - boxH >= 0 ? y1 - boxH : y1 + 2;

    ctx.fillStyle = color;
    ctx.fillRect(x1, labelY, boxW, boxH);
    ctx.fillStyle = "#090d16";
    ctx.fillText(label, x1 + padX, labelY + padY);
  }
}

function clearOverlay(canvas: HTMLCanvasElement | null): void {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (ctx) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
}

function safePlay(video: HTMLVideoElement | null): void {
  if (!video || typeof video.play !== "function") return;
  try {
    const res = video.play();
    if (res && typeof res.catch === "function") {
      void res.catch(() => {});
    }
  } catch {
    // ignore playback errors in headless/jsdom
  }
}

export function LiveView() {
  const { t } = useTranslation();
  const meta = useAppStore((s) => s.meta);
  const confThreshold = useAppStore((s) => s.confThreshold);
  const setConfThreshold = useAppStore((s) => s.setConfThreshold);
  const ortProgress = useAppStore((s) => s.ortProgress);
  const setOrtProgress = useAppStore((s) => s.setOrtProgress);
  const ortModelCached = useAppStore((s) => s.ortModelCached);
  const setOrtModelCached = useAppStore((s) => s.setOrtModelCached);

  const [sourceKind, setSourceKind] = React.useState<LiveSourceKind>("sample");
  const [videoSrc, setVideoSrc] = React.useState<string>(SAMPLE_VIDEO_URL);
  const [cameraActive, setCameraActive] = React.useState(false);
  const [paused, setPaused] = React.useState(false);
  const [errorMsg, setErrorMsg] = React.useState<string | null>(null);

  const [fps, setFps] = React.useState<number>(0);
  const [modelMs, setModelMs] = React.useState<number>(0);
  const [droppedFrames, setDroppedFrames] = React.useState<number>(0);
  const [framesProcessed, setFramesProcessed] = React.useState<number>(0);
  const [totalBoxesDrawn, setTotalBoxesDrawn] = React.useState<number>(0);
  const [activeDetections, setActiveDetections] = React.useState<Detection[]>([]);
  const [backendLabel, setBackendLabel] = React.useState<string>("—");
  const [wasmFallbackExpected, setWasmFallbackExpected] = React.useState<boolean>(
    () => getOrtEpPrefFromUrl() === "wasm",
  );

  const videoRef = React.useRef<HTMLVideoElement | null>(null);
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null);
  const fileInputRef = React.useRef<HTMLInputElement | null>(null);
  const objectUrlRef = React.useRef<string | null>(null);
  const streamRef = React.useRef<MediaStream | null>(null);

  const inFlightRef = React.useRef(false);
  const pausedRef = React.useRef(paused);
  pausedRef.current = paused;
  const confRef = React.useRef(confThreshold);
  confRef.current = confThreshold;
  const metaRef = React.useRef(meta);
  metaRef.current = meta;

  const timestampsRef = React.useRef<number[]>([]);
  const lastVideoTimeRef = React.useRef<number>(-1);
  const detectorRef = React.useRef<OrtDetector | null>(null);
  const mountedRef = React.useRef(true);

  const stopCameraStream = React.useCallback(() => {
    if (streamRef.current) {
      for (const track of streamRef.current.getTracks()) {
        track.stop();
      }
      streamRef.current = null;
    }
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
  }, []);

  const resetMetrics = React.useCallback(() => {
    timestampsRef.current = [];
    lastVideoTimeRef.current = -1;
    setFps(0);
    setModelMs(0);
    setDroppedFrames(0);
    setFramesProcessed(0);
    setTotalBoxesDrawn(0);
    setActiveDetections([]);
    clearOverlay(canvasRef.current);
  }, []);

  const handleSelectSample = React.useCallback(() => {
    setErrorMsg(null);
    stopCameraStream();
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }
    setSourceKind("sample");
    setVideoSrc(SAMPLE_VIDEO_URL);
    setPaused(false);
    resetMetrics();
  }, [resetMetrics, stopCameraStream]);

  const handleFileChange = React.useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      e.target.value = "";
      if (!file) return;
      setErrorMsg(null);
      stopCameraStream();
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
      }
      const url = URL.createObjectURL(file);
      objectUrlRef.current = url;
      setSourceKind("file");
      setVideoSrc(url);
      setPaused(false);
      resetMetrics();
    },
    [resetMetrics, stopCameraStream],
  );

  const handleStartCamera = React.useCallback(async () => {
    setErrorMsg(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setErrorMsg(t("live.cameraError"));
      return;
    }
    try {
      stopCameraStream();
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
        objectUrlRef.current = null;
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" } },
        audio: false,
      });
      if (!mountedRef.current) {
        for (const tr of stream.getTracks()) tr.stop();
        return;
      }
      streamRef.current = stream;
      setSourceKind("camera");
      setVideoSrc("");
      setCameraActive(true);
      setPaused(false);
      resetMetrics();

      const video = videoRef.current;
      if (video) {
        video.srcObject = stream;
        safePlay(video);
      }
    } catch (err) {
      const detail = err instanceof Error ? ` (${err.message})` : "";
      setErrorMsg(`${t("live.cameraError")}${detail}`);
    }
  }, [resetMetrics, stopCameraStream, t]);

  const handleTogglePause = React.useCallback(() => {
    setPaused((prev) => {
      const next = !prev;
      const video = videoRef.current;
      if (video && sourceKind !== "camera") {
        if (next) {
          video.pause();
        } else {
          safePlay(video);
        }
      }
      return next;
    });
  }, [sourceKind]);

  React.useEffect(() => {
    detectorRef.current = new OrtDetector(
      meta,
      (p) => setOrtProgress(p),
      getOrtEpPrefFromUrl(),
    );
  }, [meta, setOrtProgress]);

  React.useEffect(() => {
    let cancelled = false;
    if (getOrtEpPrefFromUrl() === "wasm") {
      setWasmFallbackExpected(true);
      return;
    }
    const gpu =
      typeof navigator !== "undefined"
        ? (
            navigator as Navigator & {
              gpu?: { requestAdapter?: () => Promise<unknown | null> };
            }
          ).gpu
        : undefined;
    if (!gpu || typeof gpu.requestAdapter !== "function") {
      setWasmFallbackExpected(true);
      return;
    }
    void gpu
      .requestAdapter()
      .then((adapter) => {
        if (!cancelled) {
          setWasmFallbackExpected(!adapter);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setWasmFallbackExpected(true);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const isWasmSlow =
    backendLabel === "ort-wasm" ||
    (backendLabel !== "ort-webgpu" && wasmFallbackExpected);

  React.useEffect(() => {
    const video = videoRef.current;
    if (!video || sourceKind === "camera" || !videoSrc) return;
    video.srcObject = null;
    video.src = videoSrc;
    if (!pausedRef.current) {
      safePlay(video);
    }
  }, [sourceKind, videoSrc]);

  React.useEffect(() => {
    mountedRef.current = true;
    let rafId = 0;

    const step = () => {
      rafId = window.requestAnimationFrame(step);
      const video = videoRef.current;
      if (!video || pausedRef.current) return;
      if (video.readyState < 2 || video.videoWidth <= 0 || video.videoHeight <= 0) {
        return;
      }
      if (sourceKind !== "camera" && video.ended) return;

      const curTime = video.currentTime;
      const hasNewVideoTime = Math.abs(curTime - lastVideoTimeRef.current) > 1e-4;

      if (inFlightRef.current) {
        if (hasNewVideoTime || sourceKind === "camera") {
          lastVideoTimeRef.current = curTime;
          setDroppedFrames((d) => d + 1);
        }
        return;
      }

      if (sourceKind !== "camera" && !hasNewVideoTime && timestampsRef.current.length > 0) {
        return;
      }

      const detector = detectorRef.current;
      if (!detector) return;

      inFlightRef.current = true;
      lastVideoTimeRef.current = curTime;

      void createImageBitmap(video, {
        colorSpaceConversion: "none",
        premultiplyAlpha: "none",
      })
        .then(async (bitmap) => {
          const result = await detector.detectBitmap(bitmap, confRef.current);
          if (!mountedRef.current) return;

          setOrtModelCached(true);
          setOrtProgress({ stage: "idle", percent: 0, message: "" });
          setBackendLabel(result.backend);
          setModelMs(result.timing.model_ms);
          setFramesProcessed((c) => c + 1);
          setActiveDetections(result.detections);
          setTotalBoxesDrawn((total) => total + result.detections.length);

          const now = performance.now();
          const list = timestampsRef.current;
          list.push(now);
          if (list.length > FPS_WINDOW_SIZE) {
            list.shift();
          }
          if (list.length >= 2) {
            const elapsedSec = (list[list.length - 1] - list[0]) / 1000;
            if (elapsedSec > 0) {
              setFps((list.length - 1) / elapsedSec);
            }
          }

          if (canvasRef.current) {
            drawDetectionsOverlay(
              canvasRef.current,
              result,
              metaRef.current.colors,
              metaRef.current.classes,
            );
          }
        })
        .catch((err) => {
          if (!mountedRef.current) return;
          setErrorMsg(err instanceof Error ? err.message : String(err));
        })
        .finally(() => {
          inFlightRef.current = false;
        });
    };

    rafId = window.requestAnimationFrame(step);
    return () => {
      window.cancelAnimationFrame(rafId);
    };
  }, [setOrtModelCached, setOrtProgress, sourceKind]);

  React.useEffect(() => {
    return () => {
      mountedRef.current = false;
      stopCameraStream();
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
        objectUrlRef.current = null;
      }
    };
  }, [stopCameraStream]);

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-3">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <CardTitle>{t("live.title")}</CardTitle>
              <CardDescription>{t("live.subtitle")}</CardDescription>
            </div>
            <Badge
              variant="outline"
              className="font-mono text-xs"
              data-testid="live-backend"
            >
              {backendLabel}
            </Badge>
          </div>
        </CardHeader>

        <CardContent className="space-y-4">
          {ortModelCached === false && ortProgress.stage === "idle" && (
            <div
              role="status"
              data-testid="live-download-notice"
              className="flex items-center gap-2 rounded-lg border border-sky-500/35 bg-sky-500/10 px-3 py-2 text-xs text-sky-900 dark:text-sky-200"
            >
              <Info className="h-4 w-4 shrink-0 text-[var(--accent)]" />
              <span>{t("detect.modelDownloadNotice", getDownloadStats())}</span>
            </div>
          )}

          {ortProgress.stage !== "idle" && ortProgress.stage !== "ready" && (
            <div
              className="rounded-lg border border-[var(--accent)]/40 bg-[var(--accent-soft)] p-3"
              role="status"
              data-testid="live-ort-progress"
            >
              <div className="mb-1.5 flex items-center justify-between text-xs font-medium">
                <span>{ortProgress.message}</span>
                <span className="font-mono tabular-nums">
                  {Math.round(ortProgress.percent)}%
                </span>
              </div>
              <div className="h-2 w-full overflow-hidden rounded-full bg-[var(--bg-muted)]">
                <div
                  className="h-full bg-[var(--accent)] transition-all"
                  style={{
                    width: `${Math.min(100, Math.max(4, ortProgress.percent))}%`,
                  }}
                />
              </div>
            </div>
          )}

          {/* Kaynak seçimi ve denetimler */}
          <div className="flex flex-wrap items-center gap-2">
            <Button
              type="button"
              size="sm"
              variant={sourceKind === "sample" ? "default" : "outline"}
              onClick={handleSelectSample}
              data-testid="live-sample-btn"
            >
              <FileVideo className="h-4 w-4" />
              <span>{t("live.loadSampleVideo")}</span>
            </Button>

            <input
              ref={fileInputRef}
              type="file"
              accept="video/mp4,video/webm,video/*"
              onChange={handleFileChange}
              className="hidden"
              data-testid="live-file-input"
            />
            <Button
              type="button"
              size="sm"
              variant={sourceKind === "file" ? "default" : "outline"}
              onClick={() => fileInputRef.current?.click()}
              data-testid="live-file-btn"
            >
              <Upload className="h-4 w-4" />
              <span>{t("live.pickVideoFile")}</span>
            </Button>

            {!cameraActive ? (
              <Button
                type="button"
                size="sm"
                variant={sourceKind === "camera" ? "default" : "outline"}
                onClick={() => void handleStartCamera()}
                data-testid="live-camera-btn"
              >
                <Camera className="h-4 w-4" />
                <span>{t("live.startCamera")}</span>
              </Button>
            ) : (
              <Button
                type="button"
                size="sm"
                variant="destructive"
                onClick={handleSelectSample}
                data-testid="live-stop-camera-btn"
              >
                <Square className="h-4 w-4" />
                <span>{t("live.stopCamera")}</span>
              </Button>
            )}

            <Button
              type="button"
              size="sm"
              variant="outline"
              onClick={handleTogglePause}
              data-testid="live-pause-btn"
              className="ml-auto"
            >
              {paused ? (
                <>
                  <Play className="h-4 w-4" />
                  <span>{t("live.resumeBtn")}</span>
                </>
              ) : (
                <>
                  <Pause className="h-4 w-4" />
                  <span>{t("live.pauseBtn")}</span>
                </>
              )}
            </Button>
          </div>

          <div
            data-testid="live-camera-note"
            className="flex items-start gap-2 rounded-lg border border-amber-500/35 bg-amber-500/10 px-3 py-2 text-xs text-amber-900 dark:text-amber-200"
          >
            <Camera className="mt-0.5 h-4 w-4 shrink-0 text-amber-500" />
            <span>{t("live.cameraNote")}</span>
          </div>

          {isWasmSlow && (
            <div
              role="alert"
              data-testid="live-wasm-warning"
              className="flex items-start gap-2 rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-xs font-medium text-red-700 dark:text-red-300"
            >
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-red-500" />
              <span>{t("live.wasmSlowWarning")}</span>
            </div>
          )}

          {errorMsg && (
            <div
              role="alert"
              data-testid="live-error"
              className="rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-xs text-red-700 dark:text-red-300"
            >
              {errorMsg}
            </div>
          )}

          {/* Eşik kaydırıcı + Canlı FPS telemetri şeridi */}
          <div className="grid grid-cols-1 gap-3 lg:grid-cols-12">
            <div className="elevated-card flex flex-col justify-center rounded-lg p-3 lg:col-span-4">
              <div className="mb-2 flex items-center justify-between text-xs">
                <span className="font-medium text-[var(--text-secondary)]">
                  {t("detect.confidenceLabel")}
                </span>
                <span
                  className="font-mono font-semibold tabular-nums text-[var(--text-primary)]"
                  data-testid="live-conf-value"
                >
                  {fmtNum(confThreshold, 3)}
                </span>
              </div>
              <Slider
                min={0.1}
                max={0.95}
                step={0.005}
                value={[confThreshold]}
                onValueChange={([v]) => setConfThreshold(v)}
                aria-label={t("detect.confidenceLabel")}
                data-testid="live-conf-slider"
              />
            </div>

            <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-4 lg:col-span-8">
              <div className="elevated-card rounded-lg p-3">
                <p className="text-[11px] text-[var(--text-muted)]">
                  {t("live.fpsLabel")}
                </p>
                <p
                  className="mt-0.5 font-mono text-lg font-semibold tabular-nums text-[var(--text-primary)]"
                  data-testid="live-fps"
                >
                  {fps > 0 ? fmtNum(fps, 2) : "—"}
                </p>
              </div>
              <div className="elevated-card rounded-lg p-3">
                <p className="text-[11px] text-[var(--text-muted)]">
                  {t("live.modelMsLabel")}
                </p>
                <p
                  className="mt-0.5 font-mono text-lg font-semibold tabular-nums text-[var(--text-primary)]"
                  data-testid="live-model-ms"
                >
                  {modelMs > 0 ? `${fmtMs(modelMs, 1)}` : "—"}
                </p>
              </div>
              <div className="elevated-card rounded-lg p-3">
                <p className="text-[11px] text-[var(--text-muted)]">
                  {t("live.droppedLabel")}
                </p>
                <p
                  className="mt-0.5 font-mono text-lg font-semibold tabular-nums text-[var(--text-primary)]"
                  data-testid="live-dropped"
                >
                  {droppedFrames}
                </p>
              </div>
              <div className="elevated-card rounded-lg p-3">
                <p className="text-[11px] text-[var(--text-muted)]">
                  {t("live.detCountLabel")}
                </p>
                <p
                  className="mt-0.5 font-mono text-lg font-semibold tabular-nums text-[var(--text-primary)]"
                  data-testid="live-det-count"
                  data-total-boxes={totalBoxesDrawn}
                  data-frames={framesProcessed}
                >
                  {activeDetections.length}
                </p>
              </div>
            </div>
          </div>

          {/* Video + Canvas Katmanı */}
          <div className="flex justify-center overflow-hidden rounded-lg border border-[var(--border-hairline)] bg-black">
            <div className="relative inline-block max-w-full">
              <video
                ref={videoRef}
                autoPlay
                muted
                playsInline
                loop={sourceKind !== "camera"}
                data-testid="live-video"
                className="block h-auto max-h-[68vh] w-auto max-w-full"
              />
              <canvas
                ref={canvasRef}
                data-testid="live-overlay-canvas"
                className="pointer-events-none absolute inset-0 h-full w-full"
              />
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
