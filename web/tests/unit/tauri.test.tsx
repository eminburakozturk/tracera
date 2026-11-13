import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "../../src/App";
import { DEFAULT_API_BASE_URL } from "../../src/inference/api/ApiDetector";
import { resolveExplain } from "../../src/inference/explainResolver";
import {
  hasTauriGlobalHint,
  isTauriApp,
  type ApiReadyPayload,
  type SidecarStatus,
} from "../../src/platform/tauri";
import { useAppStore } from "../../src/store/useAppStore";

const mockInvoke = vi.fn();
const mockIsTauri = vi.fn();
const mockListen = vi.fn();
let readyListener: ((ev: { payload: ApiReadyPayload }) => void) | null = null;
const mockUnlisten = vi.fn();

vi.mock("@tauri-apps/api/core", () => ({
  isTauri: () => mockIsTauri(),
  invoke: (cmd: string, args?: unknown) => mockInvoke(cmd, args),
}));

vi.mock("@tauri-apps/api/event", () => ({
  listen: async (
    event: string,
    cb: (ev: { payload: ApiReadyPayload }) => void,
  ) => {
    mockListen(event, cb);
    if (event === "tracera://api-ready") {
      readyListener = cb;
    }
    return mockUnlisten;
  },
}));

describe("Tauri v2 köprü entegrasyonu (CONTRACT §5)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    readyListener = null;
    delete (globalThis as Record<string, unknown>).isTauri;
    delete (globalThis as Record<string, unknown>).__TAURI_INTERNALS__;

    useAppStore.setState({
      activeTab: "detect",
      lang: "tr",
      backendChoice: "ort",
      apiBaseUrl: DEFAULT_API_BASE_URL,
      tauriMode: false,
      tauriPlatform: null,
      localApiAvailable: true,
      tauriSidecarStatus: "idle",
      tauriSidecarError: null,
      selectedImage: null,
      detectResult: null,
      detectError: null,
    });

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith("/health")) {
          return new Response(
            JSON.stringify({
              status: "ok",
              model: "yolo11s",
              version: "1",
              device: "CPU",
              backends: ["openvino-int8"],
            }),
            { status: 200 },
          );
        }
        return new Response(JSON.stringify({}), { status: 404 });
      }),
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    delete (globalThis as Record<string, unknown>).isTauri;
    delete (globalThis as Record<string, unknown>).__TAURI_INTERNALS__;
  });

  it("does not load or call @tauri-apps/api outside Tauri", async () => {
    expect(hasTauriGlobalHint()).toBe(false);
    const inTauri = await isTauriApp();
    expect(inTauri).toBe(false);
    expect(mockIsTauri).not.toHaveBeenCalled();
    expect(mockInvoke).not.toHaveBeenCalled();
  });

  it("1) hazır: resolves dynamic API URL immediately and sets default engine to api", async () => {
    (globalThis as Record<string, unknown>).isTauri = true;
    mockIsTauri.mockReturnValue(true);

    const status: SidecarStatus = {
      port: 53817,
      base_url: "http://127.0.0.1:53817",
      ready: true,
      startup_ms: 1420,
      error: null,
    };
    mockInvoke.mockImplementation(async (cmd: string) => {
      if (cmd === "get_sidecar_status") return status;
      if (cmd === "get_api_base_url") return status.base_url;
      throw new Error(`unexpected cmd: ${cmd}`);
    });

    render(<App />);

    await waitFor(() => {
      const state = useAppStore.getState();
      expect(state.tauriMode).toBe(true);
      expect(state.tauriSidecarStatus).toBe("ready");
      expect(state.backendChoice).toBe("api");
      expect(state.apiBaseUrl).toBe("http://127.0.0.1:53817");
    });

    expect(screen.queryByTestId("tauri-starting-screen")).not.toBeInTheDocument();
    expect(screen.queryByTestId("tauri-error-screen")).not.toBeInTheDocument();
    expect(mockUnlisten).toHaveBeenCalled();
  });

  it("2) bekliyor → olay geldi: shows starting screen while ready=false then transitions on tracera://api-ready", async () => {
    (globalThis as Record<string, unknown>).isTauri = true;
    mockIsTauri.mockReturnValue(true);

    const status: SidecarStatus = {
      port: 0,
      base_url: "",
      ready: false,
      startup_ms: null,
      error: null,
    };
    mockInvoke.mockImplementation(async (cmd: string) => {
      if (cmd === "get_sidecar_status") return status;
      throw new Error(`unexpected cmd: ${cmd}`);
    });

    render(<App />);

    await waitFor(() => {
      expect(screen.getByTestId("tauri-starting-screen")).toHaveTextContent(
        "Yerel motor başlatılıyor… (ilk açılış ~20 sn)",
      );
    });
    expect(useAppStore.getState().backendChoice).toBe("api");
    expect(readyListener).not.toBeNull();

    await act(async () => {
      readyListener?.({
        payload: {
          port: 54123,
          base_url: "http://127.0.0.1:54123",
          startup_ms: 18200,
        },
      });
    });

    await waitFor(() => {
      expect(screen.queryByTestId("tauri-starting-screen")).not.toBeInTheDocument();
      const state = useAppStore.getState();
      expect(state.tauriSidecarStatus).toBe("ready");
      expect(state.apiBaseUrl).toBe("http://127.0.0.1:54123");
      expect(state.backendChoice).toBe("api");
    });
    expect(mockUnlisten).toHaveBeenCalled();
  });

  it("3) hata: surfaces sidecar error message when get_sidecar_status returns error", async () => {
    (globalThis as Record<string, unknown>).isTauri = true;
    mockIsTauri.mockReturnValue(true);

    const status: SidecarStatus = {
      port: 0,
      base_url: "",
      ready: false,
      startup_ms: null,
      error: "Python yan sunucusu port 53817'ye bağlanamadı",
    };
    mockInvoke.mockResolvedValue(status);

    render(<App />);

    await waitFor(() => {
      const errScreen = screen.getByTestId("tauri-error-screen");
      expect(errScreen).toHaveTextContent(
        "Python yan sunucusu port 53817'ye bağlanamadı",
      );
    });
    expect(useAppStore.getState().tauriSidecarStatus).toBe("error");
    expect(mockUnlisten).toHaveBeenCalled();
  });

  it("4) android (CONTRACT v1.2): get_platform=android hides starting screen, defaults to ort, hides api option, and restricts explain to Cloud", async () => {
    (globalThis as Record<string, unknown>).isTauri = true;
    mockIsTauri.mockReturnValue(true);

    const status: SidecarStatus = {
      port: 0,
      base_url: "",
      ready: false,
      startup_ms: null,
      error: null,
      available: false,
    };
    mockInvoke.mockImplementation(async (cmd: string) => {
      if (cmd === "get_platform") return "android";
      if (cmd === "get_sidecar_status") return status;
      throw new Error(`unexpected cmd: ${cmd}`);
    });

    render(<App />);

    await waitFor(() => {
      const state = useAppStore.getState();
      expect(state.tauriMode).toBe(true);
      expect(state.tauriPlatform).toBe("android");
      expect(state.localApiAvailable).toBe(false);
      expect(state.backendChoice).toBe("ort");
    });

    expect(screen.queryByTestId("tauri-starting-screen")).not.toBeInTheDocument();
    expect(screen.queryByTestId("tauri-error-screen")).not.toBeInTheDocument();
    expect(mockListen).not.toHaveBeenCalled();

    // Backend seçicisinde "Yerel sunucu" gizli olmalı
    const trigger = screen.getByTestId("backend-select");
    fireEvent.click(trigger);
    expect(screen.queryByText(/Yerel sunucu/i)).not.toBeInTheDocument();

    // Isı haritası yalnız Bulut'a gider, yerel API'ye dokunmaz
    const apiExplain = vi.fn();
    const cloudExplain = vi.fn(async () => ({
      overlay_png: "",
      zoom_png: "",
      method: "layercam" as const,
      layer: 4 as const,
      timing: { pre_ms: 1, model_ms: 10, post_ms: 1, total_ms: 12 },
    }));
    const dummyFile = new File(["img"], "x.jpg", { type: "image/jpeg" });
    const resolved = await resolveExplain({
      backendChoice: "ort",
      apiBaseUrl: DEFAULT_API_BASE_URL,
      cloudSpaceId: "ebboo/tracera",
      file: dummyFile,
      detection: {
        id: 1,
        cls: "short",
        cls_id: 3,
        conf: 0.9,
        box: [0, 0, 10, 10],
      },
      cloudOnly: true,
      factories: {
        api: () => ({ explain: apiExplain }),
        cloud: () => ({ explain: cloudExplain }),
      },
    });
    expect(resolved.source).toBe("cloud");
    expect(apiExplain).not.toHaveBeenCalled();
    expect(cloudExplain).toHaveBeenCalledTimes(1);
  });

  it("5) no-sidecar fallback (CONTRACT v1.2): available=false without get_platform=android also switches to ORT without starting screen", async () => {
    (globalThis as Record<string, unknown>).isTauri = true;
    mockIsTauri.mockReturnValue(true);

    const status: SidecarStatus = {
      port: 0,
      base_url: "",
      ready: false,
      startup_ms: null,
      error: null,
      available: false,
    };
    mockInvoke.mockImplementation(async (cmd: string) => {
      if (cmd === "get_platform") return "other";
      if (cmd === "get_sidecar_status") return status;
      throw new Error(`unexpected cmd: ${cmd}`);
    });

    render(<App />);

    await waitFor(() => {
      const state = useAppStore.getState();
      expect(state.tauriMode).toBe(true);
      expect(state.localApiAvailable).toBe(false);
      expect(state.backendChoice).toBe("ort");
    });

    expect(screen.queryByTestId("tauri-starting-screen")).not.toBeInTheDocument();
  });
});
