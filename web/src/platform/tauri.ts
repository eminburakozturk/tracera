export type TauriPlatform = "windows" | "android" | "other";

export interface SidecarStatus {
  port: number;
  base_url: string;
  ready: boolean;
  startup_ms: number | null;
  error: string | null;
  available?: boolean;
}

export interface ApiReadyPayload {
  port: number;
  base_url: string;
  startup_ms: number;
}

export function hasTauriGlobalHint(): boolean {
  if (typeof globalThis === "undefined") return false;
  const g = globalThis as Record<string, unknown>;
  return Boolean(g.isTauri || g.__TAURI_INTERNALS__ || g.__TAURI__);
}

export async function isTauriApp(): Promise<boolean> {
  if (!hasTauriGlobalHint()) {
    return false;
  }
  try {
    const { isTauri } = await import("@tauri-apps/api/core");
    return Boolean(isTauri());
  } catch {
    return false;
  }
}

export async function getPlatform(): Promise<TauriPlatform> {
  try {
    const { invoke } = await import("@tauri-apps/api/core");
    const plat = await invoke<string>("get_platform");
    if (plat === "windows" || plat === "android") {
      return plat;
    }
    return "other";
  } catch {
    return "other";
  }
}

export async function getApiBaseUrl(): Promise<string> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<string>("get_api_base_url");
}

export async function getSidecarStatus(): Promise<SidecarStatus> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<SidecarStatus>("get_sidecar_status");
}

export async function listenApiReady(
  onReady: (payload: ApiReadyPayload) => void,
): Promise<() => void> {
  const { listen } = await import("@tauri-apps/api/event");
  return listen<ApiReadyPayload>("tracera://api-ready", (event) => {
    if (event?.payload) {
      onReady(event.payload);
    }
  });
}

export interface ConnectSidecarCallbacks {
  onStarting: () => void;
  onReady: (baseUrl: string, startupMs: number | null) => void;
  onError: (error: string) => void;
  onUnavailable?: (platform: TauriPlatform) => void;
  pollIntervalMs?: number;
}

export async function connectTauriSidecar(
  callbacks: ConnectSidecarCallbacks,
): Promise<() => void> {
  let settled = false;
  let unlisten: (() => void) | null = null;
  let pollTimer: ReturnType<typeof setInterval> | null = null;

  const cleanup = () => {
    if (pollTimer !== null) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
    if (unlisten) {
      unlisten();
      unlisten = null;
    }
  };

  const finishReady = async (baseUrl?: string | null, startupMs?: number | null) => {
    if (settled) return;
    settled = true;
    cleanup();
    const resolvedUrl =
      baseUrl && baseUrl.trim().length > 0 ? baseUrl : await getApiBaseUrl();
    callbacks.onReady(resolvedUrl, startupMs ?? null);
  };

  const finishError = (errMsg: string) => {
    if (settled) return;
    settled = true;
    cleanup();
    callbacks.onError(errMsg);
  };

  const finishUnavailable = (platform: TauriPlatform) => {
    if (settled) return;
    settled = true;
    cleanup();
    callbacks.onUnavailable?.(platform);
  };

  const platform = await getPlatform();
  if (platform === "android") {
    finishUnavailable(platform);
    return cleanup;
  }

  unlisten = await listenApiReady((payload) => {
    void finishReady(payload.base_url, payload.startup_ms);
  });

  try {
    const status = await getSidecarStatus();
    if (status.available === false) {
      finishUnavailable(platform);
      return cleanup;
    }
    if (status.error) {
      finishError(status.error);
      return cleanup;
    }
    if (status.ready) {
      await finishReady(status.base_url, status.startup_ms);
      return cleanup;
    }
  } catch (err) {
    finishError(err instanceof Error ? err.message : String(err));
    return cleanup;
  }

  if (!settled) {
    callbacks.onStarting();
    const pollMs = callbacks.pollIntervalMs ?? 1500;
    if (pollMs > 0) {
      pollTimer = setInterval(() => {
        void getSidecarStatus()
          .then((st) => {
            if (settled) return;
            if (st.error) {
              finishError(st.error);
            } else if (st.ready) {
              void finishReady(st.base_url, st.startup_ms);
            }
          })
          .catch(() => {
            // keep waiting for event unless invoke rejects explicitly on next tick
          });
      }, pollMs);
    }
  }

  return cleanup;
}
