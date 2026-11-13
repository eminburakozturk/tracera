export const DEFAULT_CLOUD_SPACE = "ebboo/tracera";
export const CLOUD_UNAVAILABLE_MESSAGE = "Bulut açıklaması şu an kullanılamıyor";
export const DEFAULT_CLOUD_TIMEOUT_MS = 65_000;
export const DEFAULT_CLOUD_RETRIES = 1;
export const DEFAULT_CLOUD_RETRY_DELAY_MS = 500;
export const DEFAULT_WAKING_HINT_MS = 2_500;

export class CloudUnavailableError extends Error {
  constructor(message = CLOUD_UNAVAILABLE_MESSAGE) {
    super(message);
    this.name = "CloudUnavailableError";
  }
}

export interface GradioPredictableClient {
  predict: (
    endpoint: string | number,
    data: unknown[] | Record<string, unknown> | undefined,
  ) => Promise<{ data: unknown }>;
}

export type CloudConnector = (spaceId: string) => Promise<GradioPredictableClient>;

async function loadGradioClient() {
  return import("@gradio/client");
}

const defaultConnector: CloudConnector = async (spaceId: string) => {
  const { Client } = await loadGradioClient();
  return (await Client.connect(spaceId)) as unknown as GradioPredictableClient;
};

async function withTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => {
          reject(new CloudUnavailableError(CLOUD_UNAVAILABLE_MESSAGE));
        }, timeoutMs);
      }),
    ]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

export interface CloudCallOptions {
  connector?: CloudConnector;
  timeoutMs?: number;
  retries?: number;
  retryDelayMs?: number;
  wakingThresholdMs?: number;
  onWaking?: () => void;
}

function sleep(ms: number): Promise<void> {
  if (ms <= 0) return Promise.resolve();
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function callCloudEndpoint<T>(
  spaceId: string,
  endpoint: "/detect_api" | "/explain_api",
  payload: Record<string, unknown>,
  options?: CloudCallOptions,
): Promise<T> {
  const targetSpace = spaceId.trim() || DEFAULT_CLOUD_SPACE;
  const connector = options?.connector ?? defaultConnector;
  const timeoutMs = options?.timeoutMs ?? DEFAULT_CLOUD_TIMEOUT_MS;
  const retries = options?.retries ?? DEFAULT_CLOUD_RETRIES;
  const retryDelayMs = options?.retryDelayMs ?? DEFAULT_CLOUD_RETRY_DELAY_MS;
  const wakingThresholdMs = options?.wakingThresholdMs ?? DEFAULT_WAKING_HINT_MS;

  let wakingNotified = false;
  const notifyWaking = () => {
    if (!wakingNotified) {
      wakingNotified = true;
      options?.onWaking?.();
    }
  };

  for (let attempt = 0; attempt <= retries; attempt++) {
    let wakingTimer: ReturnType<typeof setTimeout> | undefined;
    if (options?.onWaking && !wakingNotified && wakingThresholdMs >= 0) {
      wakingTimer = setTimeout(notifyWaking, wakingThresholdMs);
    }

    try {
      const client = await withTimeout(connector(targetSpace), timeoutMs);
      const result = await withTimeout(
        client.predict(endpoint, payload),
        timeoutMs,
      );
      const dataArr = Array.isArray(result?.data) ? result.data : [result?.data];
      const first = dataArr[0];
      if (!first || typeof first !== "object") {
        throw new CloudUnavailableError(CLOUD_UNAVAILABLE_MESSAGE);
      }
      return first as T;
    } catch (err) {
      if (attempt < retries) {
        notifyWaking();
        await sleep(retryDelayMs);
        continue;
      }
      if (err instanceof CloudUnavailableError) {
        throw err;
      }
      throw new CloudUnavailableError(CLOUD_UNAVAILABLE_MESSAGE);
    } finally {
      if (wakingTimer !== undefined) clearTimeout(wakingTimer);
    }
  }

  throw new CloudUnavailableError(CLOUD_UNAVAILABLE_MESSAGE);
}

export async function prepareGradioFile(file: File): Promise<unknown> {
  const { handle_file } = await loadGradioClient();
  return handle_file(file);
}
