export type DownloadStats = {
  modelMb: number;
  runtimeMb: number;
  totalMb: number;
};

const FALLBACK_STATS: DownloadStats = {
  modelMb: 36,
  runtimeMb: 28,
  totalMb: 65,
};

export function getDownloadStats(): DownloadStats {
  if (
    typeof __TRACERA_DOWNLOAD_INFO__ !== "undefined" &&
    __TRACERA_DOWNLOAD_INFO__ &&
    typeof __TRACERA_DOWNLOAD_INFO__.totalMb === "number"
  ) {
    return __TRACERA_DOWNLOAD_INFO__;
  }
  return FALLBACK_STATS;
}
