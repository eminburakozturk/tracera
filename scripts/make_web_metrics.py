"""results/*.csv dosyalarından web/public/metrics.json (/metrics ile birebir) üretir."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.metrics_loader import build_metrics_payload  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
OUT_PATH = REPO_ROOT / "web" / "public" / "metrics.json"


def main() -> None:
    payload = build_metrics_payload(RESULTS_DIR)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Kaydedildi: {OUT_PATH} ({OUT_PATH.stat().st_size} bayt)")


if __name__ == "__main__":
    main()
