"""Pytest ortak hook'ları: ağırlık gerektiren testleri eksik ağırlıkta temizce atlar."""

from __future__ import annotations

import os
import sys
from pathlib import Path
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tracera_core.config import ensure_weights_from_hf, weights_ready  # noqa: E402


_WEIGHTS_SKIPPED_COUNT = 0
_WEIGHTS_STATUS_MSG = ""


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """needs_weights işaretli testlerde yerel ağırlık yoksa HF Hub'dan indirmeyi dener, olmazsa atlar."""
    global _WEIGHTS_SKIPPED_COUNT, _WEIGHTS_STATUS_MSG

    needs_items = [item for item in items if item.get_closest_marker("needs_weights") is not None]
    if not needs_items:
        return

    if weights_ready():
        _WEIGHTS_STATUS_MSG = "yerel agirliklar mevcut"
        return

    downloaded = ensure_weights_from_hf()
    if downloaded:
        _WEIGHTS_STATUS_MSG = "TRACERA_MODEL_REPO uzerinden indirildi"
        return

    repo_env = os.environ.get("TRACERA_MODEL_REPO", "").strip()
    reason = (
        f"Model agirliklari yerel olarak yok ve TRACERA_MODEL_REPO='{repo_env or '(tanimsiz)'}' uzerinden indirilemedi; test atlandi."
    )
    skip_marker = pytest.mark.skip(reason=reason)
    for item in needs_items:
        item.add_marker(skip_marker)
    _WEIGHTS_SKIPPED_COUNT = len(needs_items)
    _WEIGHTS_STATUS_MSG = reason


def pytest_terminal_summary(terminalreporter, exitstatus: int, config: pytest.Config) -> None:
    if _WEIGHTS_SKIPPED_COUNT > 0:
        terminalreporter.write_sep(
            "-",
            f"[needs_weights] {_WEIGHTS_SKIPPED_COUNT} agirlik gerektiren test ATLANDI: {_WEIGHTS_STATUS_MSG}",
        )
