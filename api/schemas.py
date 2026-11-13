"""CONTRACT §2 ve §3 ile birebir uyumlu Pydantic şemaları."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from tracera_core.schemas import Detection, DetectResult, ExplainResult, Timing


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model: str
    version: str
    device: str
    backends: list[str]


class ErrorResponse(BaseModel):
    error: str


class MetricsResponse(BaseModel):
    operating_point: list[dict[str, Any]]
    edge_benchmark: list[dict[str, Any]]
    cv: dict[str, Any]
    xai: list[dict[str, Any]]
    aoi: dict[str, Any]
    final_val: list[dict[str, Any]] = Field(default_factory=list)
    final_test: list[dict[str, Any]] = Field(default_factory=list)
    web_spike: list[dict[str, Any]] = Field(default_factory=list)


__all__ = [
    "Detection",
    "DetectResult",
    "ErrorResponse",
    "ExplainResult",
    "HealthResponse",
    "MetricsResponse",
    "Timing",
]
