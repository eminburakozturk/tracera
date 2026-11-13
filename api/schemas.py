"""CONTRACT §2 ve §3 ile birebir uyumlu Pydantic şemaları."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Detection(BaseModel):
    id: int
    cls: str
    cls_id: int
    conf: float
    box: list[float] = Field(..., min_length=4, max_length=4)


class Timing(BaseModel):
    pre_ms: float
    model_ms: float
    post_ms: float
    total_ms: float


class DetectResult(BaseModel):
    width: int
    height: int
    detections: list[Detection]
    timing: Timing
    backend: str


class ExplainResult(BaseModel):
    overlay_png: str
    zoom_png: str
    method: Literal["layercam"] = "layercam"
    layer: Literal[4] = 4
    timing: Timing


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
