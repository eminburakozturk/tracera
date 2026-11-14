"""CONTRACT §2 ile uyumlu ortak Pydantic çıktı şemaları."""

from __future__ import annotations

from typing import Literal

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
    energy_in_box: float = Field(..., ge=0.0, le=1.0)
    peak_in_box: bool
    peak_xy: list[int] = Field(..., min_length=2, max_length=2)
