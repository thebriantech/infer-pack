"""Pydantic schemas for API request / response bodies."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class PipelineSummary(BaseModel):
    name: str
    description: str
    version: str
    builtin: bool
    state: str
    models: list[str]
    stages: list[str]


class PipelineListResponse(BaseModel):
    pipelines: list[PipelineSummary]


class PipelineActivateRequest(BaseModel):
    """Empty body — pipeline name comes from the path."""


class PipelineExecuteRequest(BaseModel):
    """Inputs for pipeline execution (key-value pairs)."""
    inputs: dict[str, Any] = Field(default_factory=dict)


class StageResultSchema(BaseModel):
    name: str
    status: str
    duration_seconds: float
    error: str | None = None


class PipelineExecuteResponse(BaseModel):
    result: dict[str, Any] = Field(default_factory=dict)


class DetectionSchema(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    landmarks: list[list[float]] = Field(default_factory=list)


class FaceDetectionResultSchema(BaseModel):
    detections: list[DetectionSchema] = Field(default_factory=list)


class FaceDetectionExecuteResponse(BaseModel):
    result: FaceDetectionResultSchema


class FaceComparisonMetricsSchema(BaseModel):
    similarity_score: float | None = None
    is_match: bool | None = None
    is_same_person: bool | None = None


class FaceComparisonImageInfoSchema(BaseModel):
    original_size: list[int] | None = None
    det_scale: float | None = None
    detections: list[DetectionSchema] = Field(default_factory=list)
    preprocessed_image: dict[str, Any] | None = None
    cropped_faces: dict[str, Any] | None = None
    embeddings: dict[str, Any] | None = None


class FaceComparisonResultSchema(BaseModel):
    comparison: FaceComparisonMetricsSchema
    image_1: FaceComparisonImageInfoSchema | None = None
    image_2: FaceComparisonImageInfoSchema | None = None


class FaceComparisonExecuteResponse(BaseModel):
    result: FaceComparisonResultSchema


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ModelSummary(BaseModel):
    name: str
    state: str
    ref_count: int
    active_inferences: int
    last_used: float
    gpu_memory_bytes: int


class ModelListResponse(BaseModel):
    models: list[ModelSummary]


# ---------------------------------------------------------------------------
# System / health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str
    triton_ready: bool


class SystemStatusResponse(BaseModel):
    models: list[ModelSummary]
    pipelines: list[PipelineSummary]
    triton_ready: bool


class EventEntry(BaseModel):
    timestamp: float
    level: str
    message: str


class EventsResponse(BaseModel):
    events: list[EventEntry]
