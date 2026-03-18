"""Pydantic schemas for API request / response bodies."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class PipelineSummary(BaseModel):
    """Summary representation of a single registered pipeline."""

    name: str
    description: str
    version: str
    builtin: bool
    state: str
    models: list[str]
    stages: list[str]


class PipelineListResponse(BaseModel):
    """Response body for the list-pipelines endpoint."""

    pipelines: list[PipelineSummary]


class PipelineActivateRequest(BaseModel):
    """Empty body — pipeline name comes from the path."""


class PipelineExecuteRequest(BaseModel):
    """Inputs for pipeline execution (key-value pairs)."""
    inputs: dict[str, Any] = Field(default_factory=dict)


class StageResultSchema(BaseModel):
    """Serialised result of a single pipeline stage."""

    name: str
    status: str
    duration_seconds: float
    error: str | None = None


class PipelineExecuteResponse(BaseModel):
    """Response body for a generic pipeline execution request."""

    result: dict[str, Any] = Field(default_factory=dict)


class DetectionSchema(BaseModel):
    """Bounding-box detection result including confidence and optional landmarks."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    landmarks: list[list[float]] = Field(default_factory=list)


class FaceDetectionResultSchema(BaseModel):
    """Structured output of the face detection pipeline."""

    detections: list[DetectionSchema] = Field(default_factory=list)


class FaceDetectionExecuteResponse(BaseModel):
    """Response body for the face detection pipeline endpoint."""

    result: FaceDetectionResultSchema


class FaceComparisonMetricsSchema(BaseModel):
    """Similarity score and match verdict from a face comparison pipeline."""

    similarity_score: float | None = None
    is_match: bool | None = None
    is_same_person: bool | None = None


class FaceComparisonImageInfoSchema(BaseModel):
    """Per-image metadata and intermediate outputs from a face comparison pipeline."""

    original_size: list[int] | None = None
    det_scale: float | None = None
    detections: list[DetectionSchema] = Field(default_factory=list)
    preprocessed_image: dict[str, Any] | None = None
    cropped_faces: dict[str, Any] | None = None
    embeddings: dict[str, Any] | None = None


class FaceComparisonResultSchema(BaseModel):
    """Aggregated result of the face comparison pipeline including per-image details."""

    comparison: FaceComparisonMetricsSchema
    image_1: FaceComparisonImageInfoSchema | None = None
    image_2: FaceComparisonImageInfoSchema | None = None


class FaceComparisonExecuteResponse(BaseModel):
    """Response body for the face comparison pipeline endpoint."""

    result: FaceComparisonResultSchema


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ModelSummary(BaseModel):
    """Runtime state snapshot for a single registered model."""

    name: str
    state: str
    ref_count: int
    active_inferences: int
    last_used: float
    gpu_memory_bytes: int


class ModelListResponse(BaseModel):
    """Response body for the list-models endpoint."""

    models: list[ModelSummary]


# ---------------------------------------------------------------------------
# System / health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Response body for the liveness / readiness health endpoint."""

    status: str
    version: str
    triton_ready: bool


class SystemStatusResponse(BaseModel):
    """Response body for the aggregate system status endpoint."""

    models: list[ModelSummary]
    pipelines: list[PipelineSummary]
    triton_ready: bool


class EventEntry(BaseModel):
    """A single timestamped log or system event."""

    timestamp: float
    level: str
    message: str


class EventsResponse(BaseModel):
    """Response body for the events listing endpoint."""

    events: list[EventEntry]
