"""Core types and enumerations for InferPack."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------

class ModelState(str, Enum):
    """Model lifecycle states.

    State machine::

        UNLOADED ─► LOADING ─► WARMING ─► READY ◄─► IDLE ─► UNLOADING ─► UNLOADED

    Valid transitions:
        UNLOADED  → LOADING
        LOADING   → WARMING  | UNLOADED  (on error)
        WARMING   → READY    | UNLOADED  (on error)
        READY     → IDLE     (ref_count drops to 0)
        IDLE      → READY    (ref_count increases from 0)
        IDLE      → UNLOADING (idle timeout expired)
        UNLOADING → UNLOADED
    """

    UNLOADED = "unloaded"
    LOADING = "loading"
    WARMING = "warming"
    READY = "ready"
    IDLE = "idle"
    UNLOADING = "unloading"


MODEL_STATE_TRANSITIONS: dict[ModelState, set[ModelState]] = {
    ModelState.UNLOADED: {ModelState.LOADING},
    ModelState.LOADING: {ModelState.WARMING, ModelState.UNLOADED},
    ModelState.WARMING: {ModelState.READY, ModelState.UNLOADED},
    ModelState.READY: {ModelState.IDLE},
    ModelState.IDLE: {ModelState.READY, ModelState.UNLOADING},
    ModelState.UNLOADING: {ModelState.UNLOADED},
}


# ---------------------------------------------------------------------------
# Pipeline lifecycle
# ---------------------------------------------------------------------------

class PipelineState(str, Enum):
    """Pipeline activation states."""

    INACTIVE = "inactive"
    ACTIVATING = "activating"
    ACTIVE = "active"
    DEACTIVATING = "deactivating"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

class StageStatus(str, Enum):
    """Execution status of a single pipeline stage."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMED_OUT = "timed_out"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModelInfo:
    """Runtime information about a registered model."""

    name: str
    state: ModelState = ModelState.UNLOADED
    ref_count: int = 0
    active_inferences: int = 0
    last_used: float = 0.0
    gpu_memory_bytes: int = 0
    error_message: str | None = None

    def touch(self) -> None:
        """Update last-used timestamp to now."""
        self.last_used = time.time()


@dataclass
class StageDefinition:
    """Declarative definition of a single pipeline stage."""

    name: str
    stage_type: str
    config: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 30.0
    max_retries: int = 0
    retry_delay_seconds: float = 1.0  # base delay; multiplied by attempt number (linear backoff)


@dataclass
class PipelineDefinition:
    """Declarative definition of a complete pipeline (loaded from YAML)."""

    name: str
    description: str
    version: str
    models: list[str]
    stages: list[StageDefinition]
    builtin: bool = False

    @property
    def model_names(self) -> set[str]:
        """Set of unique model names required by this pipeline."""
        return set(self.models)


@dataclass
class StageResult:
    """Result of a single stage execution."""

    stage_name: str
    status: StageStatus
    duration_seconds: float = 0.0
    error: str | None = None
    output_keys: list[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    """Result of a complete pipeline execution."""

    pipeline_name: str
    success: bool
    stage_results: list[StageResult] = field(default_factory=list)
    total_duration_seconds: float = 0.0
    error: str | None = None
    output: dict[str, Any] = field(default_factory=dict)
