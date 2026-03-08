"""Health and system status routes."""

from __future__ import annotations

from fastapi import APIRouter, Request

from inferpack import __version__
from inferpack.api.http.schemas import (
    HealthResponse,
    ModelSummary,
    PipelineSummary,
    SystemStatusResponse,
)

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Liveness / readiness check."""
    state = request.app.state.app
    triton_ready = await state.triton_client.is_server_ready()
    return HealthResponse(
        status="ok" if triton_ready else "degraded",
        version=__version__,
        triton_ready=triton_ready,
    )


@router.get("/status", response_model=SystemStatusResponse)
async def system_status(request: Request) -> SystemStatusResponse:
    """Aggregate system status: models, pipelines, Triton health."""
    state = request.app.state.app
    models = [
        ModelSummary(
            name=m.name,
            state=m.state.value,
            ref_count=m.ref_count,
            active_inferences=m.active_inferences,
            last_used=m.last_used,
            gpu_memory_bytes=m.gpu_memory_bytes,
        )
        for m in state.model_registry.list_all()
    ]
    pipelines = [
        PipelineSummary(**p) for p in state.pipeline_registry.list_all()
    ]
    triton_ready = await state.triton_client.is_server_ready()
    return SystemStatusResponse(
        models=models,
        pipelines=pipelines,
        triton_ready=triton_ready,
    )
