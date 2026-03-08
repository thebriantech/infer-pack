"""Model management routes — list models, view state."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from inferpack.api.http.schemas import ModelListResponse, ModelSummary
from inferpack.core.errors import ModelNotFoundError

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=ModelListResponse)
async def list_models(request: Request) -> ModelListResponse:
    """List all registered models and their runtime state."""
    state = request.app.state.app
    return ModelListResponse(
        models=[
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
    )


@router.get("/{name}", response_model=ModelSummary)
async def get_model(name: str, request: Request) -> ModelSummary:
    """Get details of a single model."""
    state = request.app.state.app
    try:
        m = state.model_registry.get(name)
    except ModelNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model not found: {name}")
    return ModelSummary(
        name=m.name,
        state=m.state.value,
        ref_count=m.ref_count,
        active_inferences=m.active_inferences,
        last_used=m.last_used,
        gpu_memory_bytes=m.gpu_memory_bytes,
    )
