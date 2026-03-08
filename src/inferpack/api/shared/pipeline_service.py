"""Pipeline service layer — transport-agnostic pipeline business logic."""

from __future__ import annotations

from typing import Any

from inferpack.core.errors import PipelineActivationError, PipelineNotFoundError
from inferpack.core.types import PipelineResult


class PipelineService:
    """Shared pipeline operations used by HTTP and gRPC APIs."""

    def __init__(self, app_state: Any) -> None:
        self._state = app_state

    def list_pipelines(self) -> list[dict[str, Any]]:
        return self._state.pipeline_registry.list_all()

    async def activate_pipeline(self, name: str) -> None:
        await self._state.pipeline_registry.activate(name)

    async def deactivate_pipeline(self, name: str) -> None:
        await self._state.pipeline_registry.deactivate(name)

    async def execute_pipeline(
        self,
        name: str,
        inputs: dict[str, Any],
    ) -> PipelineResult:
        try:
            pipeline_def = self._state.pipeline_registry.get(name)
        except PipelineNotFoundError:
            raise

        pipeline_state = self._state.pipeline_registry.get_state(name)
        if pipeline_state.value != "active":
            raise PipelineActivationError(
                f"Pipeline {name} is not active (state={pipeline_state.value})"
            )

        return await self._state.executor.execute(pipeline_def, inputs)


def make_json_safe(data: dict[str, Any]) -> dict[str, Any]:
    """Convert output to JSON-safe compact values.

    Important: do not inline full tensor payloads in API responses, as this
    can produce extremely large JSON bodies and make Swagger UI appear stuck.
    """
    import numpy as np

    def _safe(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return {
                "type": "ndarray",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }

        if isinstance(value, bytes):
            return f"<bytes, len={len(value)}>"

        if isinstance(value, dict):
            return {str(k): _safe(v) for k, v in value.items()}

        if isinstance(value, list):
            if value and isinstance(value[0], np.ndarray):
                return {
                    "type": "ndarray_list",
                    "count": len(value),
                    "items": [
                        {
                            "shape": list(arr.shape),
                            "dtype": str(arr.dtype),
                        }
                        for arr in value[:5]
                    ],
                    "truncated": len(value) > 5,
                }

            if value and isinstance(value[0], dict):
                return [_safe(v) for v in value]

            if len(value) <= 20:
                return [_safe(v) for v in value]

            return {
                "type": "list",
                "count": len(value),
                "items": [_safe(v) for v in value[:20]],
                "truncated": True,
            }

        return value

    safe_flat = {
        key: _safe(value)
        for key, value in data.items()
        if key not in {"image_bytes", "image_bytes_1", "image_bytes_2"}
    }

    image_1: dict[str, Any] = {}
    image_2: dict[str, Any] = {}
    comparison: dict[str, Any] = {}
    remaining: dict[str, Any] = {}

    comparison_keys = {"similarity_score", "is_match", "is_same_person"}

    for key, value in safe_flat.items():
        if key.endswith("_1"):
            image_1[key[:-2]] = value
        elif key.endswith("_2"):
            image_2[key[:-2]] = value
        elif key in comparison_keys:
            comparison[key] = value
        else:
            remaining[key] = value

    grouped: dict[str, Any] = dict(remaining)
    if image_1:
        grouped["image_1"] = image_1
    if image_2:
        grouped["image_2"] = image_2
    if comparison:
        grouped["comparison"] = comparison

    return grouped
