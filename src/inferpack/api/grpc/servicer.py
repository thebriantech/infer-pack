"""gRPC servicer implementation for InferPack PipelineService.

Maps every gRPC RPC to the same control-plane / execution-plane logic
used by the HTTP routes, ensuring identical behaviour over both transports.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import grpc

from inferpack.api.grpc import inferpack_pb2 as pb2
from inferpack.api.grpc import inferpack_pb2_grpc as pb2_grpc
from inferpack.core.errors import (
    PipelineActivationError,
    PipelineNotFoundError,
)
from inferpack.api.shared.pipeline_service import PipelineService, make_json_safe

logger = logging.getLogger(__name__)

class PipelineServicer(pb2_grpc.PipelineServiceServicer):
    """Concrete gRPC servicer backed by the shared application state."""

    def __init__(self, app_state: Any) -> None:
        """
        Args:
            app_state: The ``AppState`` instance (same one attached to
                       ``fastapi_app.state.app``).
        """
        self._state = app_state

    # ------------------------------------------------------------------
    # ListPipelines
    # ------------------------------------------------------------------

    async def ListPipelines(
        self,
        request: pb2.ListPipelinesRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb2.ListPipelinesResponse:
        service = PipelineService(self._state)
        summaries = service.list_pipelines()
        return pb2.ListPipelinesResponse(
            pipelines=[
                pb2.PipelineSummary(
                    name=p["name"],
                    description=p["description"],
                    version=p["version"],
                    builtin=p["builtin"],
                    state=p["state"],
                    models=p["models"],
                    stages=p["stages"],
                )
                for p in summaries
            ]
        )

    # ------------------------------------------------------------------
    # ActivatePipeline
    # ------------------------------------------------------------------

    async def ActivatePipeline(
        self,
        request: pb2.PipelineRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb2.PipelineStatusResponse:
        service = PipelineService(self._state)
        name = request.name
        try:
            await service.activate_pipeline(name)
        except PipelineNotFoundError:
            await context.abort(
                grpc.StatusCode.NOT_FOUND,
                f"Pipeline not found: {name}",
            )
        except PipelineActivationError as exc:
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                str(exc),
            )
        return pb2.PipelineStatusResponse(status="activated", pipeline=name)

    # ------------------------------------------------------------------
    # DeactivatePipeline
    # ------------------------------------------------------------------

    async def DeactivatePipeline(
        self,
        request: pb2.PipelineRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb2.PipelineStatusResponse:
        service = PipelineService(self._state)
        name = request.name
        try:
            await service.deactivate_pipeline(name)
        except PipelineNotFoundError:
            await context.abort(
                grpc.StatusCode.NOT_FOUND,
                f"Pipeline not found: {name}",
            )
        except PipelineActivationError as exc:
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                str(exc),
            )
        return pb2.PipelineStatusResponse(status="deactivated", pipeline=name)

    # ------------------------------------------------------------------
    # ExecutePipeline
    # ------------------------------------------------------------------

    async def ExecutePipeline(
        self,
        request: pb2.ExecutePipelineRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb2.ExecutePipelineResponse:
        name = request.name
        inputs: dict[str, Any] = {}
        for key, value in request.image_inputs.items():
            inputs[key] = value
        if request.inputs_json:
            try:
                inputs.update(json.loads(request.inputs_json))
            except json.JSONDecodeError as exc:
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"Invalid inputs_json: {exc}",
                )
        return await self._execute_pipeline_by_name(name, inputs, context)

    async def ExecuteFaceDetection(
        self,
        request: pb2.ExecuteFaceDetectionRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb2.ExecutePipelineResponse:
        inputs: dict[str, Any] = {"image_bytes": request.image}
        return await self._execute_pipeline_by_name("face_detection", inputs, context)

    async def ExecuteFaceComparison(
        self,
        request: pb2.ExecuteFaceComparisonRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb2.ExecutePipelineResponse:
        inputs: dict[str, Any] = {
            "image_bytes_1": request.image_1,
            "image_bytes_2": request.image_2,
        }
        return await self._execute_pipeline_by_name("face_comparison", inputs, context)

    async def _execute_pipeline_by_name(
        self,
        pipeline_name: str,
        inputs: dict[str, Any],
        context: grpc.aio.ServicerContext,
    ) -> pb2.ExecutePipelineResponse:
        service = PipelineService(self._state)

        try:
            result = await service.execute_pipeline(pipeline_name, inputs)
        except PipelineNotFoundError as exc:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        except PipelineActivationError as exc:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))

        safe_output = make_json_safe(result.output)

        return pb2.ExecutePipelineResponse(
            pipeline_name=result.pipeline_name,
            success=result.success,
            stages=[
                pb2.StageResult(
                    name=sr.stage_name,
                    status=sr.status.value,
                    duration_seconds=sr.duration_seconds,
                    error=sr.error or "",
                )
                for sr in result.stage_results
            ],
            total_duration_seconds=result.total_duration_seconds,
            error=result.error or "",
            output_json=json.dumps(safe_output),
        )
