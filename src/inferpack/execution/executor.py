"""Pipeline Executor — runs an entire pipeline's stages sequentially.

This is the top-level Execution Plane component.  It:
    • Resolves stage types from the stage registry
    • Creates a ``PipelineExecutionContext``
    • Runs stages one-by-one via the stage runner
    • Collects and returns a ``PipelineResult``

The executor makes NO resource decisions and has NO Triton internals.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from inferpack.config import ExecutionConfig
from inferpack.core.errors import PipelineError, StageNotFoundError
from inferpack.core.types import (
    PipelineDefinition,
    PipelineResult,
    StageStatus,
)
from inferpack.execution.context import PipelineExecutionContext
from inferpack.execution.stage_runner import run_stage

logger = logging.getLogger(__name__)


class PipelineExecutor:
    """Deterministic, multi-stage pipeline executor."""

    def __init__(
        self,
        stage_registry: dict[str, Any],  # stage_type → PipelineStage instance
        config: ExecutionConfig,
    ) -> None:
        """Initialise the executor with a stage registry and execution config."""
        self._stages = stage_registry
        self._config = config
        self._semaphore = asyncio.Semaphore(config.max_concurrent_executions)

    async def execute(
        self,
        pipeline: PipelineDefinition,
        inputs: dict[str, Any],
    ) -> PipelineResult:
        """Execute *pipeline* with the given *inputs*.

        Returns a ``PipelineResult`` containing per-stage results and
        the final output data.
        """
        async with self._semaphore:
            return await self._run(pipeline, inputs)

    async def _run(
        self,
        pipeline: PipelineDefinition,
        inputs: dict[str, Any],
    ) -> PipelineResult:
        """Run all stages of *pipeline* sequentially and return the aggregated result."""
        ctx = PipelineExecutionContext(pipeline_name=pipeline.name, data=dict(inputs))
        t0 = time.time()

        logger.info(
            "Executing pipeline %s (id=%s, stages=%d)",
            pipeline.name,
            ctx.execution_id,
            len(pipeline.stages),
        )

        for stage_def in pipeline.stages:
            # Resolve the stage implementation
            stage_impl = self._stages.get(stage_def.stage_type)
            if stage_impl is None:
                error = f"Unknown stage type: {stage_def.stage_type}"
                logger.error(error)
                return PipelineResult(
                    pipeline_name=pipeline.name,
                    success=False,
                    stage_results=ctx.stage_results,
                    total_duration_seconds=time.time() - t0,
                    error=error,
                )

            # Apply default timeout if not set
            if stage_def.timeout_seconds <= 0:
                stage_def.timeout_seconds = self._config.default_stage_timeout_seconds
            if stage_def.max_retries < 0:
                stage_def.max_retries = self._config.default_max_retries

            result = await run_stage(stage_impl, stage_def, ctx.data)
            ctx.record_stage(result)

            if result.status not in (StageStatus.COMPLETED,):
                logger.warning(
                    "Pipeline %s aborted at stage %s: %s",
                    pipeline.name,
                    stage_def.name,
                    result.error,
                )
                return PipelineResult(
                    pipeline_name=pipeline.name,
                    success=False,
                    stage_results=ctx.stage_results,
                    total_duration_seconds=time.time() - t0,
                    error=f"Stage {stage_def.name} failed: {result.error}",
                )

        total = time.time() - t0
        logger.info(
            "Pipeline %s completed in %.3fs (id=%s)",
            pipeline.name,
            total,
            ctx.execution_id,
        )

        return PipelineResult(
            pipeline_name=pipeline.name,
            success=True,
            stage_results=ctx.stage_results,
            total_duration_seconds=total,
            output=ctx.data,
        )
