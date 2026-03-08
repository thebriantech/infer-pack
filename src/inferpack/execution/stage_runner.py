"""Stage Runner — executes a single stage with timeout and retry logic.

The stage runner is an Execution Plane component.  It:
    • Validates stage inputs via ``PipelineStage.validate_inputs``
    • Enforces per-stage timeouts
    • Retries on transient failures
    • Produces a ``StageResult`` for observability
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from inferpack.core.errors import StageTimeoutError, StageValidationError
from inferpack.core.interfaces import PipelineStage
from inferpack.core.types import StageDefinition, StageResult, StageStatus

logger = logging.getLogger(__name__)


async def run_stage(
    stage_impl: PipelineStage,
    stage_def: StageDefinition,
    context: dict[str, Any],
) -> StageResult:
    """Execute a single pipeline stage with validation, timeout, and retries.

    Args:
        stage_impl: The concrete stage implementation.
        stage_def: Declarative stage definition (timeout, retries, config).
        context: Shared execution data dict — mutated in place.

    Returns:
        A ``StageResult`` describing the outcome.
    """
    attempts = stage_def.max_retries + 1
    last_error: str | None = None

    for attempt in range(1, attempts + 1):
        t0 = time.time()
        try:
            # 1. Validate inputs
            stage_impl.validate_inputs(context, stage_def.config)

            # 2. Execute with timeout
            updated = await asyncio.wait_for(
                stage_impl.execute(context, stage_def.config),
                timeout=stage_def.timeout_seconds,
            )

            # 3. Merge updated keys back into context
            context.update(updated)

            duration = time.time() - t0
            logger.info(
                "Stage %s completed in %.3fs (attempt %d/%d)",
                stage_def.name,
                duration,
                attempt,
                attempts,
            )
            return StageResult(
                stage_name=stage_def.name,
                status=StageStatus.COMPLETED,
                duration_seconds=duration,
                output_keys=list(updated.keys()),
            )

        except asyncio.TimeoutError:
            duration = time.time() - t0
            last_error = f"Timed out after {stage_def.timeout_seconds}s"
            logger.warning(
                "Stage %s timed out (attempt %d/%d)",
                stage_def.name,
                attempt,
                attempts,
            )

        except StageValidationError as exc:
            duration = time.time() - t0
            last_error = str(exc)
            logger.error("Stage %s validation failed: %s", stage_def.name, exc)
            # Validation errors are not retried
            return StageResult(
                stage_name=stage_def.name,
                status=StageStatus.FAILED,
                duration_seconds=duration,
                error=last_error,
            )

        except Exception as exc:
            duration = time.time() - t0
            last_error = str(exc)
            logger.error(
                "Stage %s failed (attempt %d/%d): %s",
                stage_def.name,
                attempt,
                attempts,
                exc,
            )

        # Brief delay before retry
        if attempt < attempts:
            await asyncio.sleep(min(1.0 * attempt, 5.0))

    # All retries exhausted
    return StageResult(
        stage_name=stage_def.name,
        status=StageStatus.TIMED_OUT if "Timed out" in (last_error or "") else StageStatus.FAILED,
        duration_seconds=time.time() - t0,
        error=last_error,
    )
