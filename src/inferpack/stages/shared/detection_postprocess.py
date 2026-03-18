"""Detection post-processing stage — sort, filter, limit.

A **shared** stage: works with any detection output containing a
``confidence`` field.

Stage type: ``detection_postprocess``

Inputs (context):
    - ``detections`` (list of detection dicts)

Outputs (context):
    - ``detections`` (filtered / sorted detections)
    - ``num_faces`` (int)  — kept for backward compat; semantically = num_detections
"""

from __future__ import annotations

from typing import Any

from inferpack.core.errors import StageValidationError
from inferpack.core.interfaces import PipelineStage


class DetectionPostprocessStage(PipelineStage):
    """Post-process raw detections: sort by confidence, limit count."""

    async def execute(
        self,
        context: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Sort detections by confidence, apply max-count limit, and return results."""
        detections: list[dict[str, Any]] = context["detections"]
        max_detections = config.get("max_detections", config.get("max_faces", 100))

        detections = sorted(
            detections, key=lambda d: d["confidence"], reverse=True
        )
        detections = detections[:max_detections]

        return {
            "detections": detections,
            "num_detections": len(detections),
            # Legacy alias
            "num_faces": len(detections),
        }

    def validate_inputs(
        self,
        context: dict[str, Any],
        config: dict[str, Any],
    ) -> None:
        """Raise ``StageValidationError`` if ``detections`` is absent from the context."""
        if "detections" not in context:
            raise StageValidationError("Missing required input: detections")
