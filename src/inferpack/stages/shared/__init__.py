"""Shared pipeline stages — domain-agnostic, reusable across pipelines.

Every stage here works with any model domain (face, object, pose, …).
Domain-specific stages live in their own sub-package under ``stages/``.
"""

from __future__ import annotations

from typing import Any

from inferpack.core.interfaces import PipelineStage

from inferpack.stages.shared.image_preprocess import ImagePreprocessStage
from inferpack.stages.shared.region_crop import RegionCropStage
from inferpack.stages.shared.detection_postprocess import DetectionPostprocessStage
from inferpack.stages.shared.feature_comparison import FeatureComparisonStage


def get_stages() -> dict[str, PipelineStage]:
    """Return shared stage instances keyed by stage-type name.

    These stages have no external dependencies (no inference client,
    no model registry) — they are pure data-processing stages.
    """
    preprocess = ImagePreprocessStage()
    crop = RegionCropStage()
    postprocess = DetectionPostprocessStage()
    comparison = FeatureComparisonStage()

    return {
        # Canonical names
        "image_preprocess": preprocess,
        "region_crop": crop,
        "detection_postprocess": postprocess,
        "feature_comparison": comparison,
        # Legacy aliases (backward compat with existing pipeline YAMLs)
        "face_crop": crop,
    }
