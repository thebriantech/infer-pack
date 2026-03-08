"""Face-domain pipeline stages.

Contains stages specific to face detection and recognition models:
    - ``scrfd_detection``   — SCRFD multi-scale face detector
    - ``arcface_extraction`` — ArcFace embedding extractor

Other domains (object, pose, OCR …) follow the same pattern: create a
sub-package under ``stages/`` with a ``get_stages()`` entry point.
"""

from __future__ import annotations

from typing import Any

from inferpack.core.interfaces import InferenceClient, PipelineStage
from inferpack.control.model_registry import ModelRegistry

from inferpack.stages.face.scrfd_detection import SCRFDDetectionStage
from inferpack.stages.face.arcface_extraction import ArcFaceExtractionStage


def get_stages(
    inference_client: InferenceClient,
    model_registry: ModelRegistry,
) -> dict[str, PipelineStage]:
    """Return face-domain stage instances keyed by stage-type name.

    These stages require an inference client and model registry because
    they call Triton for model inference.
    """
    return {
        # Canonical names
        "scrfd_detection": SCRFDDetectionStage(inference_client, model_registry),
        "arcface_extraction": ArcFaceExtractionStage(inference_client, model_registry),
        # Legacy aliases (backward compat with existing pipeline YAMLs)
        "face_detection": SCRFDDetectionStage(inference_client, model_registry),
        "face_feature_extraction": ArcFaceExtractionStage(
            inference_client, model_registry
        ),
    }
