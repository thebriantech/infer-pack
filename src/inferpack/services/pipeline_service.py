"""Compatibility shim for moved API shared pipeline service."""

from inferpack.api.shared.pipeline_service import PipelineService, make_json_safe

__all__ = ["PipelineService", "make_json_safe"]
