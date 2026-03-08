"""Shared API-layer business logic used by HTTP and gRPC transports."""

from inferpack.api.shared.pipeline_service import PipelineService, make_json_safe

__all__ = ["PipelineService", "make_json_safe"]
