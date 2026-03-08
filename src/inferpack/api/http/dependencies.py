"""Application-wide dependency container.

This is the single place where Control Plane, Execution Plane, and
Inference Plane instances are created and wired together.  FastAPI
route handlers receive them via ``request.app.state``.
"""

from __future__ import annotations

from dataclasses import dataclass

from inferpack.config import InferPackConfig
from inferpack.control.model_registry import ModelRegistry
from inferpack.control.model_lifecycle import ModelLifecycleManager
from inferpack.control.pipeline_registry import PipelineRegistry
from inferpack.core.interfaces import InferenceClient
from inferpack.execution.executor import PipelineExecutor
from inferpack.inference import create_inference_client
from inferpack.stages.registry import build_stage_registry


@dataclass
class AppState:
    """Holds all runtime components — attached to ``app.state``."""

    config: InferPackConfig
    triton_client: InferenceClient
    model_registry: ModelRegistry
    lifecycle_manager: ModelLifecycleManager
    pipeline_registry: PipelineRegistry
    executor: PipelineExecutor


def create_app_state(config: InferPackConfig) -> AppState:
    """Wire up all components and return an ``AppState``."""

    # Inference Plane — protocol chosen by config.triton.protocol
    triton_client = create_inference_client(config.triton)

    # Control Plane
    model_registry = ModelRegistry()
    lifecycle_manager = ModelLifecycleManager(
        registry=model_registry,
        inference_client=triton_client,
        config=config.model_lifecycle,
    )
    pipeline_registry = PipelineRegistry(
        model_registry=model_registry,
        lifecycle_manager=lifecycle_manager,
    )

    # Execution Plane
    stage_registry = build_stage_registry(triton_client, model_registry)
    executor = PipelineExecutor(
        stage_registry=stage_registry,
        config=config.execution,
    )

    return AppState(
        config=config,
        triton_client=triton_client,
        model_registry=model_registry,
        lifecycle_manager=lifecycle_manager,
        pipeline_registry=pipeline_registry,
        executor=executor,
    )
