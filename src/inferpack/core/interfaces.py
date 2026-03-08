"""Abstract interfaces for InferPack components.

These interfaces define the contracts between the three planes
(Control, Execution, Inference) and must NOT be merged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class PipelineStage(ABC):
    """Abstract interface for a pipeline stage implementation.

    Every concrete stage (preprocessing, inference, postprocessing, etc.)
    must implement this interface so the execution plane can invoke it
    uniformly.
    """

    @abstractmethod
    async def execute(
        self,
        context: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute this stage.

        Args:
            context: Mutable execution context carrying data between stages.
            config: Stage-specific configuration from the pipeline definition.

        Returns:
            Updated context dictionary.
        """
        ...

    @abstractmethod
    def validate_inputs(
        self,
        context: dict[str, Any],
        config: dict[str, Any],
    ) -> None:
        """Validate that required inputs exist in the context.

        Raises:
            StageValidationError: If required inputs are missing or invalid.
        """
        ...


class InferenceClient(ABC):
    """Abstract interface for communication with the inference backend.

    In Phase 1 the backend is NVIDIA Triton Inference Server, but it is
    treated as a black box behind this interface.
    """

    @abstractmethod
    async def load_model(self, model_name: str) -> None:
        """Request model loading on the inference server."""
        ...

    @abstractmethod
    async def unload_model(self, model_name: str) -> None:
        """Request model unloading on the inference server."""
        ...

    @abstractmethod
    async def is_model_ready(self, model_name: str) -> bool:
        """Check if a model is loaded and ready for inference."""
        ...

    @abstractmethod
    async def infer(
        self,
        model_name: str,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Run inference on a loaded model.

        Args:
            model_name: Name of the model in the repository.
            inputs: Mapping of input tensor name → numpy array.

        Returns:
            Mapping of output tensor name → numpy array.
        """
        ...

    @abstractmethod
    async def get_model_metadata(self, model_name: str) -> dict[str, Any]:
        """Retrieve model metadata (inputs, outputs, etc.)."""
        ...

    @abstractmethod
    async def is_server_ready(self) -> bool:
        """Check whether the inference server is reachable and ready."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release underlying connections / resources."""
        ...
