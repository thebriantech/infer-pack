"""Custom exceptions for InferPack.

Hierarchy::

    InferPackError
    ├── ModelError
    │   ├── ModelNotFoundError
    │   ├── ModelStateError
    │   └── ModelBusyError
    ├── PipelineError
    │   ├── PipelineNotFoundError
    │   ├── PipelineAlreadyExistsError
    │   └── PipelineActivationError
    ├── StageError
    │   ├── StageValidationError
    │   ├── StageTimeoutError
    │   └── StageNotFoundError
    ├── InferenceError
    │   └── InferenceServerNotReady
    └── ConfigurationError
"""


class InferPackError(Exception):
    """Base exception for all InferPack errors."""


# ---------------------------------------------------------------------------
# Model errors
# ---------------------------------------------------------------------------

class ModelError(InferPackError):
    """Error related to model operations."""


class ModelNotFoundError(ModelError):
    """Model not found in the model registry."""


class ModelStateError(ModelError):
    """Invalid model state transition attempted."""


class ModelBusyError(ModelError):
    """Model has active inference requests and cannot be unloaded."""


# ---------------------------------------------------------------------------
# Pipeline errors
# ---------------------------------------------------------------------------

class PipelineError(InferPackError):
    """Error related to pipeline operations."""


class PipelineNotFoundError(PipelineError):
    """Pipeline not found in the pipeline registry."""


class PipelineAlreadyExistsError(PipelineError):
    """A pipeline with this name is already registered."""


class PipelineActivationError(PipelineError):
    """Error while activating or deactivating a pipeline."""


class PipelineNotActiveError(PipelineError):
    """Pipeline is not in an active state for execution."""


# ---------------------------------------------------------------------------
# Stage errors
# ---------------------------------------------------------------------------

class StageError(InferPackError):
    """Error during pipeline stage execution."""


class StageValidationError(StageError):
    """Required stage inputs are missing or invalid."""


class StageTimeoutError(StageError):
    """Stage execution exceeded its timeout."""


class StageNotFoundError(StageError):
    """Stage type not registered in the stage registry."""


# ---------------------------------------------------------------------------
# Inference errors
# ---------------------------------------------------------------------------

class InferenceError(InferPackError):
    """Error communicating with the inference backend."""


class InferenceServerNotReady(InferenceError):
    """The inference server is not reachable or not ready."""


# ---------------------------------------------------------------------------
# Configuration errors
# ---------------------------------------------------------------------------

class ConfigurationError(InferPackError):
    """Invalid or missing configuration."""
