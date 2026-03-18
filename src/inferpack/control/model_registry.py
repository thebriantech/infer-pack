"""Model Registry — tracks known models and their runtime state.

The model registry is a Control Plane component.  It owns the source of
truth for which models exist and what state each one is in.  It does NOT
perform inference and does NOT talk to Triton directly.
"""

from __future__ import annotations

import logging
import threading
from typing import Sequence

from inferpack.core.errors import ModelNotFoundError, ModelStateError
from inferpack.core.types import MODEL_STATE_TRANSITIONS, ModelInfo, ModelState

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Thread-safe registry of known models and their runtime metadata."""

    def __init__(self) -> None:
        self._models: dict[str, ModelInfo] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, model_name: str) -> ModelInfo:
        """Register a model (idempotent). Returns the ``ModelInfo``."""
        with self._lock:
            if model_name not in self._models:
                info = ModelInfo(name=model_name)
                self._models[model_name] = info
                logger.info("Registered model: %s", model_name)
            return self._models[model_name]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, model_name: str) -> ModelInfo:
        """Return ``ModelInfo`` for *model_name* or raise ``ModelNotFoundError``."""
        with self._lock:
            try:
                return self._models[model_name]
            except KeyError:
                raise ModelNotFoundError(f"Model not found: {model_name}")

    def list_all(self) -> list[ModelInfo]:
        """Return a snapshot of all registered models."""
        with self._lock:
            return list(self._models.values())

    def list_by_state(self, state: ModelState) -> list[ModelInfo]:
        """Return models currently in *state*."""
        with self._lock:
            return [m for m in self._models.values() if m.state == state]

    def contains(self, model_name: str) -> bool:
        """Return True if *model_name* is registered."""
        with self._lock:
            return model_name in self._models

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def transition(self, model_name: str, target: ModelState) -> ModelInfo:
        """Attempt to transition *model_name* to *target* state.

        Raises:
            ModelNotFoundError: If the model is not registered.
            ModelStateError: If the transition is invalid.
        """
        with self._lock:
            info = self._models.get(model_name)
            if info is None:
                raise ModelNotFoundError(f"Model not found: {model_name}")

            valid = MODEL_STATE_TRANSITIONS.get(info.state, set())
            if target not in valid:
                raise ModelStateError(
                    f"Invalid transition for {model_name}: "
                    f"{info.state.value} → {target.value}"
                )
            old = info.state
            info.state = target
            logger.info("Model %s: %s → %s", model_name, old.value, target.value)
            return info

    # ------------------------------------------------------------------
    # Reference counting
    # ------------------------------------------------------------------

    def increment_ref(self, model_name: str) -> int:
        """Increment the reference count and return the new value."""
        with self._lock:
            info = self._require(model_name)
            info.ref_count += 1
            logger.debug("Model %s ref_count → %d", model_name, info.ref_count)
            return info.ref_count

    def decrement_ref(self, model_name: str) -> int:
        """Decrement the reference count (floor 0) and return the new value."""
        with self._lock:
            info = self._require(model_name)
            info.ref_count = max(0, info.ref_count - 1)
            logger.debug("Model %s ref_count → %d", model_name, info.ref_count)
            return info.ref_count

    # ------------------------------------------------------------------
    # Inference tracking
    # ------------------------------------------------------------------

    def begin_inference(self, model_name: str) -> None:
        """Mark the start of an inference request on this model."""
        with self._lock:
            info = self._require(model_name)
            info.active_inferences += 1
            info.touch()

    def end_inference(self, model_name: str) -> None:
        """Mark the end of an inference request on this model."""
        with self._lock:
            info = self._require(model_name)
            info.active_inferences = max(0, info.active_inferences - 1)
            info.touch()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _require(self, model_name: str) -> ModelInfo:
        """Return ``ModelInfo`` for *model_name*, raising ``ModelNotFoundError`` if absent."""
        info = self._models.get(model_name)
        if info is None:
            raise ModelNotFoundError(f"Model not found: {model_name}")
        return info
