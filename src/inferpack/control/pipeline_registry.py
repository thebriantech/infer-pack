"""Pipeline Registry — discovers, stores, and manages pipeline definitions.

This is a Control Plane component.  It:
    • Loads built-in pipeline YAML files from ``builtin_pipelines_dir``
    • Loads user-defined pipeline YAML files from ``custom_pipelines_dir``
    • Allows runtime registration of additional pipelines
    • Tracks each pipeline's activation state
    • Delegates model reference-counting to the ModelRegistry

The registry does NOT execute pipelines (that is the Execution Plane).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from inferpack.core.errors import (
    PipelineActivationError,
    PipelineAlreadyExistsError,
    PipelineNotFoundError,
)
from inferpack.core.types import PipelineDefinition, PipelineState, StageDefinition
from inferpack.control.model_lifecycle import ModelLifecycleManager
from inferpack.control.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class _PipelineEntry:
    """Internal bookkeeping for a registered pipeline."""

    def __init__(self, definition: PipelineDefinition) -> None:
        self.definition = definition
        self.state: PipelineState = PipelineState.INACTIVE


class PipelineRegistry:
    """Stores pipeline definitions and manages their activation lifecycle."""

    def __init__(
        self,
        model_registry: ModelRegistry,
        lifecycle_manager: ModelLifecycleManager,
    ) -> None:
        self._entries: dict[str, _PipelineEntry] = {}
        self._model_registry = model_registry
        self._lifecycle = lifecycle_manager

    # ------------------------------------------------------------------
    # Discovery — load YAML files from directories
    # ------------------------------------------------------------------

    def discover(
        self,
        builtin_dir: str,
        custom_dir: str,
    ) -> None:
        """Scan directories and register every ``*.yaml`` / ``*.yml`` pipeline."""
        for directory, builtin in [(builtin_dir, True), (custom_dir, False)]:
            path = Path(directory)
            if not path.is_dir():
                logger.warning("Pipeline directory not found: %s", directory)
                continue

            for fp in sorted(path.glob("*.y*ml")):
                try:
                    definition = self._load_yaml(fp, builtin=builtin)
                    self.register(definition)
                except Exception:
                    logger.exception("Failed to load pipeline from %s", fp)

    @staticmethod
    def _load_yaml(path: Path, *, builtin: bool) -> PipelineDefinition:
        """Parse a pipeline YAML file into a ``PipelineDefinition``."""
        with open(path) as fh:
            data: dict[str, Any] = yaml.safe_load(fh) or {}

        stages = [
            StageDefinition(
                name=s["name"],
                stage_type=s["type"],
                config=s.get("config", {}),
                timeout_seconds=s.get("timeout_seconds", 30.0),
                max_retries=s.get("max_retries", 0),
            )
            for s in data.get("stages", [])
        ]

        return PipelineDefinition(
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", "0.1.0"),
            models=data.get("models", []),
            stages=stages,
            builtin=builtin,
        )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, definition: PipelineDefinition) -> None:
        """Register a pipeline definition (must be unique by name)."""
        if definition.name in self._entries:
            raise PipelineAlreadyExistsError(
                f"Pipeline already registered: {definition.name}"
            )
        self._entries[definition.name] = _PipelineEntry(definition)

        # Ensure every model referenced by the pipeline exists in the model registry
        for model in definition.models:
            self._model_registry.register(model)

        logger.info(
            "Registered pipeline: %s (builtin=%s, models=%s)",
            definition.name,
            definition.builtin,
            definition.models,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, name: str) -> PipelineDefinition:
        entry = self._entries.get(name)
        if entry is None:
            raise PipelineNotFoundError(f"Pipeline not found: {name}")
        return entry.definition

    def get_state(self, name: str) -> PipelineState:
        entry = self._entries.get(name)
        if entry is None:
            raise PipelineNotFoundError(f"Pipeline not found: {name}")
        return entry.state

    def list_all(self) -> list[dict[str, Any]]:
        """Return a summary of every registered pipeline."""
        return [
            {
                "name": e.definition.name,
                "description": e.definition.description,
                "version": e.definition.version,
                "builtin": e.definition.builtin,
                "state": e.state.value,
                "models": e.definition.models,
                "stages": [s.name for s in e.definition.stages],
            }
            for e in self._entries.values()
        ]

    # ------------------------------------------------------------------
    # Activation / Deactivation
    # ------------------------------------------------------------------

    async def activate(self, name: str) -> None:
        """Activate a pipeline — loads all required models."""
        entry = self._entries.get(name)
        if entry is None:
            raise PipelineNotFoundError(f"Pipeline not found: {name}")
        if entry.state == PipelineState.ACTIVE:
            return
        if entry.state not in (PipelineState.INACTIVE, PipelineState.ERROR):
            raise PipelineActivationError(
                f"Cannot activate pipeline in state {entry.state.value}"
            )

        entry.state = PipelineState.ACTIVATING
        try:
            for model_name in entry.definition.models:
                self._model_registry.increment_ref(model_name)
                await self._lifecycle.ensure_model_ready(model_name)
            entry.state = PipelineState.ACTIVE
            logger.info("Pipeline %s activated", name)
        except Exception as exc:
            entry.state = PipelineState.ERROR
            # Roll back ref counts
            for model_name in entry.definition.models:
                self._model_registry.decrement_ref(model_name)
            raise PipelineActivationError(
                f"Failed to activate pipeline {name}: {exc}"
            ) from exc

    async def deactivate(self, name: str) -> None:
        """Deactivate a pipeline — decrements model reference counts."""
        entry = self._entries.get(name)
        if entry is None:
            raise PipelineNotFoundError(f"Pipeline not found: {name}")
        if entry.state == PipelineState.INACTIVE:
            return
        if entry.state not in (PipelineState.ACTIVE, PipelineState.ERROR):
            raise PipelineActivationError(
                f"Cannot deactivate pipeline in state {entry.state.value}"
            )

        entry.state = PipelineState.DEACTIVATING
        for model_name in entry.definition.models:
            self._model_registry.decrement_ref(model_name)
        entry.state = PipelineState.INACTIVE
        logger.info("Pipeline %s deactivated", name)
