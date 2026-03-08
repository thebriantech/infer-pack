"""Model Lifecycle Manager — orchestrates load / warm / idle / unload.

This is a Control Plane component.  It uses the ModelRegistry for state
tracking and delegates actual load/unload calls to the InferenceClient
(Inference Plane).

The lifecycle manager runs a periodic background task that:
    1. Checks for IDLE models whose idle timeout has expired.
    2. Unloads them (if no active inferences remain).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from inferpack.config import ModelLifecycleConfig
from inferpack.core.errors import (
    ModelBusyError,
    ModelNotFoundError,
    ModelStateError,
)
from inferpack.core.interfaces import InferenceClient
from inferpack.core.types import ModelState
from inferpack.control.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class ModelLifecycleManager:
    """Coordinates model loading, warming, idling, and unloading."""

    def __init__(
        self,
        registry: ModelRegistry,
        inference_client: InferenceClient,
        config: ModelLifecycleConfig,
    ) -> None:
        self._registry = registry
        self._client = inference_client
        self._config = config

        # Serialise load operations to respect max_concurrent_loads
        self._load_semaphore = asyncio.Semaphore(config.max_concurrent_loads)
        # Prevent duplicate in-flight loads for the same model
        self._loading_locks: dict[str, asyncio.Lock] = {}

        self._background_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Lifecycle control
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background lifecycle checker."""
        if self._background_task is None:
            self._background_task = asyncio.create_task(self._lifecycle_loop())
            logger.info("Model lifecycle manager started")

    async def stop(self) -> None:
        """Cancel the background lifecycle checker and wait for it."""
        if self._background_task is not None:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
            self._background_task = None
            logger.info("Model lifecycle manager stopped")

    # ------------------------------------------------------------------
    # Load / Unload API (called by Control Plane pipeline activation)
    # ------------------------------------------------------------------

    async def ensure_model_ready(self, model_name: str) -> None:
        """Guarantee that *model_name* is in READY state.

        If the model is UNLOADED or IDLE it will be (re-)loaded.
        If it is already LOADING/WARMING this call will wait.
        If it is already READY this is a no-op.
        """
        if model_name not in self._loading_locks:
            self._loading_locks[model_name] = asyncio.Lock()

        async with self._loading_locks[model_name]:
            info = self._registry.get(model_name)

            if info.state == ModelState.READY:
                return
            if info.state == ModelState.IDLE:
                # Transition back to READY — model is still loaded in Triton
                self._registry.transition(model_name, ModelState.READY)
                return
            if info.state in (ModelState.LOADING, ModelState.WARMING):
                # Another coroutine is already loading — wait
                pass  # the lock serialises; by the time we get here it's done
            if info.state == ModelState.UNLOADING:
                # Wait for unload to finish, then re-load
                await self._wait_for_state(model_name, ModelState.UNLOADED)

            info = self._registry.get(model_name)
            if info.state == ModelState.READY:
                return

            await self._load_model(model_name)

    async def request_unload(self, model_name: str) -> None:
        """Request that a model be unloaded (respects active inferences)."""
        info = self._registry.get(model_name)

        if info.state == ModelState.UNLOADED:
            return
        if info.active_inferences > 0:
            raise ModelBusyError(
                f"Model {model_name} has {info.active_inferences} active requests"
            )

        await self._unload_model(model_name)

    # ------------------------------------------------------------------
    # Internal: load pipeline
    # ------------------------------------------------------------------

    async def _load_model(self, model_name: str) -> None:
        """Perform the full UNLOADED → LOADING → WARMING → READY cycle."""
        async with self._load_semaphore:
            try:
                self._registry.transition(model_name, ModelState.LOADING)
                await self._client.load_model(model_name)

                self._registry.transition(model_name, ModelState.WARMING)
                await self._warmup(model_name)

                self._registry.transition(model_name, ModelState.READY)
                info = self._registry.get(model_name)
                info.touch()
                logger.info("Model %s is READY", model_name)
            except Exception:
                logger.exception("Failed to load model %s", model_name)
                # Roll back to UNLOADED on any error
                try:
                    self._registry.transition(model_name, ModelState.UNLOADED)
                except ModelStateError:
                    pass
                raise

    async def _warmup(self, model_name: str) -> None:
        """Send warm-up inference requests to prime the model."""
        for i in range(self._config.warmup_requests):
            try:
                ready = await self._client.is_model_ready(model_name)
                if not ready:
                    await asyncio.sleep(0.5)
            except Exception:
                logger.warning("Warm-up check %d failed for %s", i, model_name)

    async def _unload_model(self, model_name: str) -> None:
        """Perform the → UNLOADING → UNLOADED cycle."""
        try:
            self._registry.transition(model_name, ModelState.UNLOADING)
            await self._client.unload_model(model_name)
            self._registry.transition(model_name, ModelState.UNLOADED)
            logger.info("Model %s unloaded", model_name)
        except Exception:
            logger.exception("Failed to unload model %s", model_name)
            try:
                self._registry.transition(model_name, ModelState.UNLOADED)
            except ModelStateError:
                pass

    # ------------------------------------------------------------------
    # Background lifecycle loop
    # ------------------------------------------------------------------

    async def _lifecycle_loop(self) -> None:
        """Periodically check for idle models and unload them."""
        interval = self._config.lifecycle_check_interval_seconds
        timeout = self._config.idle_timeout_seconds

        while True:
            try:
                await asyncio.sleep(interval)
                now = time.time()

                for info in self._registry.list_by_state(ModelState.IDLE):
                    idle_duration = now - info.last_used
                    if idle_duration >= timeout and info.active_inferences == 0:
                        logger.info(
                            "Model %s idle for %.0fs — unloading",
                            info.name,
                            idle_duration,
                        )
                        try:
                            await self._unload_model(info.name)
                        except Exception:
                            logger.exception(
                                "Background unload of %s failed", info.name
                            )

                # Transition READY models with ref_count 0 to IDLE
                for info in self._registry.list_by_state(ModelState.READY):
                    if info.ref_count == 0:
                        try:
                            self._registry.transition(info.name, ModelState.IDLE)
                        except ModelStateError:
                            pass

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Error in lifecycle loop")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _wait_for_state(
        self,
        model_name: str,
        target: ModelState,
        timeout: float = 60.0,
    ) -> None:
        """Poll until model reaches *target* state or *timeout* expires."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            info = self._registry.get(model_name)
            if info.state == target:
                return
            await asyncio.sleep(0.5)
        raise ModelStateError(
            f"Timed out waiting for {model_name} to reach {target.value}"
        )
