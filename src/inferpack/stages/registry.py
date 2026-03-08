"""Stage registry — auto-discovers stages from sub-packages.

The registry collects stages from two sources:

1. **Shared stages** (``stages.shared``) — domain-agnostic, no deps.
2. **Domain packages** (``stages.face``, ``stages.object``, …) — each
   exposes a ``get_stages(client, registry)`` entry point.

Usage::

    registry = build_stage_registry(inference_client, model_registry)
    stage_impl = registry["face_detection"]

To add a new domain, create ``stages/<domain>/__init__.py`` with a
``get_stages(inference_client, model_registry)`` function and register
the package in ``_DOMAIN_PACKAGES`` below.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

from inferpack.core.interfaces import InferenceClient, PipelineStage
from inferpack.control.model_registry import ModelRegistry
from inferpack.stages.shared import get_stages as get_shared_stages

logger = logging.getLogger(__name__)

# Domain sub-packages that contribute stages.
# Each must have a ``get_stages(inference_client, model_registry)`` function.
_DOMAIN_PACKAGES: list[str] = [
    "inferpack.stages.face",
    # "inferpack.stages.object",   # future
    # "inferpack.stages.pose",     # future
]


def build_stage_registry(
    inference_client: InferenceClient,
    model_registry: ModelRegistry,
) -> dict[str, PipelineStage]:
    """Construct the stage registry from all sub-packages.

    Returns a flat dict mapping stage-type names to ``PipelineStage``
    instances.  Users can extend this by adding entries after construction.
    """
    registry: dict[str, PipelineStage] = {}

    # 1. Shared (dependency-free) stages
    registry.update(get_shared_stages())
    logger.info("Registered %d shared stages", len(registry))

    # 2. Domain stages (need inference client + model registry)
    for pkg_name in _DOMAIN_PACKAGES:
        try:
            mod = importlib.import_module(pkg_name)
            domain_stages = mod.get_stages(inference_client, model_registry)
            registry.update(domain_stages)
            logger.info(
                "Registered %d stages from %s", len(domain_stages), pkg_name
            )
        except Exception:
            logger.warning("Failed to load domain package %s", pkg_name, exc_info=True)

    return registry
