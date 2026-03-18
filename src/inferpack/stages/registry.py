"""Stage registry — auto-discovers stages from sub-packages.

The registry collects stages from two sources:

1. **Shared stages** (``stages.shared``) — domain-agnostic, no deps.
2. **Domain packages** (``stages.face``, ``stages.object``, …) — each
   exposes a ``get_stages(client, registry)`` entry point.

Usage::

    registry = build_stage_registry(inference_client, model_registry)
    stage_impl = registry["face_detection"]

To add a new domain without modifying this file, use either:

* **Environment variable** — set ``INFERPACK_STAGE_PACKAGES`` to a
  comma-separated list of extra package names::

      INFERPACK_STAGE_PACKAGES=mycompany.stages.ocr,mycompany.stages.pose

* **Programmatic** — call :func:`register_stage_package` before the
  registry is built::

      from inferpack.stages.registry import register_stage_package
      register_stage_package("mycompany.stages.ocr")

Both mechanisms append to the built-in list; they do not replace it.
"""

from __future__ import annotations

import importlib
import logging
import os
from typing import Any

from inferpack.core.interfaces import InferenceClient, PipelineStage
from inferpack.control.model_registry import ModelRegistry
from inferpack.stages.shared import get_stages as get_shared_stages

logger = logging.getLogger(__name__)

# Built-in domain sub-packages.
# Each must expose ``get_stages(inference_client, model_registry) -> dict``.
_BUILTIN_DOMAIN_PACKAGES: list[str] = [
    "inferpack.stages.face",
]

# Extra packages registered at runtime (before build_stage_registry is called).
_EXTRA_DOMAIN_PACKAGES: list[str] = []


def register_stage_package(package_name: str) -> None:
    """Register an additional domain package to be loaded at startup.

    Must be called before :func:`build_stage_registry`.

    Args:
        package_name: Fully-qualified Python package name, e.g.
                      ``"mycompany.stages.ocr"``.
    """
    if package_name not in _EXTRA_DOMAIN_PACKAGES:
        _EXTRA_DOMAIN_PACKAGES.append(package_name)


def _get_domain_packages() -> list[str]:
    """Return the full list of domain packages to load.

    Order: built-ins → env-var additions → programmatic additions.
    """
    packages = list(_BUILTIN_DOMAIN_PACKAGES)

    env_value = os.environ.get("INFERPACK_STAGE_PACKAGES", "").strip()
    if env_value:
        for pkg in env_value.split(","):
            pkg = pkg.strip()
            if pkg and pkg not in packages:
                packages.append(pkg)

    for pkg in _EXTRA_DOMAIN_PACKAGES:
        if pkg not in packages:
            packages.append(pkg)

    return packages


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
    for pkg_name in _get_domain_packages():
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
