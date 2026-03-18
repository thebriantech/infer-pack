"""Inference Plane — Triton Inference Server communication."""

from __future__ import annotations

import importlib

from inferpack.config import TritonConfig
from inferpack.core.interfaces import InferenceClient

# Registry mapping protocol name → fully-qualified class path.
# Use register_inference_protocol() to add custom backends without
# touching this file.
_PROTOCOL_REGISTRY: dict[str, str] = {
    "http":  "inferpack.inference.triton_client.TritonHTTPClient",
    "grpc":  "inferpack.inference.triton_grpc_client.TritonGRPCClient",
}


def register_inference_protocol(protocol: str, class_path: str) -> None:
    """Register a custom inference client for *protocol*.

    Args:
        protocol:   Protocol name (case-insensitive), e.g. ``"rest"``.
        class_path: Fully-qualified class path, e.g. ``"mymodule.MyClient"``.

    Example::

        from inferpack.inference import register_inference_protocol
        register_inference_protocol("rest", "mypkg.clients.RestClient")
    """
    _PROTOCOL_REGISTRY[protocol.lower()] = class_path


def create_inference_client(config: TritonConfig) -> InferenceClient:
    """Factory — build the correct ``InferenceClient`` for *config.protocol*.

    Looks up the protocol in :data:`_PROTOCOL_REGISTRY` (which is
    user-extensible via :func:`register_inference_protocol`) and
    lazy-imports the corresponding class.

    Raises:
        ValueError: If the protocol string is not registered.
    """
    protocol = config.protocol.lower()
    class_path = _PROTOCOL_REGISTRY.get(protocol)
    if class_path is None:
        supported = list(_PROTOCOL_REGISTRY.keys())
        raise ValueError(
            f"Unknown triton.protocol={config.protocol!r}. "
            f"Registered protocols: {supported}. "
            f"Use register_inference_protocol() to add a custom backend."
        )

    module_name, class_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    return cls(config)
