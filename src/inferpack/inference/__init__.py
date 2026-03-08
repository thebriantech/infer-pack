"""Inference Plane — Triton Inference Server communication."""

from __future__ import annotations

from inferpack.config import TritonConfig
from inferpack.core.interfaces import InferenceClient


def create_inference_client(config: TritonConfig) -> InferenceClient:
    """Factory — build the correct ``InferenceClient`` for *config.protocol*.

    Supported protocols:
        ``"http"`` (default) — uses Triton's v2 REST API via ``httpx``.
        ``"grpc"`` — uses Triton's v2 gRPC API via ``tritonclient[grpc]``.

    Raises:
        ValueError: If the protocol string is not recognised.
    """
    protocol = config.protocol.lower()
    if protocol == "http":
        from inferpack.inference.triton_client import TritonHTTPClient

        return TritonHTTPClient(config)

    if protocol == "grpc":
        from inferpack.inference.triton_grpc_client import TritonGRPCClient

        return TritonGRPCClient(config)

    raise ValueError(
        f"Unknown triton.protocol={config.protocol!r}. "
        f"Supported values: 'http', 'grpc'."
    )
