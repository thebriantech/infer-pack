"""Triton Inference Server gRPC client.

Alternative Inference Plane implementation that communicates with Triton
via the KServe v2 gRPC inference protocol instead of HTTP/REST.

Advantages over HTTP:
    • Lower latency (persistent HTTP/2 connections, binary framing)
    • More efficient for large tensor payloads (no JSON encoding overhead)
    • Native streaming support (future)
"""

from __future__ import annotations

import logging
from typing import Any

import grpc
import numpy as np

from inferpack.config import TritonConfig
from inferpack.core.errors import InferenceError, InferenceServerNotReady
from inferpack.core.interfaces import InferenceClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KServe v2 gRPC stubs are generated from the standard proto.
# To avoid a hard dependency on tritonclient we build lightweight
# message helpers on top of raw grpc.  The service and message names
# follow the KServe v2 inference protocol:
#   https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/grpc_predict_v2.proto
# ---------------------------------------------------------------------------

# These are the full method strings Triton's gRPC endpoint exposes.
_SVC = "inference.GRPCInferenceService"
_SERVER_READY = f"/{_SVC}/ServerReady"
_MODEL_READY = f"/{_SVC}/ModelReady"
_MODEL_METADATA = f"/{_SVC}/ModelMetadata"
_MODEL_LOAD = f"/{_SVC}/RepositoryModelLoad"
_MODEL_UNLOAD = f"/{_SVC}/RepositoryModelUnload"
_INFER = f"/{_SVC}/ModelInfer"


# ---------------------------------------------------------------------------
# Numpy ↔ Triton dtype mapping
# ---------------------------------------------------------------------------

_NP_TO_TRITON: dict[str, str] = {
    "float32": "FP32",
    "float64": "FP64",
    "float16": "FP16",
    "int8": "INT8",
    "int16": "INT16",
    "int32": "INT32",
    "int64": "INT64",
    "uint8": "UINT8",
    "uint16": "UINT16",
    "uint32": "UINT32",
    "uint64": "UINT64",
    "bool": "BOOL",
}

_TRITON_TO_NP: dict[str, str] = {v: k for k, v in _NP_TO_TRITON.items()}


def _numpy_to_triton_dtype(dtype: np.dtype) -> str:  # type: ignore[type-arg]
    name = dtype.name
    triton = _NP_TO_TRITON.get(name)
    if triton is None:
        raise InferenceError(f"Unsupported numpy dtype for Triton: {name}")
    return triton


def _triton_to_numpy_dtype(triton_dtype: str) -> np.dtype:  # type: ignore[type-arg]
    name = _TRITON_TO_NP.get(triton_dtype)
    if name is None:
        raise InferenceError(f"Unsupported Triton dtype: {triton_dtype}")
    return np.dtype(name)


# ---------------------------------------------------------------------------
# Lightweight JSON-based gRPC message builder
# ---------------------------------------------------------------------------
# Triton's gRPC API uses protobuf messages.  Rather than importing the
# Triton-specific proto stubs (which pull in a large wheel), we use the
# ``grpc`` channel to call methods directly with serialised protobuf
# messages.  We depend on ``tritonclient.grpc`` for the proto stubs
# since the user opted into grpc — it packages the compiled protobufs.
# ---------------------------------------------------------------------------


class TritonGRPCClient(InferenceClient):
    """Async gRPC client that talks to Triton's v2 gRPC API.

    Uses ``tritonclient.grpc.aio`` under the hood for properly typed
    protobuf messages while keeping the same ``InferenceClient`` contract.
    """

    def __init__(self, config: TritonConfig) -> None:
        self._target = config.grpc_url

        try:
            import tritonclient.grpc.aio as triton_grpc
        except ImportError as exc:
            raise ImportError(
                "tritonclient[grpc] is required when triton.protocol='grpc'. "
                "Install it with: pip install tritonclient[grpc]"
            ) from exc

        self._triton_grpc = triton_grpc
        self._client = triton_grpc.InferenceServerClient(url=self._target)

    async def close(self) -> None:
        await self._client.close()

    # ------------------------------------------------------------------
    # InferenceClient interface
    # ------------------------------------------------------------------

    async def is_server_ready(self) -> bool:
        try:
            return await self._client.is_server_ready()
        except Exception:
            return False

    async def load_model(self, model_name: str) -> None:
        try:
            await self._client.load_model(model_name)
            logger.info("Triton gRPC load requested for %s", model_name)
        except Exception as exc:
            raise InferenceError(
                f"gRPC error loading model {model_name}: {exc}"
            ) from exc

    async def unload_model(self, model_name: str) -> None:
        try:
            await self._client.unload_model(model_name)
            logger.info("Triton gRPC unload requested for %s", model_name)
        except Exception as exc:
            raise InferenceError(
                f"gRPC error unloading model {model_name}: {exc}"
            ) from exc

    async def is_model_ready(self, model_name: str) -> bool:
        try:
            return await self._client.is_model_ready(model_name)
        except Exception:
            return False

    async def get_model_metadata(self, model_name: str) -> dict[str, Any]:
        try:
            meta = await self._client.get_model_metadata(model_name)
            # Convert tritonclient metadata object to a plain dict
            result: dict[str, Any] = {
                "name": meta.model_name,
                "versions": list(meta.model_version) if hasattr(meta, "model_version") else [],
                "platform": getattr(meta, "platform", ""),
                "inputs": [],
                "outputs": [],
            }
            for inp in meta.inputs:
                result["inputs"].append({
                    "name": inp.name,
                    "datatype": inp.datatype,
                    "shape": list(inp.shape),
                })
            for out in meta.outputs:
                result["outputs"].append({
                    "name": out.name,
                    "datatype": out.datatype,
                    "shape": list(out.shape),
                })
            return result
        except Exception as exc:
            raise InferenceError(
                f"gRPC error fetching metadata for {model_name}: {exc}"
            ) from exc

    async def infer(
        self,
        model_name: str,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Run inference via Triton's v2 gRPC inference API.

        ``inputs`` maps input tensor names to numpy arrays.
        Returns a dict mapping output tensor names to numpy arrays.
        """
        triton_grpc = self._triton_grpc

        # Build InferInput objects
        infer_inputs = []
        for name, array in inputs.items():
            arr = np.asarray(array)
            triton_dtype = _numpy_to_triton_dtype(arr.dtype)
            inp = triton_grpc.InferInput(name, list(arr.shape), triton_dtype)
            inp.set_data_from_numpy(arr)
            infer_inputs.append(inp)

        try:
            result = await self._client.infer(
                model_name=model_name,
                inputs=infer_inputs,
            )
        except Exception as exc:
            raise InferenceError(
                f"gRPC inference failed for {model_name}: {exc}"
            ) from exc

        # Extract outputs
        outputs: dict[str, Any] = {}
        # The result object exposes get_output() for each output name.
        # We iterate over what the server returned.
        response = result.get_response()
        for out_meta in response.outputs:
            out_name = out_meta.name
            outputs[out_name] = result.as_numpy(out_name)

        return outputs
