"""Triton Inference Server HTTP client.

This is the Inference Plane implementation.  Triton is treated as an
opaque inference backend — no business logic lives here.

Communication uses plain HTTP via ``httpx`` so there is no dependency
on the Triton-specific Python client library.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
import numpy as np

from inferpack.config import TritonConfig
from inferpack.core.errors import InferenceError, InferenceServerNotReady
from inferpack.core.interfaces import InferenceClient
from inferpack.inference.dtype_utils import numpy_to_triton_dtype

logger = logging.getLogger(__name__)


class TritonHTTPClient(InferenceClient):
    """Async HTTP client that talks to Triton's v2 REST API."""

    def __init__(self, config: TritonConfig) -> None:
        self._base_url = config.http_url
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

    async def close(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------
    # InferenceClient interface
    # ------------------------------------------------------------------

    async def is_server_ready(self) -> bool:
        try:
            resp = await self._client.get("/v2/health/ready")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def load_model(self, model_name: str) -> None:
        url = f"/v2/repository/models/{model_name}/load"
        try:
            resp = await self._client.post(url)
            if resp.status_code not in (200, 409):
                raise InferenceError(
                    f"Failed to load model {model_name}: "
                    f"{resp.status_code} {resp.text}"
                )
            logger.info("Triton load requested for %s", model_name)
        except httpx.HTTPError as exc:
            raise InferenceError(
                f"HTTP error loading model {model_name}: {exc}"
            ) from exc

    async def unload_model(self, model_name: str) -> None:
        url = f"/v2/repository/models/{model_name}/unload"
        try:
            resp = await self._client.post(url)
            if resp.status_code not in (200, 404):
                raise InferenceError(
                    f"Failed to unload model {model_name}: "
                    f"{resp.status_code} {resp.text}"
                )
            logger.info("Triton unload requested for %s", model_name)
        except httpx.HTTPError as exc:
            raise InferenceError(
                f"HTTP error unloading model {model_name}: {exc}"
            ) from exc

    async def is_model_ready(self, model_name: str) -> bool:
        url = f"/v2/models/{model_name}/ready"
        try:
            resp = await self._client.get(url)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def get_model_metadata(self, model_name: str) -> dict[str, Any]:
        url = f"/v2/models/{model_name}"
        try:
            resp = await self._client.get(url)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            raise InferenceError(
                f"Error fetching metadata for {model_name}: {exc}"
            ) from exc

    async def infer(
        self,
        model_name: str,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Run inference via Triton's v2 HTTP inference API.

        ``inputs`` maps input tensor names to numpy arrays.
        Returns a dict mapping output tensor names to numpy arrays.
        """
        url = f"/v2/models/{model_name}/infer"

        # Build the v2 inference request body
        request_inputs = []
        for name, array in inputs.items():
            arr = np.asarray(array)
            request_inputs.append(
                {
                    "name": name,
                    "shape": list(arr.shape),
                    "datatype": numpy_to_triton_dtype(arr.dtype),
                    "data": arr.flatten().tolist(),
                }
            )

        body: dict[str, Any] = {"inputs": request_inputs}

        try:
            resp = await self._client.post(url, json=body)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise InferenceError(
                f"Inference failed for {model_name}: {exc}"
            ) from exc

        result = resp.json()
        outputs: dict[str, Any] = {}
        for out in result.get("outputs", []):
            shape = out.get("shape", [])
            data = out.get("data", [])
            outputs[out["name"]] = np.array(data).reshape(shape)

        return outputs


