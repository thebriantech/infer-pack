"""Shared numpy ↔ Triton dtype mappings used by all inference clients.

Centralised here so both HTTP and gRPC clients stay in sync without
duplicating the mapping table.
"""

from __future__ import annotations

import numpy as np

from inferpack.core.errors import InferenceError

# fmt: off
_NP_TO_TRITON: dict[str, str] = {
    "float32": "FP32",
    "float64": "FP64",
    "float16": "FP16",
    "int8":    "INT8",
    "int16":   "INT16",
    "int32":   "INT32",
    "int64":   "INT64",
    "uint8":   "UINT8",
    "uint16":  "UINT16",
    "uint32":  "UINT32",
    "uint64":  "UINT64",
    "bool":    "BOOL",
}
# fmt: on

_TRITON_TO_NP: dict[str, str] = {v: k for k, v in _NP_TO_TRITON.items()}


def numpy_to_triton_dtype(dtype: "np.dtype[Any]") -> str:  # type: ignore[type-arg]
    """Return the Triton datatype string for a numpy dtype.

    Raises:
        InferenceError: If the dtype has no Triton equivalent.
    """
    name = np.dtype(dtype).name
    triton = _NP_TO_TRITON.get(name)
    if triton is None:
        raise InferenceError(f"Unsupported numpy dtype for Triton: {name!r}")
    return triton


def triton_to_numpy_dtype(triton_dtype: str) -> "np.dtype[Any]":  # type: ignore[type-arg]
    """Return the numpy dtype for a Triton datatype string.

    Raises:
        InferenceError: If the Triton dtype has no numpy equivalent.
    """
    name = _TRITON_TO_NP.get(triton_dtype)
    if name is None:
        raise InferenceError(f"Unsupported Triton dtype: {triton_dtype!r}")
    return np.dtype(name)
