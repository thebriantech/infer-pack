"""Image preprocessing stage — decode, resize, normalise.

A **shared** stage: works with any detection or recognition model.
The ``norm_mode`` config parameter selects the appropriate normalisation
(``scrfd``, ``arcface``, ``0to1``, or any mode registered in
``_utils.image_ops.NORM_PARAMS``).

Stage type: ``image_preprocess``

Inputs (context):
    - ``image_bytes`` or keyed by ``image_keys`` config

Outputs (context):
    - preprocessed numpy array (float32, NCHW, BGR)
    - ``original_size`` (w, h)
    - ``det_scale`` (float)
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image

from inferpack.core.errors import StageValidationError
from inferpack.core.interfaces import PipelineStage
from inferpack.stages._utils.image_ops import (
    letterbox_resize,
    normalize_array,
    rgb_to_bgr,
    hwc_to_nchw,
)


class ImagePreprocessStage(PipelineStage):
    """Decode, resize (letterbox), and normalise an input image."""

    async def execute(
        self,
        context: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        target_size = config.get("target_size", [640, 640])
        norm_mode = config.get("norm_mode", "scrfd")
        output: dict[str, Any] = {}

        image_keys = config.get("image_keys", ["image_bytes"])
        output_keys = config.get("output_keys", ["preprocessed_image"])

        for img_key, out_key in zip(image_keys, output_keys):
            raw = context[img_key]
            arr, orig_size, det_scale = self._preprocess(raw, target_size, norm_mode)
            output[out_key] = arr

            suffix = out_key.replace("preprocessed_image", "")
            output[f"original_size{suffix}"] = orig_size
            output[f"det_scale{suffix}"] = det_scale

        return output

    def validate_inputs(
        self,
        context: dict[str, Any],
        config: dict[str, Any],
    ) -> None:
        image_keys = config.get("image_keys", ["image_bytes"])
        for key in image_keys:
            if key not in context:
                raise StageValidationError(f"Missing required input: {key}")

    @staticmethod
    def _preprocess(
        raw: bytes,
        target_size: list[int],
        norm_mode: str,
    ) -> tuple[np.ndarray, tuple[int, int], float]:
        """Decode, letterbox-resize, normalise to float32 NCHW BGR."""
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        orig_w, orig_h = img.size

        target_w, target_h = target_size
        canvas, scale = letterbox_resize(img, target_w, target_h)

        arr = np.array(canvas, dtype=np.float32)  # HWC, RGB
        arr = rgb_to_bgr(arr)
        arr = normalize_array(arr, norm_mode)
        arr = hwc_to_nchw(arr)

        return arr, (orig_w, orig_h), scale
