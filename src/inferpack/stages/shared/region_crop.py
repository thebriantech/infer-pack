"""Region crop stage — crop detected regions and prepare for recognition.

A **shared** stage: not tied to any specific model domain.  Works with
any detection format providing ``x1, y1, x2, y2`` bounding boxes.

The file is named ``region_crop`` to emphasise reusability.  For backward
compatibility the stage type ``face_crop`` is registered as an alias.

Stage type: ``region_crop`` (alias: ``face_crop``)

Inputs (context):
    - raw image bytes
    - detection dicts with ``x1, y1, x2, y2``

Outputs (context):
    - list of cropped numpy arrays (NCHW, normalised)
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image

from inferpack.core.errors import StageValidationError
from inferpack.core.interfaces import PipelineStage
from inferpack.stages._utils.image_ops import normalize_array, rgb_to_bgr, hwc_to_nchw


class RegionCropStage(PipelineStage):
    """Crop bounding-box regions from the source image."""

    async def execute(
        self,
        context: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Crop, resize, and normalise each detected bounding-box region."""
        target_size = config.get("crop_size", [112, 112])
        margin = config.get("margin", 0.1)
        norm_mode = config.get("norm_mode", "arcface")

        image_keys = config.get("image_keys", ["image_bytes"])
        detection_keys = config.get("detection_keys", ["detections"])
        output_keys = config.get("output_keys", ["cropped_faces"])

        result: dict[str, Any] = {}
        for img_key, det_key, out_key in zip(image_keys, detection_keys, output_keys):
            raw = context[img_key]
            detections = context[det_key]
            crops = self._crop_regions(raw, detections, target_size, margin, norm_mode)
            result[out_key] = crops

        return result

    def validate_inputs(
        self,
        context: dict[str, Any],
        config: dict[str, Any],
    ) -> None:
        """Raise ``StageValidationError`` if any required image or detection key is absent."""
        image_keys = config.get("image_keys", ["image_bytes"])
        detection_keys = config.get("detection_keys", ["detections"])
        for key in [*image_keys, *detection_keys]:
            if key not in context:
                raise StageValidationError(f"Missing required input: {key}")

    @staticmethod
    def _crop_regions(
        raw: bytes,
        detections: list[dict[str, Any]],
        target_size: list[int],
        margin: float,
        norm_mode: str,
    ) -> list[np.ndarray]:
        """Crop, resize, and normalise each detected region."""
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        w, h = img.size
        crops: list[np.ndarray] = []

        for det in detections:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            rw, rh = x2 - x1, y2 - y1

            # Apply margin
            x1 = max(0, x1 - rw * margin)
            y1 = max(0, y1 - rh * margin)
            x2 = min(w, x2 + rw * margin)
            y2 = min(h, y2 + rh * margin)

            region = img.crop((int(x1), int(y1), int(x2), int(y2)))
            region = region.resize(
                (target_size[0], target_size[1]), Image.BILINEAR
            )

            arr = np.array(region, dtype=np.float32)
            arr = rgb_to_bgr(arr)
            arr = normalize_array(arr, norm_mode)
            arr = hwc_to_nchw(arr)
            crops.append(arr)

        return crops
