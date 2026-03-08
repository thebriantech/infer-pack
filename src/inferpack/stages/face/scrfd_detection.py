"""SCRFD face detection inference stage.

Sends a preprocessed image to the SCRFD model via Triton and decodes
the multi-scale anchor outputs into bounding boxes with landmarks.

SCRFD-10G outputs 9 tensors (3 strides x {scores, boxes, landmarks}):
    stride  8  : scores [12800,1]  boxes [12800,4]   landmarks [12800,10]
    stride 16  : scores [ 3200,1]  boxes [ 3200,4]   landmarks [ 3200,10]
    stride 32  : scores [  800,1]  boxes [  800,4]   landmarks [  800,10]

Stage type: ``scrfd_detection`` (alias: ``face_detection``)

Inputs (context):
    - ``preprocessed_image``  (numpy array, [1,3,640,640] FP32)
    - ``det_scale``           (float, from preprocess)

Outputs (context):
    - ``detections``  (list of dicts with x1, y1, x2, y2, confidence, landmarks)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from inferpack.core.errors import StageValidationError
from inferpack.core.interfaces import InferenceClient, PipelineStage
from inferpack.control.model_registry import ModelRegistry
from inferpack.stages._utils.nms import greedy_nms

logger = logging.getLogger(__name__)


# SCRFD anchor configuration for 640x640 input
_SCRFD_STRIDES = [8, 16, 32]
_SCRFD_SCORE_NAMES = ["448", "471", "494"]
_SCRFD_BOX_NAMES = ["451", "474", "497"]
_SCRFD_LANDMARK_NAMES = ["454", "477", "500"]
_FEAT_SIZES = [(80, 80), (40, 40), (20, 20)]  # 640 / stride
_NUM_ANCHORS = 2  # SCRFD uses 2 anchors per location


class SCRFDDetectionStage(PipelineStage):
    """Run SCRFD face detection inference and decode results."""

    def __init__(
        self,
        inference_client: InferenceClient,
        model_registry: ModelRegistry,
    ) -> None:
        self._client = inference_client
        self._model_registry = model_registry
        self._anchor_centers = _build_anchor_centers()

    async def execute(
        self,
        context: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        model_name = config.get("model_name", "face_detection_model")
        input_key = config.get("input_key", "preprocessed_image")
        output_key = config.get("output_key", "detections")

        # If input key is suffixed (e.g. preprocessed_image_1), use matching
        # det_scale key unless explicitly overridden.
        if "det_scale_key" in config:
            det_scale_key = config["det_scale_key"]
        elif input_key.startswith("preprocessed_image_"):
            suffix = input_key.removeprefix("preprocessed_image")
            det_scale_key = f"det_scale{suffix}"
        else:
            det_scale_key = "det_scale"

        image = context[input_key]
        det_scale = context.get(det_scale_key, 1.0)

        self._model_registry.begin_inference(model_name)
        try:
            raw_output = await self._client.infer(
                model_name=model_name,
                inputs={"input.1": image},
            )
        finally:
            self._model_registry.end_inference(model_name)

        detections = self._decode_scrfd(raw_output, det_scale, config)
        logger.debug(
            "SCRFD decoded %d detections (input_key=%s, output_key=%s, det_scale_key=%s)",
            len(detections),
            input_key,
            output_key,
            det_scale_key,
        )
        return {output_key: detections}

    def validate_inputs(
        self,
        context: dict[str, Any],
        config: dict[str, Any],
    ) -> None:
        input_key = config.get("input_key", "preprocessed_image")
        if input_key not in context:
            raise StageValidationError(f"Missing required input: {input_key}")

    # ------------------------------------------------------------------
    # SCRFD decoding
    # ------------------------------------------------------------------

    def _decode_scrfd(
        self,
        raw: dict[str, Any],
        det_scale: float,
        config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Decode SCRFD multi-scale outputs into detections."""
        conf_thresh = config.get("confidence_threshold", 0.5)
        nms_thresh = config.get("nms_threshold", 0.4)

        all_boxes: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        all_landmarks: list[np.ndarray] = []

        for i, stride in enumerate(_SCRFD_STRIDES):
            scores = np.asarray(raw[_SCRFD_SCORE_NAMES[i]]).reshape(-1)
            boxes_raw = np.asarray(raw[_SCRFD_BOX_NAMES[i]]).reshape(-1, 4)
            lmk_raw = np.asarray(raw[_SCRFD_LANDMARK_NAMES[i]]).reshape(-1, 10)
            anchor_centers = self._anchor_centers[i]

            mask = scores >= conf_thresh
            if not np.any(mask):
                continue

            scores_f = scores[mask]
            boxes_f = boxes_raw[mask]
            lmk_f = lmk_raw[mask]
            anchors_f = anchor_centers[mask]

            boxes = _distance2bbox(anchors_f, boxes_f, stride)
            landmarks = _distance2landmarks(anchors_f, lmk_f, stride)

            all_boxes.append(boxes)
            all_scores.append(scores_f)
            all_landmarks.append(landmarks)

        if not all_boxes:
            return []

        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        landmarks = np.concatenate(all_landmarks, axis=0)

        # Use shared NMS utility
        keep = greedy_nms(boxes, scores, nms_thresh)
        boxes = boxes[keep]
        scores = scores[keep]
        landmarks = landmarks[keep]

        detections: list[dict[str, Any]] = []
        for box, score, lmk in zip(boxes, scores, landmarks):
            detections.append(
                {
                    "x1": float(box[0] / det_scale),
                    "y1": float(box[1] / det_scale),
                    "x2": float(box[2] / det_scale),
                    "y2": float(box[3] / det_scale),
                    "confidence": float(score),
                    "landmarks": (lmk / det_scale).reshape(5, 2).tolist(),
                }
            )

        return detections


# ------------------------------------------------------------------
# SCRFD decoding helpers
# ------------------------------------------------------------------

def _build_anchor_centers() -> list[np.ndarray]:
    """Pre-compute anchor centre coordinates for each stride level."""
    centers = []
    for i, stride in enumerate(_SCRFD_STRIDES):
        fh, fw = _FEAT_SIZES[i]
        y, x = np.mgrid[:fh, :fw]
        grid = np.stack([x.ravel(), y.ravel()], axis=1).astype(np.float32)
        grid = np.repeat(grid, _NUM_ANCHORS, axis=0)
        centers.append(grid)
    return centers


def _distance2bbox(
    anchors: np.ndarray,
    distances: np.ndarray,
    stride: int,
) -> np.ndarray:
    """Convert anchor + distances to x1, y1, x2, y2 boxes."""
    cx = anchors[:, 0] * stride
    cy = anchors[:, 1] * stride
    x1 = cx - distances[:, 0] * stride
    y1 = cy - distances[:, 1] * stride
    x2 = cx + distances[:, 2] * stride
    y2 = cy + distances[:, 3] * stride
    return np.stack([x1, y1, x2, y2], axis=1)


def _distance2landmarks(
    anchors: np.ndarray,
    distances: np.ndarray,
    stride: int,
) -> np.ndarray:
    """Convert anchor + distances to landmark coordinates (5 points x 2)."""
    cx = anchors[:, 0] * stride
    cy = anchors[:, 1] * stride
    lmk = distances.copy()
    for j in range(5):
        lmk[:, j * 2] = cx + lmk[:, j * 2] * stride
        lmk[:, j * 2 + 1] = cy + lmk[:, j * 2 + 1] * stride
    return lmk
