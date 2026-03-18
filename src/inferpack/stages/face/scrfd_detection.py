"""SCRFD face detection inference stage.

Sends a preprocessed image to the SCRFD model via Triton and decodes
the multi-scale anchor outputs into bounding boxes with landmarks.

SCRFD-10G default outputs — 9 tensors (3 strides × {scores, boxes, landmarks}):
    stride  8  : scores [12800,1]  boxes [12800,4]   landmarks [12800,10]
    stride 16  : scores [ 3200,1]  boxes [ 3200,4]   landmarks [ 3200,10]
    stride 32  : scores [  800,1]  boxes [  800,4]   landmarks [  800,10]

Stage type: ``scrfd_detection`` (alias: ``face_detection``)

Inputs (context):
    - ``preprocessed_image``  (numpy array, [1,3,H,W] FP32)
    - ``det_scale``           (float, from preprocess)

Outputs (context):
    - ``detections``  (list of dicts with x1, y1, x2, y2, confidence, landmarks)

Anchor configuration
--------------------
The defaults target 640×640 input with SCRFD-10G.  Override any of the
following in the stage ``config`` block to support other variants::

    config:
      strides: [8, 16, 32]
      feat_sizes: [[80, 80], [40, 40], [20, 20]]
      num_anchors: 2
      score_names: ["448", "471", "494"]
      box_names:   ["451", "474", "497"]
      landmark_names: ["454", "477", "500"]
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


# ── Default anchor configuration for SCRFD-10G at 640×640 input ─────────

_DEFAULT_STRIDES = [8, 16, 32]
_DEFAULT_SCORE_NAMES = ["448", "471", "494"]
_DEFAULT_BOX_NAMES = ["451", "474", "497"]
_DEFAULT_LANDMARK_NAMES = ["454", "477", "500"]
_DEFAULT_FEAT_SIZES = [(80, 80), (40, 40), (20, 20)]  # input_size / stride
_DEFAULT_NUM_ANCHORS = 2


class SCRFDDetectionStage(PipelineStage):
    """Run SCRFD face detection inference and decode results."""

    def __init__(
        self,
        inference_client: InferenceClient,
        model_registry: ModelRegistry,
    ) -> None:
        self._client = inference_client
        self._model_registry = model_registry

    async def execute(
        self,
        context: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Run SCRFD inference and return decoded bounding-box detections."""
        model_name = config.get("model_name", "face_detection_model")
        input_key = config.get("input_key", "preprocessed_image")
        output_key = config.get("output_key", "detections")
        triton_input_name = config.get("triton_input_name", "input.1")

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
                inputs={triton_input_name: image},
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
        """Raise ``StageValidationError`` if the preprocessed-image input key is absent."""
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

        # Anchor configuration — use config overrides or module defaults
        strides: list[int] = config.get("strides", _DEFAULT_STRIDES)
        raw_feat_sizes = config.get("feat_sizes", _DEFAULT_FEAT_SIZES)
        feat_sizes: list[tuple[int, int]] = [tuple(fs) for fs in raw_feat_sizes]  # type: ignore[misc]
        num_anchors: int = config.get("num_anchors", _DEFAULT_NUM_ANCHORS)

        # Tensor name overrides
        score_names: list[str] = config.get("score_names", _DEFAULT_SCORE_NAMES)
        box_names: list[str] = config.get("box_names", _DEFAULT_BOX_NAMES)
        landmark_names: list[str] = config.get("landmark_names", _DEFAULT_LANDMARK_NAMES)

        # Rebuild anchor centers using the active config
        anchor_centers = _build_anchor_centers(strides, feat_sizes, num_anchors)

        all_boxes: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        all_landmarks: list[np.ndarray] = []

        def _normalize_output(arr: Any, expected_cols: int) -> np.ndarray:
            """Strip leading batch dimensions and reshape to 2-D (N, cols)."""
            a = np.asarray(arr)
            while a.ndim > 2 and a.shape[0] == 1:
                a = a[0]
            if a.ndim == 3:
                a = a.reshape(-1, a.shape[-1])
            if expected_cols == 1 and a.ndim == 2 and a.shape[1] == 1:
                return a.reshape(-1)
            return a

        # Auto-detect output tensor names by shape when the model uses
        # different names (e.g. after re-export or '_reshaped' suffix).
        def _infer_output_names(
            raw_outputs: dict[str, Any],
        ) -> tuple[list[str], list[str], list[str]] | None:
            """Infer score, box, and landmark tensor names from output shapes, or return None."""
            info: dict[int, list[tuple[str, int]]] = {}
            for k, v in raw_outputs.items():
                a = np.asarray(v)
                while a.ndim > 2 and a.shape[0] == 1:
                    a = a[0]
                if a.ndim == 1:
                    anchors, cols = a.shape[0], 1
                elif a.ndim == 2:
                    anchors, cols = a.shape
                else:
                    a2 = a.reshape(-1, a.shape[-1])
                    anchors, cols = a2.shape
                info.setdefault(cols, []).append((k, anchors))

            num_strides = len(strides)
            expected_total = num_strides * 3  # scores + boxes + landmarks
            if set(info.keys()) >= {1, 4, 10} and sum(len(v) for v in info.values()) >= expected_total:
                def sort_keys(group: list[tuple[str, int]]) -> list[str]:
                    """Sort tensor names by descending anchor count and return the name list."""
                    return [name for name, _ in sorted(group, key=lambda x: -x[1])]

                return (
                    sort_keys(info[1])[:num_strides],
                    sort_keys(info[4])[:num_strides],
                    sort_keys(info[10])[:num_strides],
                )

            # '_reshaped' suffix fallback
            stripped = {k.rstrip("_reshaped"): k for k in raw_outputs if k.endswith("_reshaped")}
            if stripped:
                rev = {v: k for k, v in stripped.items()}
                scores = [rev.get(n, n) for n in score_names if n in rev or n in raw_outputs]
                boxes = [rev.get(n, n) for n in box_names if n in rev or n in raw_outputs]
                lmks = [rev.get(n, n) for n in landmark_names if n in rev or n in raw_outputs]
                if len(scores) == num_strides and len(boxes) == num_strides and len(lmks) == num_strides:
                    return scores, boxes, lmks

            return None

        inferred = _infer_output_names(raw)
        if inferred is not None:
            s_names, b_names, l_names = inferred
            logger.debug("Auto-detected SCRFD output names: %s, %s, %s", s_names, b_names, l_names)
        else:
            s_names, b_names, l_names = score_names, box_names, landmark_names

        for i, stride in enumerate(strides):
            scores = _normalize_output(raw[s_names[i]], expected_cols=1).reshape(-1)
            boxes_raw = _normalize_output(raw[b_names[i]], expected_cols=4).reshape(-1, 4)
            lmk_raw = _normalize_output(raw[l_names[i]], expected_cols=10).reshape(-1, 10)
            centers = anchor_centers[i]

            mask = scores >= conf_thresh
            if not np.any(mask):
                continue

            scores_f = scores[mask]
            boxes_f = boxes_raw[mask]
            lmk_f = lmk_raw[mask]
            anchors_f = centers[mask]

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

        keep = greedy_nms(boxes, scores, nms_thresh)
        boxes = boxes[keep]
        scores = scores[keep]
        landmarks = landmarks[keep]

        return [
            {
                "x1": float(box[0] / det_scale),
                "y1": float(box[1] / det_scale),
                "x2": float(box[2] / det_scale),
                "y2": float(box[3] / det_scale),
                "confidence": float(score),
                "landmarks": (lmk / det_scale).reshape(5, 2).tolist(),
            }
            for box, score, lmk in zip(boxes, scores, landmarks)
        ]


# ------------------------------------------------------------------
# SCRFD decoding helpers
# ------------------------------------------------------------------

def _build_anchor_centers(
    strides: list[int],
    feat_sizes: list[tuple[int, int]],
    num_anchors: int,
) -> list[np.ndarray]:
    """Pre-compute anchor centre coordinates for each stride level."""
    centers = []
    for stride, (fh, fw) in zip(strides, feat_sizes):
        y, x = np.mgrid[:fh, :fw]
        grid = np.stack([x.ravel(), y.ravel()], axis=1).astype(np.float32)
        grid = np.repeat(grid, num_anchors, axis=0)
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
    """Convert anchor + distances to landmark coordinates (5 points × 2)."""
    cx = anchors[:, 0] * stride
    cy = anchors[:, 1] * stride
    lmk = distances.copy()
    for j in range(5):
        lmk[:, j * 2] = cx + lmk[:, j * 2] * stride
        lmk[:, j * 2 + 1] = cy + lmk[:, j * 2 + 1] * stride
    return lmk
