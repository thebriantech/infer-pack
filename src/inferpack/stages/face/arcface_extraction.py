"""ArcFace feature extraction stage.

Sends cropped and aligned face images to the ArcFace model via Triton
and returns L2-normalised 512-d embedding vectors.

The w600k_r50 model expects:
    Input :  ``input.1``  [1, 3, 112, 112] FP32, BGR, (pixel-127.5)/127.5
    Output:  ``683``      [1, 512] FP32

Stage type: ``arcface_extraction`` (alias: ``face_feature_extraction``)

Inputs (context):
    - ``cropped_faces``  (list of numpy arrays, NCHW BGR float32)

Outputs (context):
    - ``embeddings``  (list of numpy arrays, each 512-d L2-normalised)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from inferpack.core.errors import StageValidationError
from inferpack.core.interfaces import InferenceClient, PipelineStage
from inferpack.control.model_registry import ModelRegistry


class ArcFaceExtractionStage(PipelineStage):
    """Extract face embedding vectors via ArcFace Triton inference."""

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
        """Run ArcFace inference on each cropped face and return L2-normalised embeddings."""
        model_name = config.get("model_name", "face_feature_extraction_model")
        input_key = config.get("input_key", "cropped_faces")
        output_key = config.get("output_key", "embeddings")
        triton_input_name = config.get("triton_input_name", "input.1")
        triton_output_name = config.get("triton_output_name", "683")
        enable_triton_batching = bool(config.get("enable_triton_batching", False))
        triton_batch_size = max(1, int(config.get("triton_batch_size", 8)))

        faces: list[np.ndarray] = context[input_key]

        embeddings: list[np.ndarray] = []
        self._model_registry.begin_inference(model_name)
        try:
            if enable_triton_batching and len(faces) > 1:
                for idx in range(0, len(faces), triton_batch_size):
                    chunk = faces[idx : idx + triton_batch_size]
                    batch_input = np.concatenate(chunk, axis=0)

                    raw_output = await self._client.infer(
                        model_name=model_name,
                        inputs={triton_input_name: batch_input},
                    )

                    batch_embeddings = np.asarray(
                        raw_output.get(
                            triton_output_name,
                            raw_output.get("output", []),
                        )
                    )

                    if batch_embeddings.ndim == 1:
                        batch_embeddings = np.expand_dims(batch_embeddings, axis=0)

                    for emb in batch_embeddings:
                        embedding = emb.flatten().astype(np.float32)
                        norm = np.linalg.norm(embedding)
                        if norm > 0:
                            embedding = embedding / norm
                        embeddings.append(embedding)
            else:
                for face_arr in faces:
                    raw_output = await self._client.infer(
                        model_name=model_name,
                        inputs={triton_input_name: face_arr},
                    )
                    embedding = np.asarray(
                        raw_output.get(
                            triton_output_name,
                            raw_output.get("output", []),
                        )
                    ).flatten().astype(np.float32)

                    # L2-normalise
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm

                    embeddings.append(embedding)
        finally:
            self._model_registry.end_inference(model_name)

        return {output_key: embeddings}

    def validate_inputs(
        self,
        context: dict[str, Any],
        config: dict[str, Any],
    ) -> None:
        """Raise ``StageValidationError`` if the cropped-faces input key is absent."""
        input_key = config.get("input_key", "cropped_faces")
        if input_key not in context:
            raise StageValidationError(f"Missing required input: {input_key}")
