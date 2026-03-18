"""Feature comparison stage — cosine similarity between embeddings.

A **shared** stage: compares any pair of L2-normalised embedding vectors
regardless of the model that produced them.

Stage type: ``feature_comparison``

Inputs (context):
    - two lists of embedding arrays (configurable keys)

Outputs (context):
    - ``similarity_score`` (float, 0.0–1.0)
    - ``is_same_person``   (bool — legacy name; semantically = is_match)
    - ``is_match``         (bool)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from inferpack.core.errors import StageValidationError
from inferpack.core.interfaces import PipelineStage


class FeatureComparisonStage(PipelineStage):
    """Compute cosine similarity between two embedding vectors."""

    async def execute(
        self,
        context: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute cosine similarity between two embedding vectors and apply a threshold."""
        threshold = config.get("similarity_threshold", 0.6)
        key1 = config.get("embedding_key_1", "embeddings_1")
        key2 = config.get("embedding_key_2", "embeddings_2")

        emb1: list[np.ndarray] = context[key1]
        emb2: list[np.ndarray] = context[key2]

        v1 = emb1[0].flatten().astype(np.float64)
        v2 = emb2[0].flatten().astype(np.float64)

        score = self._cosine_similarity(v1, v2)
        is_match = bool(score >= threshold)

        return {
            "similarity_score": float(score),
            "is_match": is_match,
            # Legacy alias
            "is_same_person": is_match,
        }

    def validate_inputs(
        self,
        context: dict[str, Any],
        config: dict[str, Any],
    ) -> None:
        """Raise ``StageValidationError`` if either embedding key is absent or empty."""
        key1 = config.get("embedding_key_1", "embeddings_1")
        key2 = config.get("embedding_key_2", "embeddings_2")
        for key in [key1, key2]:
            if key not in context:
                raise StageValidationError(f"Missing required input: {key}")
            embeddings = context[key]
            if not embeddings or len(embeddings) == 0:
                raise StageValidationError(f"No embeddings in {key}")

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Return the cosine similarity between vectors *a* and *b*, or 0.0 if either is zero."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
