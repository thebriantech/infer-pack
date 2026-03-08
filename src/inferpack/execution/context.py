"""Pipeline execution context — explicit data flow between stages.

The context is a plain dictionary that carries all data produced by
previous stages.  Each stage reads named keys and writes new ones.
There is NO hidden state.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from inferpack.core.types import StageResult, StageStatus


@dataclass
class PipelineExecutionContext:
    """Mutable context passed through every stage of a pipeline execution.

    Attributes:
        execution_id: Unique identifier for this execution run.
        pipeline_name: Name of the pipeline being executed.
        data: The shared data dictionary — stages read and write here.
        stage_results: Ordered list of per-stage outcomes.
        start_time: Wall-clock start of execution.
    """

    pipeline_name: str
    execution_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    data: dict[str, Any] = field(default_factory=dict)
    stage_results: list[StageResult] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    def set(self, key: str, value: Any) -> None:
        """Store a value in the context."""
        self.data[key] = value

    def get(self, key: str) -> Any:
        """Retrieve a value from the context (raises ``KeyError`` if absent)."""
        return self.data[key]

    def has(self, key: str) -> bool:
        return key in self.data

    def record_stage(self, result: StageResult) -> None:
        self.stage_results.append(result)

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of the execution."""
        return {
            "execution_id": self.execution_id,
            "pipeline_name": self.pipeline_name,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "stages": [
                {
                    "name": r.stage_name,
                    "status": r.status.value,
                    "duration_seconds": round(r.duration_seconds, 4),
                    "error": r.error,
                }
                for r in self.stage_results
            ],
        }
