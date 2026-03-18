"""Configuration management for InferPack.

Configuration is loaded with the following priority (highest first):
    1. Environment variables  (INFERPACK_<SECTION>_<KEY>)
    2. YAML config file
    3. Built-in defaults
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TritonConfig:
    """Triton Inference Server connection settings."""

    host: str = "localhost"
    http_port: int = 8000
    grpc_port: int = 8001
    metrics_port: int = 8002
    model_repository: str = "/models"
    protocol: str = "http"  # "http" or "grpc"

    @property
    def http_url(self) -> str:
        """Return the full HTTP base URL for the Triton server."""
        return f"http://{self.host}:{self.http_port}"

    @property
    def grpc_url(self) -> str:
        """Return the host:port gRPC address for the Triton server."""
        return f"{self.host}:{self.grpc_port}"


@dataclass
class ModelLifecycleConfig:
    """Model lifecycle management settings."""

    idle_timeout_seconds: float = 300.0
    warmup_requests: int = 1
    lifecycle_check_interval_seconds: float = 30.0
    max_concurrent_loads: int = 1


@dataclass
class ExecutionConfig:
    """Pipeline execution settings."""

    default_stage_timeout_seconds: float = 30.0
    default_max_retries: int = 0
    max_concurrent_executions: int = 10


@dataclass
class ServerConfig:
    """InferPack server settings (HTTP + gRPC)."""

    host: str = "0.0.0.0"
    port: int = 9000
    grpc_port: int = 9001
    log_level: str = "info"
    cors_origins: list[str] = field(default_factory=lambda: ["*"])


@dataclass
class InferPackConfig:
    """Root configuration for the InferPack runtime."""

    server: ServerConfig = field(default_factory=ServerConfig)
    triton: TritonConfig = field(default_factory=TritonConfig)
    model_lifecycle: ModelLifecycleConfig = field(default_factory=ModelLifecycleConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    builtin_pipelines_dir: str = "pipelines/builtins"
    custom_pipelines_dir: str = "pipelines/custom"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(config_path: str | None = None) -> InferPackConfig:
    """Build an ``InferPackConfig`` from defaults → YAML → env vars."""

    config = InferPackConfig()

    # 1. YAML file
    path = config_path or os.environ.get("INFERPACK_CONFIG", "config/inferpack.yaml")
    if Path(path).exists():
        logger.info("Loading config from %s", path)
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        _apply_yaml(config, data)
    else:
        logger.info("No config file at %s — using defaults", path)

    # 2. Environment overrides
    _apply_env(config)

    return config


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_yaml(config: InferPackConfig, data: dict[str, Any]) -> None:
    """Merge *data* from a parsed YAML file into *config* in-place."""

    if "server" in data:
        s = data["server"]
        config.server.host = s.get("host", config.server.host)
        config.server.port = s.get("port", config.server.port)
        config.server.grpc_port = s.get("grpc_port", config.server.grpc_port)
        config.server.log_level = s.get("log_level", config.server.log_level)
        if "cors_origins" in s:
            config.server.cors_origins = list(s["cors_origins"])

    if "triton" in data:
        t = data["triton"]
        config.triton.host = t.get("host", config.triton.host)
        config.triton.http_port = t.get("http_port", config.triton.http_port)
        config.triton.grpc_port = t.get("grpc_port", config.triton.grpc_port)
        config.triton.metrics_port = t.get("metrics_port", config.triton.metrics_port)
        config.triton.model_repository = t.get(
            "model_repository", config.triton.model_repository
        )
        config.triton.protocol = t.get("protocol", config.triton.protocol)

    if "model_lifecycle" in data:
        m = data["model_lifecycle"]
        config.model_lifecycle.idle_timeout_seconds = m.get(
            "idle_timeout_seconds", config.model_lifecycle.idle_timeout_seconds
        )
        config.model_lifecycle.warmup_requests = m.get(
            "warmup_requests", config.model_lifecycle.warmup_requests
        )
        config.model_lifecycle.lifecycle_check_interval_seconds = m.get(
            "lifecycle_check_interval_seconds",
            config.model_lifecycle.lifecycle_check_interval_seconds,
        )
        config.model_lifecycle.max_concurrent_loads = m.get(
            "max_concurrent_loads", config.model_lifecycle.max_concurrent_loads
        )

    if "execution" in data:
        e = data["execution"]
        config.execution.default_stage_timeout_seconds = e.get(
            "default_stage_timeout_seconds",
            config.execution.default_stage_timeout_seconds,
        )
        config.execution.default_max_retries = e.get(
            "default_max_retries", config.execution.default_max_retries
        )
        config.execution.max_concurrent_executions = e.get(
            "max_concurrent_executions", config.execution.max_concurrent_executions
        )

    config.builtin_pipelines_dir = data.get(
        "builtin_pipelines_dir", config.builtin_pipelines_dir
    )
    config.custom_pipelines_dir = data.get(
        "custom_pipelines_dir", config.custom_pipelines_dir
    )


def _apply_env(config: InferPackConfig) -> None:
    """Override configuration values with ``INFERPACK_*`` environment variables."""

    _env_map: dict[str, Any] = {
        "INFERPACK_SERVER_HOST": ("server", "host", str),
        "INFERPACK_SERVER_PORT": ("server", "port", int),
        "INFERPACK_SERVER_GRPC_PORT": ("server", "grpc_port", int),
        "INFERPACK_SERVER_LOG_LEVEL": ("server", "log_level", str),
        "INFERPACK_TRITON_HOST": ("triton", "host", str),
        "INFERPACK_TRITON_HTTP_PORT": ("triton", "http_port", int),
        "INFERPACK_TRITON_GRPC_PORT": ("triton", "grpc_port", int),
        "INFERPACK_TRITON_METRICS_PORT": ("triton", "metrics_port", int),
        "INFERPACK_TRITON_MODEL_REPOSITORY": ("triton", "model_repository", str),
        "INFERPACK_TRITON_PROTOCOL": ("triton", "protocol", str),
        "INFERPACK_MODEL_IDLE_TIMEOUT": ("model_lifecycle", "idle_timeout_seconds", float),
        "INFERPACK_MODEL_WARMUP_REQUESTS": ("model_lifecycle", "warmup_requests", int),
    }

    for env_key, (section, attr, cast) in _env_map.items():
        value = os.environ.get(env_key)
        if value is not None:
            setattr(getattr(config, section), attr, cast(value))

    # Top-level overrides
    for env_key, attr in [
        ("INFERPACK_BUILTIN_PIPELINES_DIR", "builtin_pipelines_dir"),
        ("INFERPACK_CUSTOM_PIPELINES_DIR", "custom_pipelines_dir"),
    ]:
        value = os.environ.get(env_key)
        if value is not None:
            setattr(config, attr, value)

    # CORS origins — comma-separated env var overrides the list
    cors_env = os.environ.get("INFERPACK_SERVER_CORS_ORIGINS", "").strip()
    if cors_env:
        config.server.cors_origins = [o.strip() for o in cors_env.split(",") if o.strip()]
