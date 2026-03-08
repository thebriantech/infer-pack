"""InferPack application entrypoint (HTTP + gRPC).

Creates the ASGI application, wires up routes, starts the gRPC server,
and manages startup / shutdown lifecycle for both transports.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import grpc
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from inferpack import __version__
from inferpack.config import load_config
from inferpack.api.http.dependencies import create_app_state
from inferpack.api.http.routes_health import router as health_router
from inferpack.api.http.routes_pipelines import router as pipelines_router
from inferpack.api.http.routes_models import router as models_router
from inferpack.api.grpc import inferpack_pb2_grpc
from inferpack.api.grpc.servicer import PipelineServicer


logger = logging.getLogger("inferpack")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup and shutdown logic."""
    config = load_config()

    # Logging
    logging.basicConfig(
        level=getattr(logging, config.server.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("InferPack %s starting", __version__)

    # Build all components
    state = create_app_state(config)
    app.state.app = state

    # Discover pipelines
    base_dir = Path(__file__).resolve().parent.parent.parent  # project root
    builtin_dir = str(base_dir / config.builtin_pipelines_dir)
    custom_dir = str(base_dir / config.custom_pipelines_dir)
    state.pipeline_registry.discover(builtin_dir, custom_dir)

    # Start model lifecycle background task
    state.lifecycle_manager.start()

    # ----- Start gRPC server alongside HTTP -----
    grpc_server = grpc.aio.server()
    inferpack_pb2_grpc.add_PipelineServiceServicer_to_server(
        PipelineServicer(state), grpc_server,
    )
    grpc_address = f"{config.server.host}:{config.server.grpc_port}"
    grpc_server.add_insecure_port(grpc_address)
    await grpc_server.start()

    logger.info(
        "InferPack ready — HTTP on %s:%d, gRPC on %s:%d",
        config.server.host,
        config.server.port,
        config.server.host,
        config.server.grpc_port,
    )

    yield

    # Shutdown
    logger.info("InferPack shutting down")
    await grpc_server.stop(grace=5)
    await state.lifecycle_manager.stop()
    await state.triton_client.close()


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Build and return the ASGI application."""
    app = FastAPI(
        title="InferPack",
        description="Production-grade AI inference runtime",
        version=__version__,
        lifespan=lifespan,
    )

    # CORS — permissive for the local UI; tighten in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routes
    app.include_router(health_router, prefix="/api/v1")
    app.include_router(pipelines_router, prefix="/api/v1")
    app.include_router(models_router, prefix="/api/v1")

    # Serve the minimal Web UI (static files) if the directory exists
    ui_dir = Path(__file__).resolve().parent.parent.parent / "ui"
    if ui_dir.is_dir():
        app.mount("/ui", StaticFiles(directory=str(ui_dir), html=True), name="ui")

    return app


# ---------------------------------------------------------------------------
# Module-level app for ``uvicorn inferpack.main:app``
# ---------------------------------------------------------------------------

app = create_app()


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the server directly via ``python -m inferpack.main``."""
    import uvicorn

    config = load_config()
    uvicorn.run(
        "inferpack.main:app",
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level,
        reload=False,
    )


if __name__ == "__main__":
    main()
