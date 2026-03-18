# InferPack — Phase 1

## Version: Initial Release (March 2026)

This is the first public release of InferPack.

**Highlights:**
- Ready-to-use AI pipelines: face_detection, face_comparison
- HTTP and gRPC APIs
- Docker-first deployment (with NVIDIA Triton)
- Model and pipeline lifecycle management
- Shared business logic for all transports
- Custom pipeline and model support

See below for full usage and architecture details.

Production-grade AI inference runtime with ready-to-use pipelines, powered by NVIDIA Triton Inference Server.

## Overview

InferPack provides:

- **Ready-to-use AI pipelines** — built-in `face_detection` and `face_comparison`, plus custom YAML pipelines.
- **Dual API transports** — HTTP (FastAPI) and gRPC endpoints.
- **Shared API business logic** — both HTTP and gRPC reuse `src/inferpack/api/shared/pipeline_service.py`.
- **Docker-first deployment** — one compose stack for Triton + runtime.
- **Model lifecycle management** — load / warm / idle / unload with reference counting.

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│            API Layer (HTTP + gRPC transports)              │
├──────────────┬──────────────────────┬──────────────────────┤
│ Control Plane│   Execution Plane    │   Inference Plane    │
│              │                      │                      │
│ • Pipeline   │ • Pipeline executor  │ • Triton HTTP client │
│   registry   │ • Stage runner       │ • Triton gRPC client │
│ • Model      │ • Timeout & retry    │ • Model load/unload  │
│   registry   │ • Explicit data flow │ • Inference calls    │
│ • Lifecycle  │                      │                      │
│   manager    │                      │                      │
└──────────────┴──────────────────────┴──────────────────────┘
                          │
                          ▼
               ┌──────────────────────────┐
               │ Triton Inference Server  │
               └──────────────────────────┘
```

The planes are strictly separated:

| Plane | Responsibility | Does NOT |
| --- | --- | --- |
| Control | Pipeline/model registry, lifecycle, model readiness decisions | Execute stages, call Triton infer directly |
| Execution | Stage orchestration, timeout, retry, context data flow | Resource/lifecycle decisions |
| Inference | Triton protocol communication (HTTP/gRPC) | Business logic |

## Project Structure

```
infer-pack/
├── config/
│   └── inferpack.yaml
├── docker/
│   ├── Dockerfile.inferpack
│   ├── Dockerfile.triton
│   └── docker-compose.yaml
├── models/
│   ├── face_detection_model/
│   └── face_feature_extraction_model/
├── pipelines/
│   ├── builtins/
│   └── custom/
├── scripts/
│   └── gen_proto.sh
├── tools/
│   └── test_pipeline_api.py
├── src/inferpack/
│   ├── api/
│   │   ├── http/
│   │   ├── grpc/
│   │   └── shared/
│   ├── control/
│   ├── core/
│   ├── execution/
│   ├── inference/
│   ├── stages/
│   └── main.py
└── README.md
```

## Quick Start

### Prerequisites

- Docker + Docker Compose
- NVIDIA Container Toolkit (for GPU Triton)
- Model files placed in `models/`

### Run with Docker Compose

```bash
cd docker
docker compose up -d --build
```

Exposed ports are configurable via `docker/docker-compose.yaml` and
`config/inferpack.yaml`.

- InferPack HTTP API: `http://<host>:<inferpack_http_port>`
- InferPack HTTP docs: `http://<host>:<inferpack_http_port>/docs`
- InferPack gRPC API: `<host>:<inferpack_grpc_port>`
- Triton HTTP: `http://<host>:<triton_http_port>`
- Triton gRPC: `<host>:<triton_grpc_port>`
- Triton Metrics: `http://<host>:<triton_metrics_port>/metrics`

### Run locally (development)

```bash
pip install -r requirements.txt
PYTHONPATH=src INFERPACK_TRITON_HOST=localhost python -m inferpack.main
```

## API Reference

### Health / Status (HTTP)

| Endpoint | Method | Description |
| --- | --- | --- |
| `/api/v1/health` | GET | Health / readiness |
| `/api/v1/status` | GET | Full system status |

### Pipelines (HTTP)

| Endpoint | Method | Description |
| --- | --- | --- |
| `/api/v1/pipelines` | GET | List pipelines |
| `/api/v1/pipelines/{name}/activate` | POST | Activate pipeline |
| `/api/v1/pipelines/{name}/deactivate` | POST | Deactivate pipeline |
| `/api/v1/pipelines/face_detection` | POST | Execute face detection (requires `image`) |
| `/api/v1/pipelines/face_comparison` | POST | Execute face comparison (requires `image_1`, `image_2`) |

### Pipelines (gRPC)

Defined in `src/inferpack/api/grpc/inferpack.proto` service `PipelineService`:

- `ListPipelines`
- `ActivatePipeline`
- `DeactivatePipeline`
- `ExecutePipeline` (generic)
- `ExecuteFaceDetection` (1 image)
- `ExecuteFaceComparison` (2 images)

## Built-in Pipelines

### `face_detection`

```
image_preprocess → face_detection → detection_postprocess
```

### `face_comparison`

```
preprocess_1 → detect_1 → crop_1 → extract_1 →
preprocess_2 → detect_2 → crop_2 → extract_2 → compare
```

Notes:

- Face-comparison branches keep separate keys (`*_1`, `*_2`) to avoid collisions.
- Final API responses group outputs by `image_1`, `image_2`, and `comparison`.

## Triton Batching Support

InferPack supports Triton-side batching for feature extraction.

### Stage-level batching flags

In `face_feature_extraction` stage config:

- `enable_triton_batching: true|false`
- `triton_batch_size: <int>`

### Model-level batching

`models/face_feature_extraction_model/config.pbtxt` is configured for batching with:

- `max_batch_size > 0`
- `dynamic_batching { ... }`
- batch-compatible input/output dims

## Shared API Service Layer

Transport-independent pipeline operations live in:

- `src/inferpack/api/shared/pipeline_service.py`

Both HTTP routes and gRPC servicer call this shared service to avoid duplicated business logic.

## Testing Tools

### Generate gRPC Python stubs

```bash
python scripts/gen_proto.sh
```

### Test HTTP/gRPC endpoints quickly

```bash
# HTTP face detection
python tools/test_pipeline_api.py --protocol http --face-detection /path/image.jpg

# HTTP face comparison
python tools/test_pipeline_api.py --protocol http --face-comparison /path/img1.jpg /path/img2.jpg

# gRPC face comparison
python tools/test_pipeline_api.py --protocol grpc --face-comparison /path/img1.jpg /path/img2.jpg
```

## Adding Custom Pipelines

1. Create YAML in `pipelines/custom/`.
2. Register referenced model in `models/` with `config.pbtxt` and model files.
3. Use existing stage types or add new stages under `src/inferpack/stages/`.
4. Restart service to rediscover pipelines.

## Configuration

Config priority: **env vars > YAML > defaults**.

Primary config file: `config/inferpack.yaml`

Useful keys:

- `server.host`, `server.port`, `server.grpc_port`
- `triton.host`, `triton.http_port`, `triton.grpc_port`, `triton.protocol`
- `execution.max_concurrent_executions`

## Roadmap (Phase 2+)

- Kubernetes deployment (Helm/operator)
- Multi-GPU scheduling
- Multi-node support
- Pipeline hot-reload without restart
- Authentication & authorization
- Model versioning / A-B testing
- Metrics export (Prometheus)
- Distributed tracing (OpenTelemetry)
- Pipeline DAG execution (parallel stages)

## License

Apache 2.0 — see [LICENSE](LICENSE).