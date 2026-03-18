# InferPack

## Version: Initial Release (March 2026)

This is the first public release of InferPack.

**Highlights:**
- Built-in pipelines: face_detection, face_comparison
- HTTP and gRPC APIs
- Docker-first deployment (with NVIDIA Triton)
- Model and pipeline lifecycle management
- Shared business logic for all transports
- Custom pipeline and model support

InferPack is a production-oriented AI inference runtime built around NVIDIA Triton Inference Server.

It provides:

- HTTP and gRPC APIs
- Built-in pipelines (`face_detection`, `face_comparison`)
- Shared model lifecycle management
- Docker-first deployment

## Quick links

- [Architecture](architecture.md)
- [HTTP API](api-http.md)
- [gRPC API](api-grpc.md)
- [Pipelines](pipelines.md)
- [Deployment](deployment.md)
- [Troubleshooting](troubleshooting.md)

## Getting started

```bash
# 1. Download built-in models from HuggingFace
make download-models

# 2. Start the stack (InferPack + Triton)
make compose-up
```

Then open:

- `http://<host>:<inferpack_http_port>/docs` (Swagger)
