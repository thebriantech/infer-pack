# InferPack

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
cd docker
docker compose up -d --build
```

Then open:

- `http://<host>:<inferpack_http_port>/docs` (Swagger)
