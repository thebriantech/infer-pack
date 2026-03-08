# Architecture

InferPack separates responsibilities into three planes:

## Control Plane

- Pipeline registry and discovery
- Model registry and reference counting
- Model lifecycle transitions (load/warm/idle/unload)

## Execution Plane

- Pipeline execution orchestration
- Stage-by-stage data flow
- Timeout and retry handling

## Inference Plane

- Triton HTTP/gRPC communication
- Model metadata and readiness checks
- Tensor inference calls

## API Layer

- `api/http/`: FastAPI routes and HTTP schemas
- `api/grpc/`: gRPC proto and servicer
- `api/shared/`: shared transport-agnostic pipeline business logic

Both HTTP and gRPC use shared logic in `api/shared/pipeline_service.py` to avoid duplicated behavior.
