# gRPC API

Service: `PipelineService`

Defined in:

- `src/inferpack/api/grpc/inferpack.proto`

## RPC methods

- `ListPipelines`
- `ActivatePipeline`
- `DeactivatePipeline`
- `ExecutePipeline` (generic)
- `ExecuteFaceDetection` (1 image)
- `ExecuteFaceComparison` (2 images)

## Notes

- Generic and specialized execution RPCs reuse shared logic via `api/shared/pipeline_service.py`.
- Regenerate stubs after proto changes:

```bash
python scripts/gen_proto.sh
```
