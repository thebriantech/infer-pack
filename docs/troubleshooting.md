# Troubleshooting

## Swagger UI hangs after successful run

Cause: very large payload rendering.

Fix: compact output summaries are enabled in shared response shaping.

## Pipeline says no detection but model logs decoded detections

Cause: key mismatch between detection output and downstream stage inputs.

Fix: ensure `output_key` and `detection_keys` align in pipeline YAML.

## Protobuf gencode/runtime version mismatch

Cause: generated stubs produced by a different protobuf version.

Fix:

```bash
python scripts/gen_proto.sh
docker compose up -d --build
```

## Triton model load fails with memory allocation errors

Check:

- GPU availability inside container
- model instance kind (CPU/GPU)
- model config batch/shape settings
