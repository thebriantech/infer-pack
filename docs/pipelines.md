# Pipelines

Pipelines are YAML-defined stage graphs discovered from:

- `pipelines/builtins/`
- `pipelines/custom/`

## Built-in pipelines

### `face_detection`

Flow:

`image_preprocess -> face_detection -> detection_postprocess`

### `face_comparison`

Flow:

`preprocess_1 -> detect_1 -> crop_1 -> extract_1 -> preprocess_2 -> detect_2 -> crop_2 -> extract_2 -> compare`

## Reuse and key isolation

- Models can be shared across multiple pipelines.
- Branch-specific keys (`*_1`, `*_2`) prevent collisions in multi-image pipelines.

## Batching

Feature extraction supports Triton-side batching via stage config:

- `enable_triton_batching: true|false`
- `triton_batch_size: <int>`
