# HTTP API

Base path: `/api/v1`

## Health

- `GET /health`
- `GET /status`

## Pipeline management

- `GET /pipelines`
- `POST /pipelines/{name}/activate`
- `POST /pipelines/{name}/deactivate`

## Pipeline execution

### Face detection

`POST /pipelines/face_detection`

Multipart form fields:

- `image` (required)

### Face comparison

`POST /pipelines/face_comparison`

Multipart form fields:

- `image_1` (required)
- `image_2` (required)

## Response shape

Execution endpoints return a typed `result` payload.

- Face detection returns detections
- Face comparison returns grouped sections: `image_1`, `image_2`, `comparison`
