"""Pipeline management routes — list, activate, deactivate, execute."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, UploadFile, File

from inferpack.api.http.schemas import (
    FaceComparisonExecuteResponse,
    FaceComparisonImageInfoSchema,
    FaceComparisonMetricsSchema,
    FaceComparisonResultSchema,
    FaceDetectionExecuteResponse,
    FaceDetectionResultSchema,
    PipelineListResponse,
    PipelineSummary,
)
from inferpack.core.errors import (
    PipelineActivationError,
    PipelineNotFoundError,
)
from inferpack.api.shared.pipeline_service import PipelineService, make_json_safe

router = APIRouter(prefix="/pipelines", tags=["pipelines"])


@router.get("", response_model=PipelineListResponse)
async def list_pipelines(request: Request) -> PipelineListResponse:
    """List all registered pipelines."""
    service = PipelineService(request.app.state.app)
    return PipelineListResponse(
        pipelines=[
            PipelineSummary(**p)
            for p in service.list_pipelines()
        ]
    )


@router.post("/{name}/activate", status_code=200)
async def activate_pipeline(name: str, request: Request) -> dict[str, str]:
    """Activate a pipeline — loads its required models."""
    service = PipelineService(request.app.state.app)
    try:
        await service.activate_pipeline(name)
    except PipelineNotFoundError:
        raise HTTPException(status_code=404, detail=f"Pipeline not found: {name}")
    except PipelineActivationError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"status": "activated", "pipeline": name}


@router.post("/{name}/deactivate", status_code=200)
async def deactivate_pipeline(name: str, request: Request) -> dict[str, str]:
    """Deactivate a pipeline — releases model references."""
    service = PipelineService(request.app.state.app)
    try:
        await service.deactivate_pipeline(name)
    except PipelineNotFoundError:
        raise HTTPException(status_code=404, detail=f"Pipeline not found: {name}")
    except PipelineActivationError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"status": "deactivated", "pipeline": name}


@router.post("/face_detection", response_model=FaceDetectionExecuteResponse)
async def execute_face_detection_pipeline(
    request: Request,
    image: UploadFile = File(...),
) -> FaceDetectionExecuteResponse:
    """Execute the ``face_detection`` pipeline.

    Required multipart field:
      - ``image``
    """
    service = PipelineService(request.app.state.app)
    inputs = await _extract_http_inputs(
        image=image,
        image_1=None,
        image_2=None,
    )

    try:
        result = await service.execute_pipeline("face_detection", inputs)
    except PipelineNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except PipelineActivationError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    if not result.success:
        raise HTTPException(
            status_code=422,
            detail=result.error or "face_detection pipeline execution failed",
        )

    return _to_face_detection_response(result)


@router.post("/face_comparison", response_model=FaceComparisonExecuteResponse)
async def execute_face_comparison_pipeline(
    request: Request,
    image_1: UploadFile = File(...),
    image_2: UploadFile = File(...),
) -> FaceComparisonExecuteResponse:
    """Execute the ``face_comparison`` pipeline.

    Required multipart fields:
      - ``image_1``
      - ``image_2``
    """
    service = PipelineService(request.app.state.app)
    inputs = await _extract_http_inputs(
        image=None,
        image_1=image_1,
        image_2=image_2,
    )

    try:
        result = await service.execute_pipeline("face_comparison", inputs)
    except PipelineNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except PipelineActivationError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    if not result.success:
        raise HTTPException(
            status_code=422,
            detail=result.error or "face_comparison pipeline execution failed",
        )

    return _to_face_comparison_response(result)


async def _extract_http_inputs(
    image: UploadFile | None,
    image_1: UploadFile | None,
    image_2: UploadFile | None,
) -> dict[str, Any]:
    """Build pipeline inputs from multipart/form-data image fields."""
    inputs: dict[str, Any] = {}

    if image is not None:
        inputs["image_bytes"] = await image.read()
    if image_1 is not None:
        inputs["image_bytes_1"] = await image_1.read()
    if image_2 is not None:
        inputs["image_bytes_2"] = await image_2.read()

    return inputs


def _to_face_detection_response(result: Any) -> FaceDetectionExecuteResponse:
    safe_output = make_json_safe(result.output)
    detections = safe_output.get("detections", [])
    return FaceDetectionExecuteResponse(
        result=FaceDetectionResultSchema(detections=detections)
    )


def _to_face_comparison_response(result: Any) -> FaceComparisonExecuteResponse:
    safe_output = make_json_safe(result.output)

    comparison_raw = safe_output.get("comparison")
    if not comparison_raw:
        comparison_raw = {
            key: safe_output.get(key)
            for key in ("similarity_score", "is_match", "is_same_person")
            if key in safe_output
        }

    image_1_raw = safe_output.get("image_1")
    image_2_raw = safe_output.get("image_2")

    # Backward compatibility for legacy flat keys
    if image_1_raw is None:
        image_1_raw = {
            "original_size": safe_output.get("original_size_1") or safe_output.get("original_size"),
            "det_scale": safe_output.get("det_scale_1") or safe_output.get("det_scale"),
            "detections": safe_output.get("detections_1") or safe_output.get("detections") or [],
            "preprocessed_image": safe_output.get("preprocessed_image_1"),
            "cropped_faces": safe_output.get("cropped_faces_1"),
            "embeddings": safe_output.get("embeddings_1"),
        }
        if not any(v is not None and v != [] for v in image_1_raw.values()):
            image_1_raw = None

    if image_2_raw is None:
        image_2_raw = {
            "original_size": safe_output.get("original_size_2"),
            "det_scale": safe_output.get("det_scale_2"),
            "detections": safe_output.get("detections_2") or [],
            "preprocessed_image": safe_output.get("preprocessed_image_2"),
            "cropped_faces": safe_output.get("cropped_faces_2"),
            "embeddings": safe_output.get("embeddings_2"),
        }
        if not any(v is not None and v != [] for v in image_2_raw.values()):
            image_2_raw = None

    return FaceComparisonExecuteResponse(
        result=FaceComparisonResultSchema(
            comparison=FaceComparisonMetricsSchema(**comparison_raw),
            image_1=FaceComparisonImageInfoSchema(**image_1_raw) if image_1_raw else None,
            image_2=FaceComparisonImageInfoSchema(**image_2_raw) if image_2_raw else None,
        )
    )


