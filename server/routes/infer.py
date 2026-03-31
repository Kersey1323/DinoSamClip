"""
Single-image inference routes.

All three endpoints accept the image in ONE of two ways (mutually exclusive):
  1. Multipart file upload  → `image_file` field (UploadFile)
  2. Server-side file path  → `image_path` form field (string)

Additional inference parameters are passed as Form fields alongside the file.

Endpoints:
  POST /infer          — Full DINOv2 + SAM + CLIP pipeline
  POST /infer/auto     — SAM auto-grid + CLIP (no DINOv2)
  POST /infer/attention — Return DINOv2 attention heatmap only
"""

from __future__ import annotations

import io
import time
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from PIL import Image

from server.dependencies import get_pipeline
from server.schemas import AttentionResponse, InferResponse
from server.utils import create_overlay, detections_to_dicts, numpy_to_b64

router = APIRouter(tags=["Inference"])


# ─── Shared helper: resolve image from upload OR path ─────────────────────────

def _load_image(image_file: Optional[UploadFile], image_path: Optional[str]) -> Image.Image:
    """
    Load a PIL Image from either an uploaded file or a server-side path.
    Raises HTTP 422 if neither or both are provided.
    """
    if image_file is None and not image_path:
        raise HTTPException(
            status_code=422,
            detail="Provide either 'image_file' (upload) or 'image_path' (server path), not neither."
        )
    if image_file is not None and image_path:
        raise HTTPException(
            status_code=422,
            detail="Provide either 'image_file' (upload) or 'image_path' (server path), not both."
        )

    if image_file is not None:
        try:
            contents = image_file.file.read()
            return Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Cannot read uploaded image: {e}")
    else:
        try:
            return Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            raise HTTPException(status_code=422, detail=f"image_path not found: {image_path}")
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Cannot open image_path: {e}")


# ─── POST /infer — Full pipeline ─────────────────────────────────────────────

@router.post(
    "/infer",
    response_model=InferResponse,
    summary="Single image: DINOv2 + SAM + CLIP",
    description=(
        "Upload an image **or** supply a server-side `image_path`. "
        "Runs the full DINOv2 → SAM → CLIP pipeline and returns detected objects."
    ),
)
def infer(
    image_file: Optional[UploadFile] = File(None, description="Image file to upload (JPEG / PNG / BMP …)"),
    image_path: Optional[str]        = Form(None, description="Absolute path to an image on the server"),
    candidate_classes: Optional[str] = Form(None, description="Comma-separated class names, e.g. 'person,car,dog'. Defaults to built-in 60-class list."),
    num_prompts: int                  = Form(10,   description="Number of DINOv2 attention prompts (1–100)"),
    confidence_threshold: float       = Form(0.2,  description="CLIP confidence threshold (0–1)"),
    return_visualization: bool        = Form(False, description="Return annotated result image as base64 PNG"),
    pipeline=Depends(get_pipeline),
):
    image = _load_image(image_file, image_path)
    classes = [c.strip() for c in candidate_classes.split(",")] if candidate_classes else None

    t0 = time.perf_counter()
    results = pipeline.detect_and_classify(
        image,
        candidate_classes=classes,
        num_prompts=num_prompts,
        confidence_threshold=confidence_threshold,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    viz_b64 = None
    if return_visualization:
        overlay = create_overlay(np.array(image), results["detections"])
        viz_b64 = numpy_to_b64(overlay)

    return InferResponse(
        num_objects=results["num_objects"],
        detections=detections_to_dicts(results["detections"]),
        visualization_base64=viz_b64,
        elapsed_ms=round(elapsed_ms, 2),
    )


# ─── POST /infer/auto — SAM auto-grid ────────────────────────────────────────

@router.post(
    "/infer/auto",
    response_model=InferResponse,
    summary="Single image: SAM auto-grid + CLIP",
    description=(
        "Upload an image **or** supply a server-side `image_path`. "
        "Uses SAM automatic grid prompts instead of DINOv2 attention maps."
    ),
)
def infer_auto(
    image_file: Optional[UploadFile] = File(None, description="Image file to upload"),
    image_path: Optional[str]        = Form(None, description="Absolute path to an image on the server"),
    candidate_classes: Optional[str] = Form(None, description="Comma-separated class names"),
    num_points: int                   = Form(64,   description="SAM grid point count (1–256)"),
    confidence_threshold: float       = Form(0.2,  description="CLIP confidence threshold (0–1)"),
    return_visualization: bool        = Form(False, description="Return annotated result image as base64 PNG"),
    pipeline=Depends(get_pipeline),
):
    image = _load_image(image_file, image_path)
    classes = [c.strip() for c in candidate_classes.split(",")] if candidate_classes else None

    t0 = time.perf_counter()
    results = pipeline.detect_and_classify_automatic(
        image,
        candidate_classes=classes,
        num_points=num_points,
        confidence_threshold=confidence_threshold,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    viz_b64 = None
    if return_visualization:
        overlay = create_overlay(np.array(image), results["detections"])
        viz_b64 = numpy_to_b64(overlay)

    return InferResponse(
        num_objects=results["num_objects"],
        detections=detections_to_dicts(results["detections"]),
        visualization_base64=viz_b64,
        elapsed_ms=round(elapsed_ms, 2),
    )


# ─── POST /infer/attention — DINOv2 heatmap ──────────────────────────────────

@router.post(
    "/infer/attention",
    response_model=AttentionResponse,
    summary="Single image: DINOv2 attention heatmap",
    description=(
        "Upload an image **or** supply a server-side `image_path`. "
        "Returns only the DINOv2 CLS-similarity attention heatmap overlaid on the original image."
    ),
)
def infer_attention(
    image_file: Optional[UploadFile] = File(None, description="Image file to upload"),
    image_path: Optional[str]        = Form(None, description="Absolute path to an image on the server"),
    pipeline=Depends(get_pipeline),
):
    image = _load_image(image_file, image_path)

    t0 = time.perf_counter()
    attention_map = pipeline.dinov2.extract_attention_map(
        image, target_size=(image.width, image.height)
    )
    heatmap_arr = pipeline.dinov2.get_attention_visualization(image, attention_map)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return AttentionResponse(
        heatmap_base64=numpy_to_b64(heatmap_arr),
        elapsed_ms=round(elapsed_ms, 2),
    )
