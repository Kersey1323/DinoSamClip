"""
Pydantic response schemas for the DinoSamClip API.
Request parameters for file-upload routes are declared directly as
FastAPI Form/File fields in the route functions (not as Pydantic models),
because multipart/form-data cannot be mixed with a JSON body.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


# ─── Sub-models ───────────────────────────────────────────────────────────────

class Prediction(BaseModel):
    cls: str = Field(..., alias="class")
    confidence: float

    class Config:
        populate_by_name = True


class Detection(BaseModel):
    cls: str = Field(..., alias="class", description="Detected class name")
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: List[int] = Field(..., description="[x1, y1, x2, y2]")
    all_predictions: Optional[List[Prediction]] = None

    class Config:
        populate_by_name = True


# ─── Single-image inference responses ────────────────────────────────────────

class InferResponse(BaseModel):
    num_objects: int
    detections: List[Detection]
    visualization_base64: Optional[str] = Field(
        None, description="Base64 PNG with overlaid masks (only when return_visualization=true)"
    )
    elapsed_ms: float


# ─── Attention heatmap response ───────────────────────────────────────────────

class AttentionResponse(BaseModel):
    heatmap_base64: str = Field(..., description="Base64 PNG of DINOv2 attention heatmap")
    elapsed_ms: float


# ─── Batch inference ─────────────────────────────────────────────────────────

class BatchInferRequest(BaseModel):
    image_dir: str = Field(..., description="Absolute path to a directory of images on the server")
    output_dir: str = Field(..., description="Absolute path where annotated result images will be saved")
    candidate_classes: Optional[List[str]] = None
    num_prompts: int = Field(10, ge=1, le=100)
    confidence_threshold: float = Field(0.2, ge=0.0, le=1.0)
    auto_mode: bool = Field(False, description="Use SAM auto-grid mode (skip DINOv2)")
    dino_checkpoint: Optional[str] = Field(None, description="Path to a fine-tuned DINOv2 checkpoint (.pth)")


class BatchImageResult(BaseModel):
    filename: str
    num_objects: int
    detections: List[Detection]
    result_image_path: str
    error: Optional[str] = None


class BatchInferResponse(BaseModel):
    total_images: int
    processed: int
    failed: int
    output_dir: str
    results: List[BatchImageResult]
    elapsed_ms: float


# ─── Health check ─────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    app_env: str
    device: str
    cuda_available: bool
    pipeline_loaded: bool
    models: dict
