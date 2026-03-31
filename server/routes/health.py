"""
GET /health — Service and model status.
"""

import torch
from fastapi import APIRouter, Depends

from server.dependencies import get_pipeline
from server.schemas import HealthResponse
from settings import APP_ENV, ModelConfig, PipelineConfig

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="Service health check")
def health(pipeline=Depends(get_pipeline)):
    return HealthResponse(
        status="ok",
        app_env=APP_ENV,
        device=pipeline.device,
        cuda_available=torch.cuda.is_available(),
        pipeline_loaded=True,
        models={
            "dinov2": ModelConfig.DINOV2_MODEL_NAME,
            "sam": ModelConfig.SAM_CHECKPOINT_PATH,
            "sam_type": ModelConfig.SAM_MODEL_TYPE,
            "clip": ModelConfig.CLIP_MODEL_NAME,
        },
    )
