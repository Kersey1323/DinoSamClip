"""
FastAPI application entry point for DinoSamClip API.

Usage:
    APP_ENV=local uvicorn server.main:app --host 127.0.0.1 --port 8000 --reload
    APP_ENV=dev   uvicorn server.main:app --host 0.0.0.0  --port 8000
"""

from __future__ import annotations

import sys
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── Ensure project root is importable ─────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from server import dependencies
from server.routes import health, infer, batch
from settings import APP_ENV, ModelConfig, PipelineConfig, ServerConfig
from core.pipeline import DinoSAMClipPipeline


# ── Lifespan: load pipeline once on startup ───────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"\n[DinoSamClip API] APP_ENV = {APP_ENV}")
    print(f"[DinoSamClip API] Loading pipeline on device: {PipelineConfig.DEVICE}")

    pipeline = DinoSAMClipPipeline(
        device=PipelineConfig.DEVICE,
        sam_checkpoint=ModelConfig.SAM_CHECKPOINT_PATH,
    )
    dependencies.set_pipeline(pipeline)
    print("[DinoSamClip API] Pipeline ready.\n")

    yield

    print("[DinoSamClip API] Shutting down.")


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="DinoSamClip API",
    description=(
        "REST API for the DINOv2 + SAM + CLIP object detection & classification pipeline. "
        f"Active environment: **{APP_ENV}**"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routes ───────────────────────────────────────────────────────────

app.include_router(health.router)
app.include_router(infer.router)
app.include_router(batch.router)


@app.get("/", include_in_schema=False)
def root():
    return {"message": "DinoSamClip API is running. Visit /docs for the API documentation."}
