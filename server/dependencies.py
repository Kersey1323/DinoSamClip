"""
Pipeline singleton dependency injection.
The DinoSAMClipPipeline is loaded once on startup and shared across all requests.
"""

from __future__ import annotations

import sys
import os

# Ensure project root is on sys.path so `settings` and `core` are importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.pipeline import DinoSAMClipPipeline

# Module-level singleton; populated by lifespan in main.py
_pipeline: DinoSAMClipPipeline | None = None


def set_pipeline(p: DinoSAMClipPipeline) -> None:
    global _pipeline
    _pipeline = p


def get_pipeline() -> DinoSAMClipPipeline:
    """FastAPI dependency that returns the shared pipeline instance."""
    if _pipeline is None:
        raise RuntimeError("Pipeline has not been initialised yet.")
    return _pipeline
