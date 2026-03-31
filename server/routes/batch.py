"""
POST /batch/infer — Batch directory inference.
Processes all images in a local directory and saves annotated results.
"""

from __future__ import annotations

import os
import time
import torch
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from fastapi import APIRouter, Depends, HTTPException

from server.dependencies import get_pipeline
from server.schemas import BatchInferRequest, BatchInferResponse, BatchImageResult
from server.utils import create_overlay, detections_to_dicts

router = APIRouter(tags=["Batch"])

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")


def _save_result(image: Image.Image, detections: list, attention_map, save_path: str):
    """Save annotated result figure (overlay + optional heatmap + summary)."""
    img_arr = np.array(image)
    overlay = create_overlay(img_arr, detections, alpha=0.4)

    n_cols = 3 if attention_map is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
    if n_cols == 1:
        axes = [axes]

    axes[0].imshow(overlay)
    axes[0].set_title("Segmentation Overlay")
    axes[0].axis("off")

    if attention_map is not None:
        im = axes[1].imshow(attention_map, cmap="jet")
        axes[1].set_title("DINOv2 Attention Heatmap")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    ax_text = axes[-1]
    ax_text.axis("off")
    summary = "Detection Summary:\n\n"
    if not detections:
        summary += "No objects detected."
    else:
        for i, det in enumerate(detections):
            summary += f"{i+1}. {det['class']}: {det['confidence']:.1%}\n"
    ax_text.text(0.05, 0.95, summary, fontsize=11, va="top", transform=ax_text.transAxes)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@router.post("/batch/infer", response_model=BatchInferResponse, summary="Batch directory inference")
def batch_infer(req: BatchInferRequest, pipeline=Depends(get_pipeline)):
    if not os.path.isdir(req.image_dir):
        raise HTTPException(status_code=422, detail=f"image_dir not found: {req.image_dir}")

    os.makedirs(req.output_dir, exist_ok=True)

    if req.dino_checkpoint and not req.auto_mode:
        if not os.path.isfile(req.dino_checkpoint):
            raise HTTPException(status_code=422, detail=f"dino_checkpoint not found: {req.dino_checkpoint}")
        try:
            state = torch.load(req.dino_checkpoint, map_location=pipeline.device)
            pipeline.dinov2.model.load_state_dict(state)
            pipeline.dinov2.model.eval()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load dino_checkpoint: {e}")

    image_files = sorted(
        f for f in os.listdir(req.image_dir) if f.lower().endswith(VALID_EXTS)
    )
    if not image_files:
        raise HTTPException(status_code=422, detail="No valid images found in image_dir.")

    t0 = time.perf_counter()
    results: List[BatchImageResult] = []
    processed = failed = 0

    for fname in image_files:
        img_path = os.path.join(req.image_dir, fname)
        out_name = os.path.splitext(fname)[0] + "_result.png"
        out_path = os.path.join(req.output_dir, out_name)

        try:
            image = Image.open(img_path).convert("RGB")

            if req.auto_mode:
                raw = pipeline.detect_and_classify_automatic(
                    image,
                    candidate_classes=req.candidate_classes,
                    confidence_threshold=req.confidence_threshold,
                )
            else:
                raw = pipeline.detect_and_classify(
                    image,
                    candidate_classes=req.candidate_classes,
                    num_prompts=req.num_prompts,
                    confidence_threshold=req.confidence_threshold,
                    image_name=fname,
                )

            _save_result(image, raw["detections"], raw.get("attention_map"), out_path)

            results.append(BatchImageResult(
                filename=fname,
                num_objects=raw["num_objects"],
                detections=detections_to_dicts(raw["detections"]),
                result_image_path=out_path,
            ))
            processed += 1

        except Exception as e:
            results.append(BatchImageResult(
                filename=fname,
                num_objects=0,
                detections=[],
                result_image_path="",
                error=str(e),
            ))
            failed += 1

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return BatchInferResponse(
        total_images=len(image_files),
        processed=processed,
        failed=failed,
        output_dir=req.output_dir,
        results=results,
        elapsed_ms=round(elapsed_ms, 2),
    )
