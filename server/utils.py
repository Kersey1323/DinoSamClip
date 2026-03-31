"""
Helper utilities shared across API routes:
  - base64 ↔ PIL/numpy conversion
  - visualization (overlay masks on image and return as base64 PNG)
"""

from __future__ import annotations

import base64
import io
import time
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

# ─── Colour palette for mask overlay ─────────────────────────────────────────
COLORS = [
    (255, 0,   0),    # Red
    (0,   255, 0),    # Green
    (0,   100, 255),  # Blue
    (255, 255, 0),    # Yellow
    (255, 0,   255),  # Magenta
    (0,   255, 255),  # Cyan
    (255, 128, 0),    # Orange
]


def b64_to_pil(b64: str) -> Image.Image:
    """Decode a base64 string to a PIL RGBA→RGB image."""
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")


def pil_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL image to a base64 string."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def numpy_to_b64(arr: np.ndarray, fmt: str = "PNG") -> str:
    """Encode an RGB numpy array to a base64 string."""
    img = Image.fromarray(arr.astype(np.uint8))
    return pil_to_b64(img, fmt=fmt)


def create_overlay(image_array: np.ndarray, detections: list, alpha: float = 0.45) -> np.ndarray:
    """
    Draw semi-transparent masks + contours + labels onto *image_array*.
    Returns a new RGB uint8 ndarray.
    """
    output = image_array.copy()
    overlay = image_array.copy()

    for i, det in enumerate(detections):
        mask = det["mask"]
        color = COLORS[i % len(COLORS)]
        overlay[mask] = np.array(color, dtype=np.uint8)

    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    for i, det in enumerate(detections):
        mask = det["mask"]
        color = COLORS[i % len(COLORS)]
        mask_u8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(output, contours, -1, color, thickness=3)
        cv2.drawContours(output, contours, -1, (255, 255, 255), thickness=1)

        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            label = f"{i+1}. {det['class']} ({det['confidence']:.1%})"
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(output, (x, y - th - bl - 5), (x + tw, y), color, -1)
            text_color = (0, 0, 0) if (color[0]*0.299 + color[1]*0.587 + color[2]*0.114) > 150 else (255, 255, 255)
            cv2.putText(output, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)

    return output


def detections_to_dicts(detections: list) -> list:
    """Convert raw pipeline detections to JSON-serialisable dicts."""
    result = []
    for det in detections:
        bbox = det.get("bbox", [])
        # bbox may be tuple or list; normalise to list[int]
        if bbox is not None:
            bbox = [int(v) for v in bbox]
        result.append({
            "class": det["class"],
            "confidence": float(det["confidence"]),
            "bbox": bbox,
            "all_predictions": [
                {"class": p["class"], "confidence": float(p["confidence"])}
                for p in det.get("all_predictions", [])
            ] or None,
        })
    return result
