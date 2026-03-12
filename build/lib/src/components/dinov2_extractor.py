"""
DinoV2 Feature Extractor Module
Extracts visual features and attention maps from images using DinoV2
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import cv2
from typing import Tuple, Optional
from src.config import ModelConfig

class DinoV2Extractor:
    """
    DinoV2 feature extractor for generating attention maps and visual features
    """

    def __init__(self, model_name: str = ModelConfig.DINOV2_MODEL_NAME, device: str = "cuda"):
        """
        Initialize DinoV2 extractor

        Args:
            model_name: HuggingFace model name for DinoV2
            device: Device to run the model on
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name

        print(f"Loading DinoV2 model: {model_name}")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        print(f"DinoV2 loaded successfully on {self.device}")

    def extract_attention_map(self, image: Image.Image, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Extract attention map from DinoV2

        Args:
            image: PIL Image
            target_size: Target size for the attention map (width, height)

        Returns:
            Attention map as numpy array
        """
        # Prepare inputs
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Extract features
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Get attention from last hidden state
            attention = outputs.last_hidden_state

        # Get CLS token attention (first token)
        cls_attention = attention[:, 0, 1:]  # Remove CLS token itself

        # Reshape to 2D grid
        # DinoV2-base uses 14x14 patches for 224x224 input
        h = w = int(np.sqrt(cls_attention.shape[1]))
        attention_map = cls_attention[0].reshape(h, w).cpu().numpy()

        # Upsample to target size
        if target_size is not None:
            attention_map = cv2.resize(
                attention_map,
                (target_size[1], target_size[0]),  # (width, height)
                interpolation=cv2.INTER_LINEAR
            )

        # Normalize to [0, 1]
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)

        return attention_map

    def extract_features(self, image: Image.Image) -> torch.Tensor:
        """
        Extract global feature vector from image

        Args:
            image: PIL Image

        Returns:
            Feature vector as tensor
        """
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state[:, 0]  # CLS token

        return features

    def generate_prompts_from_attention(
        self,
        attention_map: np.ndarray,
        num_points: int = 10,
        threshold: float = 0.5
    ) -> list:
        """
        Generate point prompts from attention map

        Args:
            attention_map: Attention map from DinoV2
            num_points: Number of points to generate
            threshold: Threshold for attention peaks

        Returns:
            List of (x, y) coordinates
        """
        h, w = attention_map.shape

        # Find peaks in attention map
        peaks = []
        for i in range(h):
            for j in range(w):
                if attention_map[i, j] > threshold:
                    peaks.append((j, i, attention_map[i, j]))  # (x, y, score)

        # Sort by attention score
        peaks.sort(key=lambda x: x[2], reverse=True)

        # Select top N points with non-maximum suppression
        selected_points = []
        min_distance = 30  # Minimum distance between points

        for x, y, score in peaks:
            # Check if this point is too close to already selected points
            too_close = False
            for px, py, _ in selected_points:
                dist = np.sqrt((x - px) ** 2 + (y - py) ** 2)
                if dist < min_distance:
                    too_close = True
                    break

            if not too_close:
                selected_points.append((x, y, score))
                if len(selected_points) >= num_points:
                    break

        # Return as list of (x, y) tuples
        return [(int(x), int(y)) for x, y, _ in selected_points]

    def get_attention_visualization(
        self,
        image: Image.Image,
        attention_map: Optional[np.ndarray] = None,
        colormap: str = "jet"
    ) -> np.ndarray:
        """
        Create visualization of attention map overlaid on image

        Args:
            image: Original PIL Image
            attention_map: Attention map (if None, will be computed)
            colormap: OpenCV colormap for visualization

        Returns:
            Visualization as numpy array (RGB)
        """
        if attention_map is None:
            attention_map = self.extract_attention_map(image, target_size=image.size[::-1])

        # Convert PIL image to numpy
        img_array = np.array(image)

        # Apply colormap to attention map
        attention_colored = cv2.applyColorMap(
            (attention_map * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        attention_colored = cv2.cvtColor(attention_colored, cv2.COLOR_BGR2RGB)

        # Blend with original image
        alpha = 0.5
        blended = (alpha * attention_colored + (1 - alpha) * img_array).astype(np.uint8)

        return blended
