"""
DinoV2 Feature Extractor Module
Extracts visual features and attention maps from images using DinoV2.

Fine-tuned weight loading:
  The training script (training/classifier/train_classifier.py) saves an
  HFDinoVisionClassifier whose state_dict contains both backbone AND classifier
  head keys.  We extract only the backbone keys (prefixed with "backbone.")
  and load them into the AutoModel, ignoring the classification head.
"""

import os
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import cv2
from typing import Tuple, Optional

from settings import ModelConfig


class DinoV2Extractor:
    """
    DinoV2 feature extractor for generating attention maps and visual features.
    Automatically loads fine-tuned backbone weights when
    ModelConfig.DINOV2_FINETUNED_CHECKPOINT is set and the file exists.
    """

    def __init__(
        self,
        model_name: str = ModelConfig.DINOV2_MODEL_NAME,
        device: str = "cuda",
        finetuned_checkpoint: Optional[str] = ModelConfig.DINOV2_FINETUNED_CHECKPOINT,
    ):
        """
        Initialize DinoV2 extractor.

        Args:
            model_name: HuggingFace model path for the base DINOv2.
            device: Device to run the model on ('cuda' / 'cpu').
            finetuned_checkpoint: Path to a fine-tuned .pth saved by
                HFDinoVisionClassifier.  The classifier head keys are
                automatically stripped; only backbone weights are loaded.
                Pass None or '' to skip fine-tuned loading.
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name

        print(f"Loading DINOv2 base model: {model_name}")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_attentions=True)
        self.model.to(self.device)
        self.model.eval()

        # ── Load fine-tuned backbone weights (optional) ───────────────────────
        if finetuned_checkpoint and os.path.isfile(finetuned_checkpoint):
            self._load_finetuned_backbone(finetuned_checkpoint)
        elif finetuned_checkpoint:
            print(
                f"[DINOv2] Warning: fine-tuned checkpoint not found at:\n"
                f"  {finetuned_checkpoint}\n"
                f"  → Using base pre-trained weights instead.\n"
                f"  → Place the .pth file there and restart to enable fine-tuned mode."
            )
        else:
            print("[DINOv2] No fine-tuned checkpoint configured → using base pre-trained weights.")

        print(f"DINOv2 loaded successfully on {self.device}")

    def _load_finetuned_backbone(self, checkpoint_path: str):
        """
        Load backbone weights from an HFDinoVisionClassifier checkpoint.

        The saved state_dict contains two kinds of keys:
          - 'backbone.<layer>'  → DINOv2 backbone (what we want)
          - 'classifier.<layer>'→ classification head  (we discard)

        We strip the 'backbone.' prefix and load matching keys only.
        """
        print(f"[DINOv2] Loading fine-tuned backbone from: {checkpoint_path}")
        try:
            full_state = torch.load(checkpoint_path, map_location=self.device)

            # Extract only backbone keys, strip prefix
            backbone_state = {
                k[len("backbone."):]: v
                for k, v in full_state.items()
                if k.startswith("backbone.")
            }

            if not backbone_state:
                # Checkpoint may already be backbone-only (no prefix)
                backbone_state = full_state
                print("[DINOv2] No 'backbone.' prefix found — treating checkpoint as backbone-only.")

            missing, unexpected = self.model.load_state_dict(backbone_state, strict=False)

            if missing:
                print(f"[DINOv2] Missing keys (expected, classifier head keys absent): {len(missing)}")
            if unexpected:
                print(f"[DINOv2] Unexpected keys: {unexpected[:5]}")

            self.model.eval()
            print("[DINOv2] ✅ Fine-tuned backbone weights loaded successfully!")

        except Exception as e:
            print(f"[DINOv2] ❌ Failed to load fine-tuned weights: {e}")
            print("[DINOv2]    Falling back to base pre-trained weights.")

    # ── Feature extraction ────────────────────────────────────────────────────

    def extract_attention_map(
        self, image: Image.Image, target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Generate a CLS-patch cosine-similarity heatmap (attention map).
        """
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        actual_h = inputs["pixel_values"].shape[2]
        actual_w = inputs["pixel_values"].shape[3]
        patch_size = 14
        h, w = actual_h // patch_size, actual_w // patch_size

        with torch.no_grad():
            outputs = self.model(**inputs)

        hidden_states = outputs.last_hidden_state[0]
        cls_token = hidden_states[0]
        patch_tokens = hidden_states[1 : 1 + h * w]

        cls_token = torch.nn.functional.normalize(cls_token, dim=0)
        patch_tokens = torch.nn.functional.normalize(patch_tokens, dim=-1)
        similarity = (patch_tokens @ cls_token).cpu().numpy()

        attention_map = similarity.reshape(h, w)
        attention_map = (attention_map - attention_map.min()) / (
            attention_map.max() - attention_map.min() + 1e-8
        )

        if target_size is not None:
            attention_map = cv2.resize(attention_map, target_size, interpolation=cv2.INTER_LINEAR)

        return attention_map

    def extract_features(self, image: Image.Image) -> torch.Tensor:
        """Extract global CLS-token feature vector from image."""
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
        threshold: float = 0.5,
    ) -> list:
        """Generate point prompts from attention map peaks."""
        h, w = attention_map.shape
        peaks = [
            (j, i, attention_map[i, j])
            for i in range(h)
            for j in range(w)
            if attention_map[i, j] > threshold
        ]
        peaks.sort(key=lambda x: x[2], reverse=True)

        selected_points = []
        min_distance = 30

        for x, y, score in peaks:
            too_close = any(
                np.sqrt((x - px) ** 2 + (y - py) ** 2) < min_distance
                for px, py, _ in selected_points
            )
            if not too_close:
                selected_points.append((x, y, score))
                if len(selected_points) >= num_points:
                    break

        return [(int(x), int(y)) for x, y, _ in selected_points]

    def get_attention_visualization(
        self,
        image: Image.Image,
        attention_map: Optional[np.ndarray] = None,
        colormap: str = "jet",
    ) -> np.ndarray:
        """Create visualization of attention map overlaid on image."""
        if attention_map is None:
            attention_map = self.extract_attention_map(image, target_size=image.size[::-1])

        img_array = np.array(image)
        attention_colored = cv2.applyColorMap(
            (attention_map * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        attention_colored = cv2.cvtColor(attention_colored, cv2.COLOR_BGR2RGB)

        alpha = 0.5
        blended = (alpha * attention_colored + (1 - alpha) * img_array).astype(np.uint8)
        return blended