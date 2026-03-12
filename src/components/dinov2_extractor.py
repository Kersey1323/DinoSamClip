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
        
        # 【关键修复 1】：必须在模型加载时强行打通 output_attentions 属性
        self.model = AutoModel.from_pretrained(model_name, output_attentions=True)
        self.model.to(self.device)
        self.model.eval()

        print(f"DinoV2 loaded successfully on {self.device}")

    def extract_attention_map(self, image: Image.Image, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        使用 CLS token 与空间 Patch 的余弦相似度，生成更稳定、覆盖更全的热力图
        """
        import cv2
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        actual_h = inputs['pixel_values'].shape[2]
        actual_w = inputs['pixel_values'].shape[3]
        patch_size = 14
        h, w = actual_h // patch_size, actual_w // patch_size

        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # 提取最后一层的特征矩阵
        hidden_states = outputs.last_hidden_state[0]
        
        # CLS 是全局特征，Patch 是局部特征
        cls_token = hidden_states[0]
        patch_tokens = hidden_states[1:1 + h * w]

        # 归一化后进行点乘运算 (即计算余弦相似度)
        cls_token = torch.nn.functional.normalize(cls_token, dim=0)
        patch_tokens = torch.nn.functional.normalize(patch_tokens, dim=-1)
        similarity = (patch_tokens @ cls_token).cpu().numpy()

        # 变形并归一化到 0-1 之间
        attention_map = similarity.reshape(h, w)
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)

        if target_size is not None:
            attention_map = cv2.resize(attention_map, target_size, interpolation=cv2.INTER_LINEAR)

        return attention_map

    def extract_features(self, image: Image.Image) -> torch.Tensor:
        """Extract global feature vector from image"""
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
        """Generate point prompts from attention map"""
        h, w = attention_map.shape
        peaks = []
        for i in range(h):
            for j in range(w):
                if attention_map[i, j] > threshold:
                    peaks.append((j, i, attention_map[i, j]))

        peaks.sort(key=lambda x: x[2], reverse=True)
        selected_points = []
        min_distance = 30

        for x, y, score in peaks:
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

        return [(int(x), int(y)) for x, y, _ in selected_points]

    def get_attention_visualization(
        self,
        image: Image.Image,
        attention_map: Optional[np.ndarray] = None,
        colormap: str = "jet"
    ) -> np.ndarray:
        """Create visualization of attention map overlaid on image"""
        if attention_map is None:
            attention_map = self.extract_attention_map(image, target_size=image.size[::-1])

        img_array = np.array(image)
        attention_colored = cv2.applyColorMap(
            (attention_map * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        attention_colored = cv2.cvtColor(attention_colored, cv2.COLOR_BGR2RGB)

        alpha = 0.5
        blended = (alpha * attention_colored + (1 - alpha) * img_array).astype(np.uint8)
        return blended