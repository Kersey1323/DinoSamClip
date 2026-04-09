"""
SigLIP 2 特征提取器
"""
import torch
import numpy as np
from PIL import Image
from typing import List, Union
from transformers import AutoProcessor, AutoModel

from vector_db.utils.logger import setup_logger

logger = setup_logger(__name__)


class SigLIPExtractor:
    """SigLIP 2 特征提取器"""

    def __init__(self, model_path: str, device: str = "cuda"):
        """
        初始化 SigLIP 2 模型

        Args:
            model_path: 模型路径
            device: 设备 ('cuda' 或 'cpu')
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading SigLIP model from: {model_path}")

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"SigLIP model loaded successfully on {self.device}")

    def extract_image_features(self, image: Image.Image) -> np.ndarray:
        """
        提取图像特征向量

        Args:
            image: PIL Image 对象

        Returns:
            归一化的图像特征向量 (1152,)
        """
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        # L2 归一化
        features = image_features.cpu().numpy()[0]
        features = features / np.linalg.norm(features)

        return features

    def extract_text_features(self, text: str) -> np.ndarray:
        """
        提取文本特征向量

        Args:
            text: 文本描述

        Returns:
            归一化的文本特征向量 (1152,)
        """
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)

        # L2 归一化
        features = text_features.cpu().numpy()[0]
        features = features / np.linalg.norm(features)

        return features

    def extract_batch_images(self, images: List[Image.Image]) -> np.ndarray:
        """
        批量提取图像特征

        Args:
            images: PIL Image 对象列表

        Returns:
            归一化的图像特征矩阵 (N, 1152)
        """
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        # L2 归一化
        features = image_features.cpu().numpy()
        features = features / np.linalg.norm(features, axis=1, keepdims=True)

        return features
