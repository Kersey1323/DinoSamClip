"""
CLIP Classification Module
Uses standard Hugging Face Transformers CLIP for zero-shot image classification
"""

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from typing import List, Dict
import sys
import os

from src.config import ModelConfig
class CLIPClassifier:
    """
    CLIP-based zero-shot classifier using standard Transformers API
    """

    def __init__(
        self,
        model_name: str = ModelConfig.CLIP_MODEL_NAME,
        device: str = "cuda"
    ):
        """
        Initialize CLIP classifier
        
        Args:
            model_name: Local path to the CLIP model folder
            device: Device to run the model on
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        print(f"Loading standard CLIP model from: {model_name}")
        # 加载标准的 CLIP 模型
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        
        print(f"CLIP loaded successfully on {self.device}")

    def classify_masks(
        self,
        images: List[np.ndarray],
        candidate_classes: List[str]
    ) -> List[Dict]:
        """
        Classify masked image regions
        
        Args:
            images: List of masked images (numpy arrays from SAM output)
            candidate_classes: List of candidate class names (MUST BE ENGLISH)
            
        Returns:
            List of classification results for each mask
        """
        if not images or not candidate_classes:
            return []

        # 1. 格式转换：SAM 传过来的是 numpy array (cv2格式)，转为 PIL Image
        # 这一步非常重要，processor 更擅长处理 PIL Image
        pil_images = [Image.fromarray(img) if isinstance(img, np.ndarray) else img for img in images]

        # 2. 使用 Processor 统一处理图像和文本
        # 它会自动帮你做 Resize(224x224), Normalize, 以及 Tokenize
        inputs = self.processor(
            text=candidate_classes,
            images=pil_images,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # 3. 模型推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 获取 图像-文本 相似度矩阵
            logits_per_image = outputs.logits_per_image
            # 计算每张图片在各个类别上的 softmax 概率分布
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        # 4. 封装结果返回给 pipeline.py
        results = []
        for i in range(len(pil_images)):
            current_probs = probs[i]
            # 按概率从大到小排序的索引
            sorted_indices = np.argsort(current_probs)[::-1]
            
            predictions = []
            for idx in sorted_indices:
                predictions.append({
                    "class": candidate_classes[idx],
                    "confidence": float(current_probs[idx])
                })
            
            # 记录当前 mask 的完整预测信息和 Top-1 结果
            results.append({
                "predictions": predictions,
                "top_class": predictions[0]["class"],
                "top_confidence": predictions[0]["confidence"]
            })

        return results