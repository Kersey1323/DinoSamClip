"""
SAM Segmentation Module
Uses Segment Anything Model (SAM) for precise object segmentation
"""

import torch
import numpy as np
from PIL import Image
import cv2
from typing import List, Tuple, Optional, Dict
from segment_anything import sam_model_registry, SamPredictor
from src.config import ModelConfig
import matplotlib.pyplot as plt

import os

class SAMSegmenter:
    """
    SAM-based segmenter for generating object masks
    """

    def __init__(
        self,
        model_type: str = "vit_h",
        checkpoint_path: str = ModelConfig.SAM_CHECKPOINT_PATH,
        device: str = "cuda"
    ):
        """
        Initialize SAM segmenter

        Args:
            model_type: SAM model type (vit_b, vit_l, vit_h)
            checkpoint_path: Path to SAM checkpoint file
            device: Device to run the model on
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_type = model_type

        print(f"Loading SAM model: {model_type}")
        
        # Initialize model structure first (without checkpoint)
        self.sam = sam_model_registry[model_type](checkpoint=None)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                print(f"Loading checkpoint from {checkpoint_path}")
                # Load state dict
                state_dict = torch.load(checkpoint_path, map_location=self.device)
                
                # Handle nested state dict (e.g. {'model': state_dict})
                if isinstance(state_dict, dict) and "model" in state_dict:
                    print("Extracting state dict from 'model' key...")
                    state_dict = state_dict["model"]
                
                # Load weights with strict=False to ignore partial mismatches if necessary
                # But typically we want strict=True. The error suggests a structure mismatch.
                # Let's try loading it manually which is more robust than build_sam's loader
                self.sam.load_state_dict(state_dict, strict=True)
                print("SAM checkpoint loaded successfully")
                
            except Exception as e:
                print(f"Warning: Failed to load SAM checkpoint: {e}")
                print("Using SAM with random initialization")
        else:
            print(f"Warning: SAM checkpoint not found at {checkpoint_path}")
            print("SAM will be loaded from default location or require manual download")

        self.sam.to(self.device)
        self.sam.eval()

        self.predictor = SamPredictor(self.sam)
        print(f"SAM loaded successfully on {self.device}")

    def set_image(self, image: np.ndarray):
        """
        Set image for segmentation

        Args:
            image: Image as numpy array (RGB)
        """
        self.predictor.set_image(image)

    def segment_from_points(
        self,
        points: List[Tuple[int, int]],
        labels: Optional[List[int]] = None,
        multimask_output: bool = True
    ) -> List[Dict]:
        """
        Segment image from point prompts

        Args:
            points: List of (x, y) coordinates
            labels: List of labels (1 = foreground, 0 = background)
            multimask_output: Whether to output multiple masks

        Returns:
            List of dictionaries containing masks, scores, and logits
        """
        if labels is None:
            labels = [1] * len(points)  # All points are foreground by default

        # Convert to numpy arrays
        point_coords = np.array(points)
        point_labels = np.array(labels)

        # Predict masks
        with torch.no_grad():
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=multimask_output
            )

        # Convert to list of dictionaries
        results = []
        for i in range(len(masks)):
            results.append({
                "mask": masks[i],
                "score": scores[i],
                "logit": logits[i]
            })

        return results

    def segment_from_box(
        self,
        box: Tuple[int, int, int, int],
        multimask_output: bool = True
    ) -> List[Dict]:
        """
        Segment image from bounding box prompt

        Args:
            box: Bounding box as (x1, y1, x2, y2)
            multimask_output: Whether to output multiple masks

        Returns:
            List of dictionaries containing masks, scores, and logits
        """
        # Convert box to the format SAM expects
        box_np = np.array([box])

        with torch.no_grad():
            masks, scores, logits = self.predictor.predict(
                point_coords=None,
                box=box_np,
                multimask_output=multimask_output
            )

        results = []
        for i in range(len(masks)):
            results.append({
                "mask": masks[i],
                "score": scores[i],
                "logit": logits[i]
            })

        return results

    def generate_grid_prompts(
        self,
        image_shape: Tuple[int, int],
        grid_size: int = 8
    ) -> List[Tuple[int, int]]:
        """
        Generate grid-based point prompts for automatic segmentation

        Args:
            image_shape: Image shape as (height, width)
            grid_size: Number of points along each dimension

        Returns:
            List of (x, y) coordinates
        """
        h, w = image_shape
        points = []

        step_h = h // (grid_size + 1)
        step_w = w // (grid_size + 1)

        for i in range(1, grid_size + 1):
            for j in range(1, grid_size + 1):
                x = j * step_w
                y = i * step_h
                points.append((x, y))

        return points

    def segment_automatic(
        self,
        image: np.ndarray,
        num_points: int = 64,
        min_mask_area: int = 1000
    ) -> List[Dict]:
        """
        Automatic segmentation using grid prompts

        Args:
            image: Image as numpy array (RGB)
            num_points: Number of grid points to use
            min_mask_area: Minimum area for valid mask

        Returns:
            List of valid mask dictionaries
        """
        self.set_image(image)

        # Generate grid prompts
        h, w = image.shape[:2]
        grid_size = int(np.sqrt(num_points))
        points = self.generate_grid_prompts((h, w), grid_size)

        # Segment from all points
        all_masks = []
        batch_size = 32  # Process in batches

        for i in range(0, len(points), batch_size):
            batch_points = points[i:i + batch_size]
            batch_labels = [1] * len(batch_points)

            with torch.no_grad():
                masks, scores, logits = self.predictor.predict(
                    point_coords=np.array(batch_points),
                    point_labels=np.array(batch_labels),
                    multimask_output=False
                )

            for j in range(len(masks)):
                mask = masks[j]
                # Filter by area
                if np.sum(mask) >= min_mask_area:
                    all_masks.append({
                        "mask": mask,
                        "score": scores[j],
                        "logit": logits[j]
                    })

        # Remove overlapping masks using NMS-like approach
        filtered_masks = self.filter_overlapping_masks(all_masks)

        return filtered_masks

    def filter_overlapping_masks(
        self,
        masks: List[Dict],
        iou_threshold: float = 0.8
    ) -> List[Dict]:
        """
        Filter overlapping masks

        Args:
            masks: List of mask dictionaries
            iou_threshold: IoU threshold for filtering

        Returns:
            Filtered list of masks
        """
        if len(masks) == 0:
            return []

        # Sort by score
        masks.sort(key=lambda x: x["score"], reverse=True)

        filtered = []
        for mask_dict in masks:
            mask = mask_dict["mask"]
            is_valid = True

            for kept_mask in filtered:
                # Calculate IoU
                intersection = np.logical_and(mask, kept_mask["mask"]).sum()
                union = np.logical_or(mask, kept_mask["mask"]).sum()
                iou = intersection / (union + 1e-8)

                if iou > iou_threshold:
                    is_valid = False
                    break

            if is_valid:
                filtered.append(mask_dict)

        return filtered

    def apply_mask_to_image(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        background_color: Tuple[int, int, int] = (0, 0, 0)
    ) -> np.ndarray:
        """
        Apply mask to image, setting background to specified color

        Args:
            image: Original image as numpy array
            mask: Binary mask
            background_color: RGB color for background

        Returns:
            Masked image
        """
        masked_image = image.copy()
        for c in range(3):
            masked_image[:, :, c] = np.where(
                mask,
                masked_image[:, :, c],
                background_color[c]
            )
        return masked_image

    def crop_masked_region(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        padding: int = 10
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Crop the region containing the mask

        Args:
            image: Original image
            mask: Binary mask
            padding: Padding around the bounding box

        Returns:
            Tuple of (cropped_image, bounding_box)
        """
        # Find bounding box
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        # Add padding
        h, w = image.shape[:2]
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        # Crop
        cropped = image[y_min:y_max, x_min:x_max]
        bbox = (x_min, y_min, x_max, y_max)

        return cropped, bbox

    
    # ==========================================================
    # 官方风格可视化工具库 (Official Visualization Utils)
    # ==========================================================
    @staticmethod
    def show_mask(mask: np.ndarray, ax, random_color: bool = False):
        """在图表轴上叠加绘制官方科技蓝(或随机颜色)的半透明 Mask"""
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    @staticmethod
    def show_points(coords: list, labels: list, ax, marker_size: int = 375):
        """在图表轴上绘制绿色(正样本)或红色(负样本)的提示点"""
        if not coords or not labels:
            return
        coords_np = np.array(coords)
        labels_np = np.array(labels)
        pos_points = coords_np[labels_np == 1]
        neg_points = coords_np[labels_np == 0]
        if len(pos_points) > 0:
            ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', 
                       s=marker_size, edgecolor='white', linewidth=1.25)
        if len(neg_points) > 0:
            ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', 
                       s=marker_size, edgecolor='white', linewidth=1.25)

    @staticmethod
    def show_box(box: tuple, ax):
        """在图表轴上绘制绿色的边界框"""
        if box is None:
            return
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

    def visualize_prediction(
        self, 
        image: np.ndarray, 
        masks: list, 
        points: list = None, 
        labels: list = None, 
        box: tuple = None, 
        title: str = "SAM Segmentation", 
        save_path: str = None
    ) -> np.ndarray:
        """
        集成可视化方法：一键生成包含原图、Mask、点和框的官方风格结果图。
        
        Args:
            image: 原始 RGB 图像数组
            masks: 掩码字典列表 (如 segment_from_points 返回的格式)
            points: 提示点列表 [(x1, y1), (x2, y2), ...]
            labels: 标签列表 [1, 1, 0, ...] (1为前景，0为背景)
            box: 边界框元组 (x1, y1, x2, y2)
            title: 图表标题
            save_path: 可选的保存路径
            
        Returns:
            np.ndarray: 生成的可视化结果的 RGB 图像数组
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(image)

        # 1. 绘制掩码 (如果传入了多个 mask，则使用随机颜色区分)
        use_random_color = len(masks) > 1
        for mask_dict in masks:
            mask = mask_dict["mask"]
            self.show_mask(mask, ax, random_color=use_random_color)

        # 2. 绘制提示点
        if points is not None:
            if labels is None:
                labels = [1] * len(points)  # 默认全是前景点
            self.show_points(points, labels, ax)

        # 3. 绘制边界框
        if box is not None:
            self.show_box(box, ax)

        ax.set_title(title, fontsize=16)
        ax.axis('off')

        # 4. 如果提供了路径，则保存到文件
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"SAM 独立可视化图已保存至: {save_path}")

        # 5. 将 matplotlib 的图表转换为 numpy 数组返回，方便在内存中流转
        fig.canvas.draw()
        img_result = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        img_result = img_result.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return img_result