"""
Main Pipeline: DinoV2 + SAM + CLIP
Integrates all three modules for complete object detection and classification
"""

from matplotlib import image
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Optional
import cv2

from components.dinov2_extractor import DinoV2Extractor
from components.sam_segmenter import SAMSegmenter
from components.clip_classifier import CLIPClassifier
from config import PipelineConfig, DEFAULT_CANDIDATE_CLASSES, ModelConfig

import os

# Set default output directory to src/results relative to this file
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

class DinoSAMClipPipeline:
    """
    Complete pipeline integrating DinoV2, SAM, and CLIP
    """

    def __init__(
        self,
        device: str = "cuda",
        candidate_classes: Optional[List[str]] = None,
        sam_checkpoint: str = ModelConfig.SAM_CHECKPOINT_PATH
    ):
        """
        Initialize the complete pipeline
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.config = PipelineConfig()
        self.candidate_classes = candidate_classes or DEFAULT_CANDIDATE_CLASSES

        print("=" * 60)
        print("Initializing DinoV2 + SAM + CLIP Pipeline")
        print("=" * 60)

        # Initialize all three models
        print("\n[1/3] Loading DinoV2...")
        self.dinov2 = DinoV2Extractor(device=self.device)

        print("\n[2/3] Loading SAM...")
        self.sam = SAMSegmenter(
            model_type=self.config.SAM_MODEL_TYPE,
            checkpoint_path=sam_checkpoint,
            device=self.device
        )

        print("\n[3/3] Loading CLIP...")
        self.clip = CLIPClassifier(device=self.device)

        print("\n" + "=" * 60)
        print("Pipeline initialized successfully!")
        print("=" * 60)

    def detect_and_classify(
        self,
        image: Image.Image,
        candidate_classes: Optional[List[str]] = None,
        use_dinov2_prompts: bool = True,
        num_prompts: int = 10,
        attention_threshold: float = 0.5,
        min_mask_area: int = 1000,
        confidence_threshold: float = 0.2,
        output_dir: str = OUTPUT_DIR
    ) -> Dict:
        """
        Main detection and classification method

        Args:
            output_dir: Directory to save debug images
        """
        classes = candidate_classes or self.candidate_classes
        image_array = np.array(image)

        print("\n" + "=" * 60)
        print("Step 1: Extracting DinoV2 attention maps...")
        print("=" * 60)

        attention_map = self.dinov2.extract_attention_map(
            image,
            target_size=(image.width, image.height)
        )
        print(f"Attention map shape: {attention_map.shape}")

        print("\n" + "=" * 60)
        print("Step 2: Generating prompts for SAM...")
        print("=" * 60)
        
        # 提前定义 prompts 变量，防止作用域问题
        prompts = []
        
        if use_dinov2_prompts:
            prompts = self.dinov2.generate_prompts_from_attention(
                attention_map,
                num_points=num_prompts,
                threshold=attention_threshold
            )
            print(f"Generated {len(prompts)} prompts from DinoV2 attention")
        else:
            prompts = self.sam.generate_grid_prompts(
                image_array.shape[:2],
                grid_size=8
            )
            print(f"Using {len(prompts)} grid prompts")

        if len(prompts) == 0:
            print("Warning: No prompts generated, using grid prompts")
            prompts = self.sam.generate_grid_prompts(image_array.shape[:2], grid_size=4)

        print("\n" + "=" * 60)
        print("Step 3: Segmenting objects with SAM...")
        print("=" * 60)

        self.sam.set_image(image_array)

        all_masks = []
        for point in prompts:
            results = self.sam.segment_from_points([point], multimask_output=False)
            all_masks.extend(results)

        valid_masks = self.sam.filter_overlapping_masks(
            all_masks,
            iou_threshold=0.7
        )

        valid_masks = [m for m in valid_masks if np.sum(m["mask"]) >= min_mask_area]

        print(f"Found {len(valid_masks)} valid masks")
        
        # 可视化排错
        if len(valid_masks) > 0:
            self.sam.visualize_prediction(
                image=image_array,
                masks=valid_masks,
                points=prompts,
                title="Pipeline Debug: SAM Valid Masks",
                save_path=f"{output_dir}/debug_pipeline_sam.png" 
            )

        # 失败提前退出时，也要保证字典结构完整！
        if len(valid_masks) == 0:
            print("No objects detected!")
            return {
                "image": image,
                "attention_map": attention_map,
                "prompts": prompts,
                "detections": [],
                "num_objects": 0,
                "all_masks": []
            }

        print("\n" + "=" * 60)
        print("Step 4: Classifying objects with CLIP...")
        print("=" * 60)

        cropped_images = []
        mask_data = []

        for i, mask_dict in enumerate(valid_masks):
            mask = mask_dict["mask"]
            cropped, bbox = self.sam.crop_masked_region(image_array, mask, padding=10)
            cropped_masked = self.sam.apply_mask_to_image(cropped, mask[bbox[1]:bbox[3], bbox[0]:bbox[2]])
            cropped_resized = cv2.resize(cropped_masked, (224, 224))

            cropped_images.append(cropped_resized)
            mask_data.append({
                "bbox": bbox,
                "mask": mask,
                "score": mask_dict["score"]
            })

        clip_results = self.clip.classify_masks(cropped_images, classes)

        detections = []
        for i, (mask_info, clip_result) in enumerate(zip(mask_data, clip_results)):
            if clip_result["top_confidence"] >= confidence_threshold:
                detections.append({
                    "class": clip_result["top_class"],
                    "confidence": clip_result["top_confidence"],
                    "bbox": mask_info["bbox"],
                    "mask": mask_info["mask"],
                    "all_predictions": clip_result["predictions"]
                })

        print(f"\nClassified {len(detections)} objects")
        for i, det in enumerate(detections):
            print(f"  Object {i+1}: {det['class']} ({det['confidence']:.2%})")

        # 完美标准的返回字典
        return {
            "image": image,
            "attention_map": attention_map,
            "prompts": prompts,              # <--- 这里传入第 102 行定义的 prompts
            "detections": detections,
            "num_objects": len(detections),
            "all_masks": valid_masks
        }

    def detect_and_classify_automatic(
        self,
        image: Image.Image,
        candidate_classes: Optional[List[str]] = None,
        num_points: int = 64,
        min_mask_area: int = 1000,
        confidence_threshold: float = 0.2
    ) -> Dict:
        """
        Automatic detection using SAM grid prompts (no DinoV2)
        """
        classes = candidate_classes or self.candidate_classes
        image_array = np.array(image)

        print("\n" + "=" * 60)
        print("Running automatic detection (SAM + CLIP)...")
        print("=" * 60)

        valid_masks = self.sam.segment_automatic(
            image_array,
            num_points=num_points,
            min_mask_area=min_mask_area
        )

        print(f"Found {len(valid_masks)} valid masks")

        # 失败提前退出时，同样保证字典结构完整！
        if len(valid_masks) == 0:
            return {
                "image": image,
                "attention_map": None,       # 自动模式没有热力图，传 None
                "prompts": [],               # 自动模式没有手动点，传空列表 []
                "detections": [],
                "num_objects": 0,
                "all_masks": []
            }

        cropped_images = []
        mask_data = []

        for mask_dict in valid_masks:
            mask = mask_dict["mask"]
            cropped, bbox = self.sam.crop_masked_region(image_array, mask, padding=10)
            cropped_masked = self.sam.apply_mask_to_image(cropped, mask[bbox[1]:bbox[3], bbox[0]:bbox[2]])
            cropped_resized = cv2.resize(cropped_masked, (224, 224))

            cropped_images.append(cropped_resized)
            mask_data.append({
                "bbox": bbox,
                "mask": mask,
                "score": mask_dict["score"]
            })

        clip_results = self.clip.classify_masks(cropped_images, classes)

        detections = []
        for mask_info, clip_result in zip(mask_data, clip_results):
            if clip_result["top_confidence"] >= confidence_threshold:
                detections.append({
                    "class": clip_result["top_class"],
                    "confidence": clip_result["top_confidence"],
                    "bbox": mask_info["bbox"],
                    "mask": mask_info["mask"],
                    "all_predictions": clip_result["predictions"]
                })

        # 完美标准的返回字典 (与上面完全一致)
        return {
            "image": image,
            "attention_map": None,           # <--- 自动模式没有热力图，传 None
            "prompts": [],                   # <--- 自动模式没有手动点，传空列表 []
            "detections": detections,
            "num_objects": len(detections),
            "all_masks": valid_masks
        }