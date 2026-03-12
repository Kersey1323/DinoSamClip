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


class DinoSAMClipPipeline:
    """
    Complete pipeline integrating DinoV2, SAM, and CLIP
    """

    def __init__(
        self,
        device: str = "cuda",
        candidate_classes: Optional[List[str]] = None,
        sam_checkpoint: str = "sam3.pth"
    ):
        """
        Initialize the complete pipeline

        Args:
            device: Device to run all models on
            candidate_classes: List of candidate class names for CLIP
            sam_checkpoint: Path to SAM checkpoint file
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
        confidence_threshold: float = 0.2
        
    ) -> Dict:
        """
        Main detection and classification method

        Args:
            image: Input PIL Image
            candidate_classes: Optional list of candidate classes (overrides default)
            use_dinov2_prompts: Whether to use DinoV2 attention for prompting SAM
            num_prompts: Number of prompts to generate from DinoV2
            attention_threshold: Threshold for DinoV2 attention peaks
            min_mask_area: Minimum area for valid masks
            confidence_threshold: Minimum confidence for classification

        Returns:
            Dictionary containing detection results
        """
        # Use provided classes or default
        classes = candidate_classes or self.candidate_classes

        # Convert image to numpy for processing
        image_array = np.array(image)

        print("\n" + "=" * 60)
        print("Step 1: Extracting DinoV2 attention maps...")
        print("=" * 60)

        # Step 1: DinoV2 extracts attention map
        attention_map = self.dinov2.extract_attention_map(
            image,
            target_size=(image.width, image.height)
        )
        print(f"Attention map shape: {attention_map.shape}")

        print("\n" + "=" * 60)
        print("Step 2: Generating prompts for SAM...")
        print("=" * 60)

        # Step 2: Generate prompts from attention map
        if use_dinov2_prompts:
            prompts = self.dinov2.generate_prompts_from_attention(
                attention_map,
                num_points=num_prompts,
                threshold=attention_threshold
            )
            print(f"Generated {len(prompts)} prompts from DinoV2 attention")
        else:
            # Use grid prompts as fallback
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

        # Step 3: SAM generates segmentation masks
        self.sam.set_image(image_array)

        # Segment from prompts
        all_masks = []
        for point in prompts:
            results = self.sam.segment_from_points([point], multimask_output=False)
            all_masks.extend(results)

        # Filter overlapping masks
        valid_masks = self.sam.filter_overlapping_masks(
            all_masks,
            iou_threshold=0.7
        )

        # Filter by minimum area
        valid_masks = [m for m in valid_masks if np.sum(m["mask"]) >= min_mask_area]

        print(f"Found {len(valid_masks)} valid masks")

        if len(valid_masks) == 0:
            print("No objects detected!")
            return {
                "image": image,
                "detections": [],
                "num_objects": 0
            }

        print("\n" + "=" * 60)
        print("Step 4: Classifying objects with CLIP...")
        print("=" * 60)

        # Step 4: CLIP classifies each mask
        cropped_images = []
        mask_data = []

        for i, mask_dict in enumerate(valid_masks):
            mask = mask_dict["mask"]

            # Crop masked region
            cropped, bbox = self.sam.crop_masked_region(image_array, mask, padding=10)

            # Apply mask to cropped region
            cropped_masked = self.sam.apply_mask_to_image(cropped, mask[bbox[1]:bbox[3], bbox[0]:bbox[2]])

            # Resize to CLIP input size
            cropped_resized = cv2.resize(cropped_masked, (224, 224))

            cropped_images.append(cropped_resized)
            mask_data.append({
                "bbox": bbox,
                "mask": mask,
                "score": mask_dict["score"]
            })

        # Classify all crops
        clip_results = self.clip.classify_masks(cropped_images, classes)

        # Combine results
        detections = []
        for i, (mask_info, clip_result) in enumerate(zip(mask_data, clip_results)):
            # Filter by confidence threshold
            if clip_result["top_confidence"] >= confidence_threshold:
                detections.append({
                    "class": clip_result["top_class"],
                    "confidence": clip_result["top_confidence"],
                    "bbox": mask_info["bbox"],
                    "mask": mask_info["mask"],
                    "all_predictions": clip_result["predictions"]
                })

        print(f"\nClassified {len(detections)} objects")

        # Print results
        for i, det in enumerate(detections):
            print(f"  Object {i+1}: {det['class']} ({det['confidence']:.2%})")

        return {
            "image": image,
            "attention_map": attention_map,
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

        Args:
            image: Input PIL Image
            candidate_classes: Optional list of candidate classes
            num_points: Number of grid points for SAM
            min_mask_area: Minimum area for valid masks
            confidence_threshold: Minimum confidence for classification

        Returns:
            Dictionary containing detection results
        """
        classes = candidate_classes or self.candidate_classes
        image_array = np.array(image)

        print("\n" + "=" * 60)
        print("Running automatic detection (SAM + CLIP)...")
        print("=" * 60)

        # Direct SAM segmentation
        valid_masks = self.sam.segment_automatic(
            image_array,
            num_points=num_points,
            min_mask_area=min_mask_area
        )

        print(f"Found {len(valid_masks)} valid masks")

        if len(valid_masks) == 0:
            return {
                "image": image,
                "detections": [],
                "num_objects": 0
            }

        # Classify with CLIP
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

        # Combine results
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

        return {
            "image": image,
            # "attention_map": attention_map,
            # "prompts": prompts,           # <--- 【新增这一行】把生成的提示点传出来
            "detections": detections,
            "num_objects": len(detections),
            "all_masks": valid_masks      # 原本就有，这是 SAM 输出的全部原始 Mask
        }
