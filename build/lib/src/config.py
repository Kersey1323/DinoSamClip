"""
Configuration file for DinoV2 + SAM + CLIP Pipeline
"""

MODEL_SOURCES = "/media/dell/新加卷1/LLM/models/CV-models"

# Model configurations
class ModelConfig:
    # DinoV2 model settings
    DINOV2_MODEL_NAME = f"{MODEL_SOURCES}/dinov2-large"
    DINOV2_DEVICE = "cuda"  # or "cpu"

    # SAM model settings
    SAM_MODEL_TYPE = "vit_h"  # vit_b, vit_l, vit_h
    SAM_CHECKPOINT_PATH = f"{MODEL_SOURCES}/sam3/sam3.pt"
    SAM_DEVICE = "cuda"  # or "cpu"

    # CLIP model settings
    CLIP_MODEL_NAME = f"{MODEL_SOURCES}/clip-vit-base-patch32"
    CLIP_DEVICE = "cuda"  # or "cpu"

# Pipeline settings
class PipelineConfig:
    
    SAM_MODEL_TYPE = "vit_h"  # vit_b, vit_l, vit_h
    
    # Attention threshold for DinoV2 prompt generation
    ATTENTION_THRESHOLD = 0.5

    # Minimum area for valid mask (in pixels)
    MIN_MASK_AREA = 1000

    # Number of points to generate from attention map
    NUM_PROMPTS = 10

    # CLIP classification confidence threshold
    CONFIDENCE_THRESHOLD = 0.2

    # Image preprocessing
    IMAGE_SIZE = 224  # CLIP input size
    DINOV2_IMAGE_SIZE = 224

    # Device settings
    DEVICE = "cuda"  # or "cpu"

# Default candidate classes for zero-shot classification
DEFAULT_CANDIDATE_CLASSES = [
    "person", "car", "dog", "cat", "bird", "horse", "cow", "sheep",
    "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "toothbrush", "basket", "ball", "kite",
    "skateboard", "surfboard", "tennis racket", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"
]
