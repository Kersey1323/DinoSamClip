"""
Global configuration loader for DinoSamClip.
Reads settings from config/<APP_ENV>.ini  (default: local.ini)

Switch environment:
    APP_ENV=dev python ...        # loads config/dev.ini
    APP_ENV=local python ...      # loads config/local.ini (default)
"""

import os
import configparser

# ─── Determine active environment ─────────────────────────────────────────────
APP_ENV = os.environ.get("APP_ENV", "local")  # "local" | "dev"

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_INI_PATH = os.path.join(_PROJECT_ROOT, "config", f"{APP_ENV}.ini")

if not os.path.exists(_INI_PATH):
    raise FileNotFoundError(
        f"Config file not found: {_INI_PATH}\n"
        f"Expected: config/local.ini or config/dev.ini\n"
        f"Current APP_ENV='{APP_ENV}'"
    )

_conf = configparser.ConfigParser()
_conf.read(_INI_PATH)

# ─── Model configurations ──────────────────────────────────────────────────────
class ModelConfig:
    DINOV2_MODEL_NAME   = _conf.get("models", "dinov2_model_name")
    SAM_MODEL_TYPE      = _conf.get("models", "sam_model_type")
    SAM_CHECKPOINT_PATH = _conf.get("models", "sam_checkpoint_path")
    CLIP_MODEL_NAME     = _conf.get("models", "clip_model_name")
    # 微调 DINOv2 权重路径：相对路径按项目根目录解析；留空则不加载
    _raw_dino_ckpt = _conf.get("models", "dinov2_finetuned_checkpoint", fallback="").strip()
    DINOV2_FINETUNED_CHECKPOINT: str = (
        os.path.join(_PROJECT_ROOT, _raw_dino_ckpt) if _raw_dino_ckpt and not os.path.isabs(_raw_dino_ckpt)
        else _raw_dino_ckpt
    )


# ─── Pipeline runtime parameters ─────────────────────────────────────────────
class PipelineConfig:
    SAM_MODEL_TYPE       = _conf.get("models",    "sam_model_type")
    ATTENTION_THRESHOLD  = _conf.getfloat("pipeline", "attention_threshold")
    MIN_MASK_AREA        = _conf.getint("pipeline",   "min_mask_area")
    NUM_PROMPTS          = _conf.getint("pipeline",   "num_prompts")
    CONFIDENCE_THRESHOLD = _conf.getfloat("pipeline", "confidence_threshold")
    DEVICE               = _conf.get("pipeline",      "device")
    IMAGE_SIZE           = 224
    DINOV2_IMAGE_SIZE    = 224

# ─── Server settings ─────────────────────────────────────────────────────────
class ServerConfig:
    HOST    = _conf.get("server",    "host")
    PORT    = _conf.getint("server", "port")
    WORKERS = _conf.getint("server", "workers")

# ─── Default candidate classes ────────────────────────────────────────────────
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
