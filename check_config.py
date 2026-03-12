import sys
import os
sys.path.append(os.getcwd())
from src.config import ModelConfig
print(f"ModelConfig.SAM_CHECKPOINT_PATH: {ModelConfig.SAM_CHECKPOINT_PATH}")
