#!/usr/bin/env python3
"""
SFT-specific configuration for math alignment experiments.
Contains only training parameters, all other configs are in shared config.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SFTConfig:
    """SFT training configuration - training parameters + experiment-specific wandb config."""
    
    # Training hyperparameters
    dataset_size: int = 6792 # full dataset size
    lr: float = 2e-5
    batch_size: int = 2
    grad_accum: int = 8
    max_steps: int = 500
    max_grad_norm: float = 1.0
    
    # Device and precision
    bf16: bool = True
    amp: bool = False
    device: str = "cuda:0"
    
    # Memory optimization
    use_flash_attention: bool = True
    torch_dtype: str = "bfloat16"  # "bfloat16", "float16", "float32"
    
    # Random seed
    seed: int = 42
    
    # Experiment-specific wandb configuration
    wandb_enabled: bool = False  # 改为False，避免wandb连接问题
    wandb_name: str = "sft_qwen_math_15b"
    wandb_tags: list = None  # 默认None，会在__post_init__中设置
    
    def __post_init__(self):
        if self.wandb_tags is None:
            self.wandb_tags = ["sft", "qwen", "math", "15b"]


# Default SFT configuration
DEFAULT_SFT_CONFIG = SFTConfig()

# Training-specific constants
LOG_EVERY = 50
EVAL_BATCH_SIZE = 2
MAX_NEW_TOKENS = 128
