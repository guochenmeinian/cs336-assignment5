#!/usr/bin/env python3
"""
SFT-specific configuration for math alignment experiments.
Contains only training parameters, model and data paths are in shared config.
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class SFTConfig:
    """SFT training configuration - only training parameters."""
    
    # Training hyperparameters
    lr: float = 2e-5
    batch_size: int = 2
    grad_accum: int = 8
    max_steps: int = 2000
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
    
    # Wandb configuration (inherited from shared config)
    wandb_enabled: bool = True
    wandb_project: Optional[str] = None  # Will use shared config default
    wandb_name: Optional[str] = None     # Will use shared config default
    wandb_tags: Optional[List] = None  # Will use shared config default


# Default SFT configuration
DEFAULT_SFT_CONFIG = SFTConfig()

# Training-specific constants
LOG_EVERY = 50
EVAL_BATCH_SIZE = 2
MAX_NEW_TOKENS = 128
