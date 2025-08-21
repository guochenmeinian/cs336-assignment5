#!/usr/bin/env python3
"""
Expert Iteration configuration for math alignment experiments.
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class EIConfig:
    """Expert Iteration configuration."""
    
    # Model configuration - use shared config path
    model_id: str = "/root/autodl-tmp/Qwen/Qwen2.5-Math-1.5B"
    
    # Training hyperparameters
    lr: float = 2e-5
    batch_size: int = 2
    grad_accum: int = 8
    max_steps: int = 1000  # Per EI iteration
    max_grad_norm: float = 1.0
    
    # Device and precision
    bf16: bool = True
    amp: bool = False
    device: str = "cuda:0"
    
    # Memory optimization
    use_flash_attention: bool = True
    torch_dtype: str = "bfloat16"
    
    # Expert Iteration specific
    n_ei_steps: int = 5  # Number of EI iterations per experiment
    G: int = 8  # Number of samples per question
    sampling_temperature: float = 1.0
    sampling_max_tokens: int = 1024
    sampling_min_tokens: int = 4  # Prevent empty strings
    seed: int = 42
    
    # vLLM sampling parameters
    sampling_top_p: float = 1.0
    sampling_stop: List[str] = None  # Will be set to ["</answer>"]
    
    # Batch size for this experiment: [512, 1024, 2048]
    experiment_batch_size: int = 512
    
    # Wandb configuration
    wandb_enabled: bool = False
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_tags: Optional[List[str]] = None


# Default EI configuration
DEFAULT_EI_CONFIG = EIConfig()

# Set default values for lists
if DEFAULT_EI_CONFIG.sampling_stop is None:
    DEFAULT_EI_CONFIG.sampling_stop = ["</answer>"]

# Training constants
LOG_EVERY = 50
EVAL_BATCH_SIZE = 2
MAX_NEW_TOKENS = 128
