#!/usr/bin/env python3
"""
GRPO configuration for math alignment experiments.
"""

from dataclasses import dataclass
from typing import Optional, List, Literal
from torch.optim import AdamW


@dataclass
class GRPOConfig:
    """GRPO configuration."""
    
    # Model configuration - use shared config path
    model_id: str = "/root/autodl-tmp/Qwen/Qwen2.5-Math-1.5B"
    
    # GRPO training hyperparameters
    n_grpo_steps: int = 200
    learning_rate: float = 1e-5
    advantage_eps: float = 1e-6
    rollout_batch_size: int = 256
    group_size: int = 8
    sampling_temperature: float = 1.0
    sampling_min_tokens: int = 4  # As in Expiter, disallow empty string responses
    sampling_max_tokens: int = 1024
    epochs_per_rollout_batch: int = 1  # On-policy
    train_batch_size: int = 256  # On-policy
    gradient_accumulation_steps: int = 128  # microbatch size is 2, will fit on H100
    gpu_memory_utilization: float = 0.85
    
    # Loss configuration
    loss_type: Literal[
        "no_baseline",
        "reinforce_with_baseline",
        "grpo_clip",
    ] = "reinforce_with_baseline"
    use_std_normalization: bool = True
    
    # Training optimization
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95
    
    # Device and precision
    device: str = "cuda:0"
    torch_dtype: str = "bfloat16"
    use_flash_attention: bool = True
    
    # Evaluation and logging
    eval_every: int = 5  # Evaluate every N steps
    eval_batch_size: int = 2
    log_every: int = 10  # Log training metrics every N steps
    
    # Seed for reproducibility
    seed: int = 42
    
    # Wandb configuration
    wandb_enabled: bool = False
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    
    # Sampling parameters
    sampling_top_p: float = 1.0
    sampling_stop: List[str] = None  # Will be set to ["</answer>"]
    
    def __post_init__(self):
        # Set default sampling stop tokens
        if self.sampling_stop is None:
            self.sampling_stop = ["</answer>"]
        
        # Validate configuration
        assert self.train_batch_size % self.gradient_accumulation_steps == 0, (
            "train_batch_size must be divisible by gradient_accumulation_steps"
        )
        assert self.rollout_batch_size % self.group_size == 0, (
            "rollout_batch_size must be divisible by group_size"
        )
        assert self.train_batch_size >= self.group_size, (
            "train_batch_size must be greater than or equal to group_size"
        )
        
        # Compute derived values
        self.micro_train_batch_size = self.train_batch_size // self.gradient_accumulation_steps
        self.n_prompts_per_rollout_batch = self.rollout_batch_size // self.group_size
        self.n_microbatches_per_rollout_batch = self.rollout_batch_size // self.micro_train_batch_size
    
    def get_optimizer(self, model):
        """Get optimizer with configured parameters."""
        return AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(self.beta1, self.beta2),
        )


# Default GRPO configuration
DEFAULT_GRPO_CONFIG = GRPOConfig()
