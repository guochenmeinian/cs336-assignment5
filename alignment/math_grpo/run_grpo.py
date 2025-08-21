#!/usr/bin/env python3
"""
GRPO training script for math alignment experiments.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from .config import GRPOConfig, DEFAULT_GRPO_CONFIG
from ..shared.math_data_utils import load_math_data
from ..shared.config import MATH_TRAIN_PATH, MATH_VALIDATION_PATH
from ..shared.drgrpo_grader import r1_zero_reward_fn


def main():
    """Main function for GRPO training."""
    parser = argparse.ArgumentParser(description="Train a model using GRPO on MATH dataset")
    parser.add_argument("--config", type=str, help="Path to config file (optional)")
    parser.add_argument("--model_id", type=str, help="Model ID to use")
    parser.add_argument("--n_steps", type=int, help="Number of GRPO steps")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--rollout_batch_size", type=int, help="Rollout batch size")
    parser.add_argument("--group_size", type=int, help="Group size for normalization")
    parser.add_argument("--loss_type", type=str, choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"], 
                       help="Loss type to use")
    parser.add_argument("--wandb_enabled", action="store_true", help="Enable wandb logging")
    parser.add_argument("--output_dir", type=str, default="./models/grpo", help="Output directory for model")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        # Load from file
        import json
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        cfg = GRPOConfig(**config_dict)
    else:
        cfg = DEFAULT_GRPO_CONFIG
    
    # Override with command line arguments
    if args.model_id:
        cfg.model_id = args.model_id
    if args.n_steps:
        cfg.n_grpo_steps = args.n_steps
    if args.lr:
        cfg.learning_rate = args.lr
    if args.rollout_batch_size:
        cfg.rollout_batch_size = args.rollout_batch_size
    if args.group_size:
        cfg.group_size = args.group_size
    if args.loss_type:
        cfg.loss_type = args.loss_type
    if args.wandb_enabled:
        cfg.wandb_enabled = True
    
    print("GRPO Configuration:")
    print(f"  Model ID: {cfg.model_id}")
    print(f"  Number of steps: {cfg.n_grpo_steps}")
    print(f"  Learning rate: {cfg.learning_rate}")
    print(f"  Rollout batch size: {cfg.rollout_batch_size}")
    print(f"  Group size: {cfg.group_size}")
    print(f"  Loss type: {cfg.loss_type}")
    print(f"  Wandb enabled: {cfg.wandb_enabled}")
    
    # Load data
    print("Loading MATH dataset...")
    train_data = load_math_data(str(MATH_TRAIN_PATH))
    validation_data = load_math_data(str(MATH_VALIDATION_PATH))
    
    print(f"Loaded {len(train_data)} training examples and {len(validation_data)} validation examples")
    
    # Initialize trainer
    print("Initializing GRPO trainer...")
    from .trainer import GRPOTrainer
    trainer = GRPOTrainer(cfg)
    
    # Start training
    print("Starting GRPO training...")
    trainer.train(train_data, validation_data, r1_zero_reward_fn)
    
    # Save model
    print(f"Saving model to {args.output_dir}...")
    trainer.save(args.output_dir)
    
    print("GRPO training completed successfully!")


if __name__ == "__main__":
    main()
